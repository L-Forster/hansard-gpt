"""
Fine-tune a Hansard base model on Enoch Powell QA pairs using chat-style SFT loss.

This script:
1. Loads Powell question/answer parquet shards from data/hansard_powell_sft/
2. Renders them as user/assistant conversations
3. Continues training from the pretrained Hansard checkpoint

Usage:
    python -m scripts.hansard_sft
    torchrun --standalone --nproc_per_node=8 -m scripts.hansard_sft

Hyperparameters: Same as chat_sft.py (proven defaults from nanochat codebase).
"""

import json
import math
import os
import random
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from contextlib import nullcontext

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import wandb

from nanochat.checkpoint_manager import find_last_step, load_checkpoint, save_checkpoint
from nanochat.common import compute_cleanup, compute_init, print0, DummyWandb, autodetect_device_type
from nanochat.dataset import HANSARD_POWELL_SFT_DIR, list_parquet_files
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer, get_tokenizer

# -----------------------------------------------------------------------------
# Hyperparameters (same as chat_sft.py)
run = "dummy"
model_tag = "d12"
step = None
device_type = ""
dtype = "bfloat16"
device_batch_size = 4
max_seq_len = -1
num_epochs = 2
num_iterations = -1
target_examples_per_step = 32
unembedding_lr = 0.001
embedding_lr = 0.05
matrix_lr = 0.005
weight_decay = 0.1
init_lr_frac = 1.0
eval_every = 10
eval_steps = 50
val_ratio = 0.05
powell_data_dir = ""
checkpoint_dir = ""
tokenizer_dir = ""
split_seed = 42
# CLI override
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# wandb
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-hansard-sft", name=run, config=user_config)

# -----------------------------------------------------------------------------
# Load model

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_existing_dir(label, explicit_path, candidates):
    if explicit_path:
        if not os.path.isdir(explicit_path):
            raise FileNotFoundError(f"{label} directory does not exist: {explicit_path}")
        return explicit_path
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    joined = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Unable to find {label} directory. Checked:\n{joined}")


checkpoint_dir = resolve_existing_dir(
    "checkpoint",
    checkpoint_dir,
    [
        os.path.join(project_dir, "checkpoints", model_tag),
        os.path.join(project_dir, "hansard", "weights"),
    ],
)

if step is None:
    step = find_last_step(checkpoint_dir)
    print0(f"Auto-detected step: {step}")

print0(f"Loading model from {checkpoint_dir} step {step}")
model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
if device.type in {"cpu", "mps"}:
    model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}

model_config_kwargs = meta_data["model_config"]
model_config = GPTConfig(**model_config_kwargs)
with torch.device("meta"):
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()
model.load_state_dict(model_data, strict=True, assign=True)
del model_data

if tokenizer_dir:
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
else:
    candidate_tokenizer_dirs = [
        os.path.join(project_dir, "data", "tokenizer_hansard"),
        os.path.join(project_dir, "hansard", "tokenizer"),
    ]
    tokenizer = None
    for candidate in candidate_tokenizer_dirs:
        if os.path.isdir(candidate):
            tokenizer = RustBPETokenizer.from_directory(candidate)
            tokenizer_dir = candidate
            break
    if tokenizer is None:
        tokenizer = get_tokenizer(name="hansard")
        tokenizer_dir = os.path.join(project_dir, "data", "tokenizer_hansard")
print0(f"Using tokenizer from {tokenizer_dir}")
assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
orig_model = model
sequence_len = model_config_kwargs["sequence_len"] if max_seq_len == -1 else max_seq_len
if sequence_len > model_config_kwargs["sequence_len"]:
    raise ValueError(
        f"max_seq_len={sequence_len} exceeds pretrained model context {model_config_kwargs['sequence_len']}"
    )

# -----------------------------------------------------------------------------
# Load Powell SFT data

def load_powell_conversations(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"No Powell data directory found at {data_dir}. Run `python -m nanochat.dataset --hansard` first."
        )
    parquet_paths = list_parquet_files(data_dir=data_dir)
    if not parquet_paths:
        raise FileNotFoundError(
            f"No Powell parquet files found in {data_dir}. Run `python -m nanochat.dataset --hansard` first."
        )

    conversations = []
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            schema_names = set(pf.schema_arrow.names)
            if "messages" in schema_names:
                rg = pf.read_row_group(rg_idx, columns=["messages"])
                for row in rg.column("messages").to_pylist():
                    if not row:
                        continue
                    messages = json.loads(row)
                    if messages:
                        conversations.append({"messages": messages})
            elif {"question", "answer"}.issubset(schema_names):
                rg = pf.read_row_group(rg_idx, columns=["question", "answer"])
                questions = rg.column("question").to_pylist()
                answers = rg.column("answer").to_pylist()
                for question, answer in zip(questions, answers):
                    if question and answer:
                        conversations.append({
                            "messages": [
                                {"role": "user", "content": question.strip()},
                                {"role": "assistant", "content": answer.strip()},
                            ]
                        })
            else:
                raise ValueError(
                    f"Powell SFT parquet {path} must contain 'messages' or ('question', 'answer') columns."
                )
    if not conversations:
        raise ValueError(f"No Powell SFT rows found in {data_dir}")
    return conversations

powell_data_dir = powell_data_dir or HANSARD_POWELL_SFT_DIR
print0(f"Loading Powell data from {powell_data_dir}")
all_conversations = load_powell_conversations(powell_data_dir)

if len(all_conversations) < 2:
    raise ValueError("Need at least two Powell conversations to create train/val splits.")

rng = random.Random(split_seed)
rng.shuffle(all_conversations)

val_size = max(1, int(len(all_conversations) * val_ratio))
if val_size >= len(all_conversations):
    val_size = len(all_conversations) - 1
train_conversations = all_conversations[:-val_size]
val_conversations = all_conversations[-val_size:]
print0(f"Train conversations: {len(train_conversations):,}, Val conversations: {len(val_conversations):,}")

def estimate_supervised_tokens(conversations):
    total = 0
    for conversation in conversations:
        ids, mask = tokenizer.render_conversation(conversation, max_tokens=sequence_len)
        total += sum(mask[1:])
    return total

train_supervised_tokens = estimate_supervised_tokens(train_conversations)
val_supervised_tokens = estimate_supervised_tokens(val_conversations)
print0(f"Train supervised tokens: {train_supervised_tokens:,}, Val supervised tokens: {val_supervised_tokens:,}")

# -----------------------------------------------------------------------------
# DataLoader

def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special_safe("<|assistant_end|>")
    if pad_token_id is None:
        pad_token_id = tokenizer.get_bos_token_id()

    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, _ in batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        for i, (ids, mask) in enumerate(batch):
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :len(ids) - 1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :len(ids) - 1] = row_targets
        return inputs.to(device), targets.to(device)

    while True:
        batch = []
        for example_idx in range(ddp_rank, len(dataset), ddp_world_size):
            ids, mask = tokenizer.render_conversation(dataset[example_idx], max_tokens=sequence_len)
            if len(ids) < 2:
                continue
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

examples_per_step = device_batch_size * ddp_world_size
assert target_examples_per_step % examples_per_step == 0
grad_accum_steps = target_examples_per_step // examples_per_step
print0(f"Grad accum steps: {grad_accum_steps}")

if num_iterations == -1:
    num_iterations = max(1, math.ceil(len(train_conversations) / target_examples_per_step) * num_epochs)
print0(f"Iterations: {num_iterations}")

train_loader = sft_data_generator(train_conversations, device_batch_size)
build_val_loader = lambda: sft_data_generator(val_conversations, device_batch_size)

# -----------------------------------------------------------------------------
# Optimizer

optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

warmup_steps = max(1, num_iterations // 10)

def get_lr_multiplier(it):
    if it < warmup_steps:
        return init_lr_frac + (1.0 - init_lr_frac) * it / warmup_steps
    progress = (it - warmup_steps) / max(1, num_iterations - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

# -----------------------------------------------------------------------------
# Training

print0("Starting training...")
step_num = 0
val_loss = float("inf")
best_val_loss = float("inf")
best_model_state = None
train_loss_item = float("inf")

for step_num in range(num_iterations):
    last_step = step_num == num_iterations - 1

    if last_step or step_num % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        losses = []
        actual_eval_steps = max(1, min(eval_steps, math.ceil(len(val_conversations) / (device_batch_size * ddp_world_size))))
        for _ in range(actual_eval_steps):
            val_inputs, val_targets = next(val_loader)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        val_loss = torch.stack(losses).mean()
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss = val_loss.item()
        print0(f"Step {step_num:05d} | Val loss: {val_loss:.6f}")
        wandb_run.log({"step": step_num, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.clone() for k, v in orig_model.state_dict().items()}
            print0(f"  -> New best val loss: {best_val_loss:.6f}")
        model.train()

    if last_step:
        break

    num_tokens = torch.tensor(0, device=device)
    for _ in range(grad_accum_steps):
        train_inputs, train_targets = next(train_loader)
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        train_loss = loss.detach()
        (loss / grad_accum_steps).backward()
        num_tokens += (train_targets >= 0).sum()

    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)

    lrm = get_lr_multiplier(step_num)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    train_loss_item = train_loss.item()
    print0(f"Step {step_num:05d}/{num_iterations:05d} | Train: {train_loss_item:.6f} | lrm: {lrm:.4f}")
    wandb_run.log({"step": step_num, "train_loss": train_loss_item, "lrm": lrm})

# -----------------------------------------------------------------------------
# Save

if master_process:
    output_dir = os.path.join(project_dir, "hansard_sft_checkpoints", model_tag)
    os.makedirs(output_dir, exist_ok=True)
    save_state = best_model_state if best_model_state is not None else orig_model.state_dict()
    print0(f"Saving best checkpoint (val_loss={best_val_loss:.6f})")
    save_checkpoint(output_dir, step_num, save_state, None, {
        "step": step_num,
        "best_val_loss": best_val_loss,
        "final_val_loss": val_loss,
        "model_config": model_config_kwargs,
        "powell_data_dir": powell_data_dir,
        "train_conversations": len(train_conversations),
        "val_conversations": len(val_conversations),
        "train_supervised_tokens": int(train_supervised_tokens),
        "val_supervised_tokens": int(val_supervised_tokens),
    })
    print(f"Saved to {output_dir}")

from nanochat.report import get_report
get_report().log(section="Hansard SFT", data=[user_config, {
    "Train conversations": len(train_conversations),
    "Val conversations": len(val_conversations),
    "Train supervised tokens": int(train_supervised_tokens),
    "Val supervised tokens": int(val_supervised_tokens),
    "Iterations": num_iterations,
    "Final train loss": train_loss_item,
    "Final val loss": val_loss,
}])

wandb_run.finish()
compute_cleanup()
