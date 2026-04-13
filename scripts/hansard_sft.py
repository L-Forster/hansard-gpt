"""
Fine-tune a Hansard base model on Enoch Powell paragraphs using plain LM loss.

This script:
1. Loads Powell-only parquet shards from data/hansard_powell_sft/
2. Tokenizes the text with the Hansard tokenizer
3. Continues LM training from the pretrained Hansard checkpoint

Usage:
    python -m scripts.hansard_sft
    torchrun --standalone --nproc_per_node=8 -m scripts.hansard_sft

Hyperparameters: Same as chat_sft.py (proven defaults from nanochat codebase).
"""

import math
import os
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
from nanochat.tokenizer import get_tokenizer

# -----------------------------------------------------------------------------
# Hyperparameters (same as chat_sft.py)
run = "dummy"
model_tag = "d12"
step = None
device_type = ""
dtype = "bfloat16"
device_batch_size = 4
max_seq_len = -1
num_epochs = 1
num_iterations = -1
target_examples_per_step = 32
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02
eval_every = 100
eval_steps = 50
val_ratio = 0.05
powell_data_dir = ""
tokenizer_batch_size = 256
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
checkpoints_dir = os.path.join(project_dir, "checkpoints")

if not os.path.exists(checkpoints_dir):
    raise FileNotFoundError(f"No checkpoints directory: {checkpoints_dir}")

checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
if not os.path.isdir(checkpoint_dir):
    raise FileNotFoundError(
        f"No pretrained checkpoint directory found at {checkpoint_dir}. Run `bash train_hansard.sh` first or override --model_tag."
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

tokenizer = get_tokenizer(name="hansard")
assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
orig_model = model
sequence_len = model_config_kwargs["sequence_len"] if max_seq_len == -1 else max_seq_len
if sequence_len > model_config_kwargs["sequence_len"]:
    raise ValueError(
        f"max_seq_len={sequence_len} exceeds pretrained model context {model_config_kwargs['sequence_len']}"
    )

# -----------------------------------------------------------------------------
# Load Powell text

def load_powell_documents(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"No Powell data directory found at {data_dir}. Run `python -m nanochat.dataset --hansard` first."
        )
    parquet_paths = list_parquet_files(data_dir=data_dir)
    if not parquet_paths:
        raise FileNotFoundError(
            f"No Powell parquet files found in {data_dir}. Run `python -m nanochat.dataset --hansard` first."
        )

    documents = []
    for path in parquet_paths:
        pf = pq.ParquetFile(path)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=["text"])
            documents.extend(text.strip() for text in rg.column("text").to_pylist() if text and text.strip())
    if not documents:
        raise ValueError(f"No Powell text rows found in {data_dir}")
    return documents

def flatten_tokenized_documents(documents):
    bos_token = tokenizer.get_bos_token_id()
    token_ids = []
    for start in range(0, len(documents), tokenizer_batch_size):
        batch = documents[start:start + tokenizer_batch_size]
        tokenized_batch = tokenizer.encode(batch, prepend=bos_token)
        for ids in tokenized_batch:
            token_ids.extend(ids)
    if len(token_ids) <= sequence_len:
        raise ValueError(
            f"Need more than {sequence_len:,} tokens to form a full sequence, got {len(token_ids):,}"
        )
    return torch.tensor(token_ids, dtype=torch.long)

powell_data_dir = powell_data_dir or HANSARD_POWELL_SFT_DIR
print0(f"Loading Powell data from {powell_data_dir}")
all_documents = load_powell_documents(powell_data_dir)

if len(all_documents) < 2:
    raise ValueError("Need at least two Powell documents to create train/val splits.")

val_size = max(1, int(len(all_documents) * val_ratio))
if val_size >= len(all_documents):
    val_size = len(all_documents) - 1
train_documents = all_documents[:-val_size]
val_documents = all_documents[-val_size:]
print0(f"Train documents: {len(train_documents):,}, Val documents: {len(val_documents):,}")

train_tokens = flatten_tokenized_documents(train_documents)
val_tokens = flatten_tokenized_documents(val_documents)
train_sequences = (train_tokens.numel() - 1) // sequence_len
val_sequences = (val_tokens.numel() - 1) // sequence_len
print0(f"Train tokens: {train_tokens.numel():,}, Val tokens: {val_tokens.numel():,}")
print0(f"Train sequences: {train_sequences:,}, Val sequences: {val_sequences:,}")

# -----------------------------------------------------------------------------
# DataLoader

def sft_data_generator(qa_list, batch_size):
    # Use assistant_end as pad token if available, otherwise fallback to BOS
    try:
        pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    except (KeyError, ValueError):
        pad_token_id = tokenizer.get_bos_token_id()
    
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :n-1] = row_targets
        return inputs.to(device), targets.to(device)
    
    batch = []
    while True:
        for i in range(ddp_rank, len(qa_list), ddp_world_size):
            doc = qa_list[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

examples_per_step = device_batch_size * ddp_world_size
assert target_examples_per_step % examples_per_step == 0
grad_accum_steps = target_examples_per_step // examples_per_step
print0(f"Grad accum steps: {grad_accum_steps}")

if num_iterations == -1:
    num_iterations = max(1, math.ceil(train_sequences / target_examples_per_step) * num_epochs)
print0(f"Iterations: {num_iterations}")

train_loader = lm_data_generator(train_tokens, device_batch_size)
build_val_loader = lambda: lm_data_generator(val_tokens, device_batch_size)

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
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"]

def get_lr_multiplier(it):
    return 1.0 - it / num_iterations

# -----------------------------------------------------------------------------
# Training

print0("Starting training...")
step_num = 0
val_loss = float("inf")
train_loss_item = float("inf")

for step_num in range(num_iterations):
    last_step = step_num == num_iterations - 1

    if last_step or step_num % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        losses = []
        actual_eval_steps = max(1, min(eval_steps, math.ceil(val_sequences / (device_batch_size * ddp_world_size))))
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
        num_tokens += train_targets.numel()

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
    save_checkpoint(output_dir, step_num, orig_model.state_dict(), None, {
        "step": step_num,
        "val_loss": val_loss,
        "model_config": model_config_kwargs,
        "powell_data_dir": powell_data_dir,
        "train_documents": len(train_documents),
        "val_documents": len(val_documents),
        "train_tokens": int(train_tokens.numel()),
        "val_tokens": int(val_tokens.numel()),
    })
    print(f"Saved to {output_dir}")

from nanochat.report import get_report
get_report().log(section="Hansard SFT", data=[user_config, {
    "Train documents": len(train_documents),
    "Val documents": len(val_documents),
    "Train tokens": int(train_tokens.numel()),
    "Val tokens": int(val_tokens.numel()),
    "Iterations": num_iterations,
    "Final train loss": train_loss_item,
    "Final val loss": val_loss,
}])

wandb_run.finish()
compute_cleanup()
