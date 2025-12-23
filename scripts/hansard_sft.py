"""
Instruction-tune a Hansard base model using Q&A pairs extracted from parliamentary proceedings.

The UK Hansard data naturally contains Q&A structure:
- Lords/MPs ask questions: "To ask Her Majesty's Government..."
- Ministers respond with answers

This script:
1. Extracts Q&A pairs from the Hansard dataset
2. Formats them as user/assistant conversations
3. Trains the model on them

Usage:
    python -m scripts.hansard_sft
    torchrun --standalone --nproc_per_node=8 -m scripts.hansard_sft

Hyperparameters: Same as chat_sft.py (proven defaults from nanochat codebase).
"""

import os
import re
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import torch
import torch.distributed as dist
from contextlib import nullcontext
from datasets import load_dataset

from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint, find_largest_model, find_last_step
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

# -----------------------------------------------------------------------------
# Hyperparameters (same as chat_sft.py)
run = "dummy"
model_tag = None
step = None
device_type = ""
dtype = "bfloat16"
device_batch_size = 4
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
max_qa_pairs = -1  # limit Q&A pairs for debugging (-1 = all)
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

if model_tag is None:
    model_tag = find_largest_model(checkpoints_dir)
    print0(f"Auto-detected model: {model_tag}")
checkpoint_dir = os.path.join(checkpoints_dir, model_tag)

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

# -----------------------------------------------------------------------------
# Extract Q&A pairs from Hansard

def extract_qa_pairs(max_pairs=-1):
    """
    Extract Q&A pairs from Hansard parliamentary proceedings.
    Pattern: Someone asks "To ask Her Majesty's Government..." then someone responds.
    """
    print0("Loading Hansard dataset...")
    dataset = load_dataset("common-pile/uk_hansard", split="train", streaming=True)
    
    qa_pairs = []
    # Pattern: Name: To ask His/Her Majesty's Government [question]
    question_pattern = re.compile(
        r'^([A-Za-z\s\-\'\.]+?):\s*(To ask (His|Her) Majesty\'s Government|asked (His|Her) Majesty\'s Government)',
        re.MULTILINE
    )
    
    for doc_idx, example in enumerate(dataset):
        if max_pairs > 0 and len(qa_pairs) >= max_pairs:
            break
        if doc_idx % 1000 == 0:
            print0(f"Processed {doc_idx} docs, found {len(qa_pairs)} Q&A pairs...")
        
        text = example["text"]
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        i = 0
        while i < len(paragraphs) - 1:
            para = paragraphs[i]
            
            # Check if this paragraph contains a question
            if any(phrase in para for phrase in [
                "To ask Her Majesty's Government", "To ask His Majesty's Government",
                "asked Her Majesty's Government", "asked His Majesty's Government"
            ]):
                # This is a question - next paragraph(s) should be the answer
                question = para
                
                # Get the answer (next paragraph that starts with a name)
                answer_parts = []
                j = i + 1
                while j < len(paragraphs):
                    next_para = paragraphs[j]
                    # Check if it looks like an answer (starts with a name followed by colon)
                    if re.match(r'^[A-Za-z\s\-\'\.\,]+?:', next_para):
                        answer_parts.append(next_para)
                        break
                    j += 1
                
                if answer_parts:
                    answer = ' '.join(answer_parts)
                    qa_pairs.append({
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer}
                        ]
                    })
                    i = j + 1
                    continue
            i += 1
    
    print0(f"Extracted {len(qa_pairs)} Q&A pairs")
    return qa_pairs

print0("Extracting Q&A pairs from Hansard...")
all_qa_pairs = extract_qa_pairs(max_pairs=max_qa_pairs)

if len(all_qa_pairs) == 0:
    raise ValueError("No Q&A pairs extracted. Check the parsing logic.")

# Split train/val
val_size = max(1, int(len(all_qa_pairs) * val_ratio))
train_qa = all_qa_pairs[:-val_size]
val_qa = all_qa_pairs[-val_size:]
print0(f"Train: {len(train_qa)}, Val: {len(val_qa)}")

# -----------------------------------------------------------------------------
# DataLoader

def sft_data_generator(qa_list, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    
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
    num_iterations = max(1, (len(train_qa) // target_examples_per_step) * num_epochs)
print0(f"Iterations: {num_iterations}")

train_loader = sft_data_generator(train_qa, device_batch_size)
build_val_loader = lambda: sft_data_generator(val_qa, device_batch_size)

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

    # Validation
    if last_step or step_num % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        losses = []
        actual_eval_steps = max(1, min(eval_steps, len(val_qa) // device_batch_size))
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

    # Training
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
    save_checkpoint(output_dir, step_num, orig_model.state_dict(), None, {
        "step": step_num,
        "val_loss": val_loss,
        "model_config": model_config_kwargs,
        "num_qa_pairs": len(train_qa),
    })
    print(f"Saved to {output_dir}")

from nanochat.report import get_report
get_report().log(section="Hansard SFT", data=[user_config, {
    "Q&A pairs": len(train_qa),
    "Iterations": num_iterations,
    "Final train loss": train_loss_item,
    "Final val loss": val_loss,
}])

wandb_run.finish()
compute_cleanup()
