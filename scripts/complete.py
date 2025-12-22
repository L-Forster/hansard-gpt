"""
Text completion for base models.
Usage: python -m scripts.complete -c hansard_model -p "The Prime Minister said"
"""
import os
import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_checkpoint, find_last_step
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer
from nanochat.engine import Engine
from contextlib import nullcontext

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint-dir', type=str, required=True, help='Model directory with weights/ and tokenizer/ subdirs')
parser.add_argument('-p', '--prompt', type=str, default='')
parser.add_argument('-s', '--step', type=int, default=None)
parser.add_argument('-n', '--max-tokens', type=int, default=128)
parser.add_argument('-t', '--temperature', type=float, default=0.7)
parser.add_argument('-k', '--top-k', type=int, default=50)
parser.add_argument('-r', '--repetition-penalty', type=float, default=1.2, help='Repetition penalty (1.0=off)')
args = parser.parse_args()

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

# Load model from weights/ subdir
weights_dir = os.path.join(args.checkpoint_dir, "weights")
step = args.step if args.step else find_last_step(weights_dir)
model_data, _, meta = load_checkpoint(weights_dir, step, device)
if device.type in {"cpu", "mps"}:
    model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}
model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
with torch.device("meta"):
    model = GPT(GPTConfig(**meta["model_config"]))
model.to_empty(device=device)
model.init_weights()
model.load_state_dict(model_data, strict=True, assign=True)
model.eval()

# Load tokenizer from tokenizer/ subdir
tokenizer_dir = os.path.join(args.checkpoint_dir, "tokenizer")
tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

engine = Engine(model, tokenizer)

print("Base Model Completion (type 'quit' to exit)")
print("-" * 50)

while True:
    if args.prompt:
        prompt = args.prompt
    else:
        try:
            prompt = input("\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
    
    if prompt.lower() in ['quit', 'exit']:
        break
    if not prompt:
        continue

    tokens = tokenizer(prompt, prepend="<|bos|>")
    print("\nCompletion: ", end="", flush=True)
    with autocast_ctx:
        for token_col, _ in engine.generate(tokens, num_samples=1, max_tokens=args.max_tokens, 
                                             temperature=args.temperature, top_k=args.top_k,
                                             repetition_penalty=args.repetition_penalty):
            print(tokenizer.decode([token_col[0]]), end="", flush=True)
    print("\n")
    
    if args.prompt:
        break

