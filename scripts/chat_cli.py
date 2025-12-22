"""
New and upgraded chat mode because a lot of the code has changed since the last one.

Intended to be run single GPU only atm:
python -m scripts.chat_cli -i mid
"""
import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from contextlib import nullcontext
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|mid|rl")
parser.add_argument('-c', '--checkpoint-dir', type=str, default=None, help='Direct path to checkpoint directory')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('-r', '--repetition-penalty', type=float, default=1.2, help='Repetition penalty (1.0=off, 1.2=default)')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
args = parser.parse_args()

# Init the model and tokenizer

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
if args.checkpoint_dir:
    import os
    from nanochat.checkpoint_manager import load_checkpoint, find_last_step
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import RustBPETokenizer
    # Load from weights/ subdir
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
    # Load from tokenizer/ subdir
    tokenizer_dir = os.path.join(args.checkpoint_dir, "tokenizer")
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
else:
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# Create Engine for efficient generation
bos = tokenizer.get_bos_token_id()
engine = Engine(model, tokenizer)

print("\nText Completion Mode")
print("-" * 50)
print("Type 'quit' or 'exit' to end")
print("-" * 50)

while True:
    if args.prompt:
        user_input = args.prompt
    else:
        try:
            user_input = input("\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break

    if not user_input:
        continue

    # Tokenize prompt
    tokens = [bos] + tokenizer.encode(user_input)

    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": 256,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
    }
    print("\nCompletion: ", end="", flush=True)
    with autocast_ctx:
        for token_column, _ in engine.generate(tokens, **generate_kwargs):
            token = token_column[0]
            print(tokenizer.decode([token]), end="", flush=True)
    print("\n")

    if args.prompt:
        break
