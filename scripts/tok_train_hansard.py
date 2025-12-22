"""
Train a BPE tokenizer on the UK Hansard parliamentary proceedings dataset.

Features:
- Streams data from HuggingFace Hub (common-pile/uk_hansard)
- Extracts only the 'text' column (ignoring metadata columns)
- Applies NFKC Unicode normalization
- Shuffles documents to mix eras (1918-2016+)
- Uses custom Hansard-specific special tokens
"""
import os
import time
import random
import argparse
import unicodedata
import torch
from datasets import load_dataset
from nanochat.tokenizer import RustBPETokenizer

# -----------------------------------------------------------------------------
# Hansard-specific special tokens
# These are treated as atomic units and not split by the tokenizer

HANSARD_SPECIAL_TOKENS = [
    "<|bos|>",  # Document delimiter
    "<|eos|>",  # End of sequence
    # Chat tokens for instruction tuning
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|output_start|>",
    "<|output_end|>",
    # Parliamentary procedural phrases (appear frequently as atomic units)
    "Her Majesty's Government",
    "My Lords,",
    "To ask Her Majesty's Government",
]

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer on UK Hansard data')
parser.add_argument('--vocab_size', type=int, default=32768, help='Vocabulary size (default: 32768 = 2^15, GPU-friendly)')
parser.add_argument('--max_sentences', type=int, default=-1, help='Maximum sentences to train on (-1 = all, default: -1)')
parser.add_argument('--shuffle_buffer', type=int, default=50_000, help='Shuffle buffer size for sentence mixing (default: 50,000)')
parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling (default: 42)')
parser.add_argument('--output_dir', type=str, default=None, help='Output directory for tokenizer (default: base_dir/tokenizer_hansard)')
args = parser.parse_args()

print("=" * 60)
print("UK Hansard BPE Tokenizer Training")
print("=" * 60)
print(f"vocab_size: {args.vocab_size:,}")
print(f"shuffle_buffer: {args.shuffle_buffer:,}")
print(f"seed: {args.seed}")
print(f"num_special_tokens: {len(HANSARD_SPECIAL_TOKENS)}")
print()

# Set random seed
random.seed(args.seed)

# -----------------------------------------------------------------------------
# Text iterator with NFKC normalization and shuffling

def text_iterator():
    """
    Stream sentences from HuggingFace Hub with:
    1) NFKC Unicode normalization
    2) Split documents into sentences (on '.')
    3) Shuffle buffer for chaotic era mixing
    """
    print("Loading dataset from HuggingFace Hub: common-pile/uk_hansard")
    dataset = load_dataset("common-pile/uk_hansard", split="train", streaming=True)

    buffer = []
    sentence_count = 0

    for example in dataset:
        # Extract only the text column, ignore metadata columns
        text = example["text"]

        # Apply NFKC normalization (unifies em-dashes, ligatures, etc.)
        text = unicodedata.normalize("NFKC", text)

        # Split into sentences on '.' and add period back
        sentences = text.split('.')
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            # Add period back (except for last fragment if it didn't end with period)
            if i < len(sentences) - 1 or text.endswith('.'):
                sentence = sentence + '.'
            buffer.append(sentence)

        # When buffer is full, shuffle and yield
        if len(buffer) >= args.shuffle_buffer:
            random.shuffle(buffer)
            for sent in buffer:
                yield sent
                sentence_count += 1
                if args.max_sentences > 0 and sentence_count >= args.max_sentences:
                    return
            buffer = []

    # Yield remaining sentences in buffer
    if buffer:
        random.shuffle(buffer)
        for sent in buffer:
            yield sent
            sentence_count += 1
            if args.max_sentences > 0 and sentence_count >= args.max_sentences:
                return

    print(f"Processed {sentence_count:,} sentences")

# -----------------------------------------------------------------------------
# Train the tokenizer

print("Starting tokenizer training...")
text_iter = text_iterator()

t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(
    text_iter,
    args.vocab_size,
    special_tokens=HANSARD_SPECIAL_TOKENS
)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk

if args.output_dir:
    tokenizer_dir = args.output_dir
else:
    # Save in project directory: nanochat/data/tokenizer_hansard/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    tokenizer_dir = os.path.join(project_dir, "data", "tokenizer_hansard")

os.makedirs(tokenizer_dir, exist_ok=True)

tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check with parliamentary-style text

test_text = """My Lords, I rise to ask Her Majesty's Government what steps they are taking.
Lord Smith: The noble Lord raises an important point.
Numbers: 123, 4567
Special chars: £1,000—funds allocated."""

encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
print(f"\nSanity check:")
print(f"  Original length: {len(test_text)} chars")
print(f"  Encoded length: {len(encoded)} tokens")
print(f"  Decode matches: {decoded == test_text}")

# Verify special tokens work
bos_id = tokenizer.encode_special("<|bos|>")
print(f"  BOS token ID: {bos_id}")
hmg_id = tokenizer.encode_special("Her Majesty's Government")
print(f"  'Her Majesty's Government' token ID: {hmg_id}")

# -----------------------------------------------------------------------------
# Cache token bytes mapping for bits-per-byte evaluation

vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id]
    if token_str in special_set:
        token_bytes.append(0)  # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8"))
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# -----------------------------------------------------------------------------
# Export human-readable vocabulary file (so you can inspect the tokens!)

vocab_path = os.path.join(tokenizer_dir, "vocab.txt")
with open(vocab_path, "w", encoding="utf-8") as f:
    f.write(f"# UK Hansard BPE Vocabulary\n")
    f.write(f"# Total tokens: {vocab_size}\n")
    f.write(f"# Special tokens: {len(special_set)}\n")
    f.write(f"# Format: token_id<TAB>token_string<TAB>num_bytes\n")
    f.write(f"#\n")
    for token_id in range(vocab_size):
        token_str = token_strings[token_id]
        num_bytes = token_bytes[token_id].item()
        # Escape special characters for readability
        display_str = repr(token_str)[1:-1]  # Remove quotes from repr
        f.write(f"{token_id}\t{display_str}\t{num_bytes}\n")
print(f"Saved human-readable vocab to {vocab_path}")

# -----------------------------------------------------------------------------
# Log to report

from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Hansard Tokenizer Training", data=[
    vars(args),
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {"special_tokens": HANSARD_SPECIAL_TOKENS},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])

print("\n" + "=" * 60)
print("Training complete!")
print(f"Tokenizer saved to: {tokenizer_dir}")
print(f"Vocab size: {vocab_size:,}")
print("=" * 60)

