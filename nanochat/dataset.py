"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import shutil
import re
import time
import json
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANSARD_DATA_DIR = os.path.join(_project_dir, "data", "hansard_data")
HANSARD_POWELL_SFT_DIR = os.path.join(_project_dir, "data", "hansard_powell_sft")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None):
    """ Looks into a data dir and returns full paths to all parquet files. Auto-detects if not specified. """
    if data_dir is None:
        # Auto-detect: prefer Hansard if it has data, else use default
        if os.path.exists(HANSARD_DATA_DIR) and any(f.endswith('.parquet') for f in os.listdir(HANSARD_DATA_DIR) if not f.endswith('.tmp')):
            data_dir = HANSARD_DATA_DIR
        else:
            data_dir = DATA_DIR
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(index):
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


def download_hansard():
    """Download UK Hansard and save as parquets."""
    import unicodedata
    import pyarrow as pa
    from datasets import load_dataset

    def has_complete_parquet_output(data_dir):
        return os.path.isdir(data_dir) and any(
            filename.endswith(".parquet") and not filename.endswith(".tmp")
            for filename in os.listdir(data_dir)
        )

    def has_powell_sft_output(data_dir):
        if not has_complete_parquet_output(data_dir):
            return False
        parquet_files = list_parquet_files(data_dir)
        if not parquet_files:
            return False
        schema_names = set(pq.ParquetFile(parquet_files[0]).schema_arrow.names)
        return {"question", "answer"}.issubset(schema_names)

    def flush_text_shard(texts, data_dir, shard_idx):
        if not texts:
            return shard_idx
        table = pa.table({"text": texts})
        path = os.path.join(data_dir, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(table, path, row_group_size=1024)
        print(f"Wrote {path} ({len(texts):,} rows)")
        texts.clear()
        return shard_idx + 1

    def flush_powell_shard(examples, data_dir, shard_idx):
        if not examples:
            return shard_idx
        questions = [example["question"] for example in examples]
        answers = [example["answer"] for example in examples]
        messages = [json.dumps(example["messages"], ensure_ascii=True) for example in examples]
        table = pa.table({"question": questions, "answer": answers, "messages": messages})
        path = os.path.join(data_dir, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(table, path, row_group_size=1024)
        print(f"Wrote {path} ({len(examples):,} rows)")
        examples.clear()
        return shard_idx + 1

    def looks_like_question(paragraph):
        lowered = paragraph.lower()
        return (
            "?" in paragraph
            or lowered.startswith("to ask ")
            or " asked " in lowered
            or lowered.startswith("asked ")
        )

    hansard_dir = HANSARD_DATA_DIR
    powell_dir = HANSARD_POWELL_SFT_DIR
    if has_complete_parquet_output(hansard_dir) and has_powell_sft_output(powell_dir):
        print(f"Using existing Hansard shards in {hansard_dir}")
        print(f"Using existing Powell shards in {powell_dir}")
        return

    hansard_tmp_dir = hansard_dir + ".tmp"
    powell_tmp_dir = powell_dir + ".tmp"
    if os.path.exists(hansard_tmp_dir):
        shutil.rmtree(hansard_tmp_dir)
    if os.path.exists(powell_tmp_dir):
        shutil.rmtree(powell_tmp_dir)
    os.makedirs(hansard_tmp_dir, exist_ok=True)
    os.makedirs(powell_tmp_dir, exist_ok=True)

    print("Downloading UK Hansard...")
    ds = load_dataset("common-pile/uk_hansard")
    dataset = ds["train"].shuffle(seed=42)

    powell_pattern = re.compile(
        r"^(?:"
        r"(?:Mr\.?\s+)?(?:J\.?\s*)?Enoch Powell"
        r"|.*\((?:Mr\.?\s+)?(?:J\.?\s*)?Enoch Powell\)"
        r")\s*:",
        re.IGNORECASE,
    )
    powell_examples = []
    hansard_shard_idx = 0
    powell_shard_idx = 0
    hansard_texts = []
    powell_matches = 0
    powell_qa_matches = 0
    for i, ex in enumerate(dataset):
        text = unicodedata.normalize("NFKC", ex["text"])
        hansard_texts.append(text)
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
        heading = paragraphs[0].rstrip(".?!") + "?" if paragraphs else None
        for paragraph_idx, paragraph in enumerate(paragraphs):
            if powell_pattern.match(paragraph):
                powell_matches += 1
                if paragraph_idx == 0:
                    continue
                question = paragraphs[paragraph_idx - 1]
                if not looks_like_question(question):
                    if heading:
                        question = heading
                    else:
                        continue
                powell_examples.append({
                    "question": question,
                    "answer": paragraph,
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": paragraph},
                    ],
                })
                powell_qa_matches += 1

        if len(hansard_texts) >= 10000:
            hansard_shard_idx = flush_text_shard(hansard_texts, hansard_tmp_dir, hansard_shard_idx)
        if len(powell_examples) >= 10000:
            powell_shard_idx = flush_powell_shard(powell_examples, powell_tmp_dir, powell_shard_idx)
        if (i + 1) % 1000 == 0:
            print(
                f"Processed {i + 1:,} docs | Powell paragraphs: {powell_matches:,} | "
                f"Powell QA pairs: {powell_qa_matches:,}"
            )

    hansard_shard_idx = flush_text_shard(hansard_texts, hansard_tmp_dir, hansard_shard_idx)
    powell_shard_idx = flush_powell_shard(powell_examples, powell_tmp_dir, powell_shard_idx)

    if os.path.exists(hansard_dir):
        shutil.rmtree(hansard_dir)
    if os.path.exists(powell_dir):
        shutil.rmtree(powell_dir)
    os.replace(hansard_tmp_dir, hansard_dir)
    os.replace(powell_tmp_dir, powell_dir)

    print(f"Done! Saved {hansard_shard_idx:,} full-data shard(s) to {hansard_dir}")
    print(f"Done! Saved {powell_shard_idx:,} Powell shard(s) to {powell_dir}")
    print(f"Powell paragraphs matched: {powell_matches:,}")
    print(f"Powell QA pairs matched: {powell_qa_matches:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1)")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("--hansard", action="store_true", help="Download UK Hansard instead of FineWeb")
    args = parser.parse_args()

    if args.hansard:
        download_hansard()
    else:
        num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
        ids_to_download = list(range(num))
        print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
        print(f"Target directory: {DATA_DIR}")
        print()
        with Pool(processes=args.num_workers) as pool:
            results = pool.map(download_single_file, ids_to_download)
        successful = sum(1 for success in results if success)
        print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
