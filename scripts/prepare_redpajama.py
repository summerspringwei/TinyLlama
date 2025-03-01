import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Config, Tokenizer

filenames_sample = [
    # "arxiv_sample.jsonl",
    # "book_sample.jsonl",
    # "c4_sample.jsonl",
    # "cc_2019-30_sample.jsonl",
    # "cc_2020-05_sample.jsonl",
    # "cc_2021-04_sample.jsonl",
    # "cc_2022-05_sample.jsonl",
    # "cc_2023-06_sample.jsonl",
    # "github_sample.jsonl",
    # "stackexchange_sample.jsonl",
    # "wikipedia_sample.jsonl",
    # "AnghaBench-assembly-g-O2.json",
    # "AnghaBench-assembly-g-O0.json"
    # "AnghaBench.json"
    "AnghaBench_text_train_paired_assembly-g-O2_C_2K.json"
]

filename_sets = {
    "arxiv": "arxiv/arxiv*",
    "book": "book/book*",
    "c4": "c4/c4-train*",
    "common_crawl": "common_crawl/*",
    "github": "github/filtered*",
    "stackexchange": "stackexchange/stackexchange*",
    "wikipedia": "wikipedia/wiki*",
}


def prepare_sample(
    source_path: Path, checkpoint_dir: Path, destination_path: Path, chunk_size: int, match: str = ""
) -> None:
    """Prepare the "Red Pajama" dataset using the original tokenizer."""
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(checkpoint_dir)

    for name in filenames_sample:
        if match and match not in name:
            continue

        filepath = source_path / name

        if not filepath.is_file():
            raise RuntimeError(
                f"Input file not found at {filepath}. \nMake sure you download the data, e.g. wget -i"
                " https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt or through"
                " \nhttps://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T"
                " \nhttps://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample \n"
            )

        prefix, _ = os.path.splitext(name)

        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=prefix,
            chunk_size=chunk_size,
            sep_token=tokenizer.eos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )
        seq_len_list = []
        print(f"Processing {name}")
        max_seq_len = 0
        with open(filepath, encoding="utf-8") as f:
            for row in tqdm(f):
                tmp = json.loads(row)
                if type(tmp) == str:
                    tmp = json.loads(tmp)
                text = tmp["text"]
                text_ids = tokenizer.encode(text)
                seq_len_list.append(len(text_ids))
                if len(text_ids) > max_seq_len:
                    max_seq_len = len(text_ids)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))
        print("Max seq len: ", max_seq_len)
        builder.write_reminder()
        # Save the list of seq_len
        seq_len_list = np.array(seq_len_list)
        np.save(destination_path / f"{prefix}_seq_len.npy", seq_len_list)
        # Draw the histogram of seq_len
        import matplotlib.pyplot as plt
        plt.hist(seq_len_list, bins=100)
        plt.title('Distribution of Sentence Lengths')
        plt.xlabel('Sentence Length')
        plt.ylabel('Frequency')
        plt.savefig(f'histogram_{prefix}.png')


def prepare_full(
    source_path: Path, checkpoint_dir: Path, destination_path: Path, chunk_size: int, match: str = ""
) -> None:
    """Prepare the "Red Pajama" dataset using the original tokenizer."""
    import zstandard as zstd

    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(checkpoint_dir)

    for set_name, pattern in filename_sets.items():
        if match and match not in set_name:
            continue

        is_cc = set_name == "common_crawl"

        filenames = glob.glob(os.path.join(source_path, pattern), recursive=True)

        if not filenames:
            raise RuntimeError(
                f"No files matching {pattern} found at {source_path}. \nMake sure you download the data, e.g. wget -i"
                " https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt or through"
                " \nhttps://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T"
                " \nhttps://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample \n"
            )

        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=set_name,
            chunk_size=chunk_size,
            sep_token=tokenizer.eos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )

        for name in filenames:
            filepath = source_path / name

            print(f"Processing {name}")

            if is_cc:
                with zstd.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
                    for row in tqdm(f):
                        text = json.loads(row)["text"]
                        text_ids = tokenizer.encode(text)
                        builder.add_array(np.array(text_ids, dtype=builder.dtype))
            else:
                with open(filepath, encoding="utf-8") as f:
                    for row in tqdm(f):
                        text = json.loads(row)["text"]
                        text_ids = tokenizer.encode(text)
                        builder.add_array(np.array(text_ids, dtype=builder.dtype))

        builder.write_reminder()


def prepare(
    source_path: Path = Path("data/RedPajama-Data-1T-Sample"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    destination_path: Path = Path("data/redpajama_sample"),
    sample: bool = True,
    match: str = "",
) -> None:
    """Prepare the "Red Pajama" dataset. We assume tokenizer has been trained."""
    with open(checkpoint_dir / "lit_config.json") as fp:
        config = Config(**json.load(fp))

    prepare_fn = prepare_sample if sample else prepare_full
    prepare_fn(
        source_path=source_path,
        checkpoint_dir=checkpoint_dir,
        destination_path=destination_path,
        chunk_size=(config.block_size + 1) * 1024,  # block size + 1 for causal, 1024 blocks
        match=match,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
    