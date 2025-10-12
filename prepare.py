from pathlib import Path

from datasets import load_dataset

from utils import format_text, get_tokenizer, train_tokenizer


FPATH_TOKENIZER = Path("tokenizer.json")
Path("data/wiki40b").mkdir(parents=True, exist_ok=True)
Path("data/wiki40b_processed").mkdir(parents=True, exist_ok=True)


def main():
    ds = load_dataset("wiki40b", "ja", split="train")
    print("Loaded original dataset.", flush=True)
    ds = format_text(ds)
    ds.to_parquet(f"data/wiki40b/train.parquet")

    if not FPATH_TOKENIZER.exists():
        train_tokenizer(
            text=ds["text"],
            fpath_tokenizer=FPATH_TOKENIZER,
            vocab_size=16000,
        )
    tokenizer = get_tokenizer()
    tokenizer.add_special_tokens({
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>",
    })
    ds = ds.map(
        lambda x: tokenizer(x["text"]),
        batched=True,
        remove_columns=["text"]
    )

    ds.to_parquet(f"data/wiki40b_processed/train.parquet")


if __name__ == "__main__":
    main()
