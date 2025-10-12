import re
import ast

from datasets import Dataset, load_dataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_dataloader(batch_size, tokenizer):
    ds = load_dataset("data/wiki40b_processed/", split="train")
    collater = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    train_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collater,
    )
    return train_loader


def format_text(ds):
    sentences = []
    texts = ds["text"]
    for text in tqdm(texts, desc="Formatting text"):
        decoded_string = ast.literal_eval(text).decode("utf-8")
        paragraphs = re.findall(
            r"_START_PARAGRAPH_\n(.*?)(?=\n_START_PARAGRAPH_|\Z)",
            decoded_string,
            re.DOTALL
        )
        for paragraph in paragraphs:
            paragraph = paragraph.replace("_NEWLINE_", "")
            paragraph = paragraph.replace("\n", "")
            if paragraph:
                sentences.append(paragraph.strip())
    ds_new = Dataset.from_dict({"text": sentences})
    return ds_new
