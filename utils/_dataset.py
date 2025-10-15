from pathlib import Path

from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, DistributedSampler

from ._tokenizer import get_tokenizer


FNAME_PARQUET_TRAIN = "train.parquet"
FNAME_PARQUET_VALID = "validation.parquet"


def get_dataloader(
    batch_size,
    dpath_data="data/",
    tokenizer="tokenizer.json",
    world_size=1,
    rank=None,
):
    dpath_data = Path(dpath_data)

    ds_train = load_dataset(str(dpath_data), split="train")
    ds_valid = load_dataset(str(dpath_data), split="validation")

    if isinstance(tokenizer, str):
        tokenizer = get_tokenizer(tokenizer)

    collater = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    if world_size >= 2 and rank is not None:
        batch_size = batch_size // world_size
        train_sampler = DistributedSampler(
            ds_train,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        valid_sampler = DistributedSampler(
            ds_valid,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        train_sampler = None
        valid_sampler = None

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        pin_memory=True,
        drop_last=True,
        collate_fn=collater,
    )
    valid_loader = DataLoader(
        ds_valid,
        batch_size=batch_size,
        sampler=valid_sampler,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=collater,
    )
    return train_loader, valid_loader
