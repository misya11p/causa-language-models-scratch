from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import OrderedDict

import torch
import torch.nn as nn
from schedulefree import RAdamScheduleFree
from tqdm import tqdm
import typer

from utils import load_config, get_tokenizer, get_dataloader


JST = timezone(timedelta(hours=9))
DPATH_CHECKPOINTS = "checkpoints"
FNAME_LATEST = "latest.pth"

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(add_completion=False, context_settings=CONTEXT_SETTINGS)


@app.command()
def main(
    fpath_config: str = typer.Option(
        "config.toml",
        "-c", "--config",
        help="File path to the config file (.toml)",
    ),
    dpath_data: str = typer.Option(
        "data/",
        "-d", "--data",
        help="Directory path to the dataset",
    ),
):
    """Train a language model."""

    config = load_config(fpath_config)
    dpath_data = Path(dpath_data)
    trainer = Trainer(config, dpath_data)
    trainer.train()


class Trainer:
    def __init__(self, config, dpath_data):
        self.tokenizer = get_tokenizer(config.model.tokenizer)

        config.model.hparams["max_len"] = config.model.max_len
        config.model.hparams["vocab_size"] = self.tokenizer.vocab_size
        self.config = config.asdict()

        match config.model.arch:
            case "vanilla":
                from models import VanillaTransformer
                self.model = VanillaTransformer(**config.model.hparams)
                print("Model: Vanilla Transformer", flush=True)
            case _:
                raise ValueError(f"Model {config.model.arch} not recognized")

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of parameters: {n_params:,}", flush=True)

        self.dataloader = get_dataloader(
            batch_size=config.train.batch_size,
            dpath_data=dpath_data,
            tokenizer=self.tokenizer,
        )

        dpath_ckpt_root = Path(DPATH_CHECKPOINTS)
        dpath_ckpt_root.mkdir(parents=True, exist_ok=True)
        now = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        dpath_ckpt = dpath_ckpt_root / now
        self.dpath_ckpt = Path(dpath_ckpt)
        self.dpath_ckpt.mkdir(parents=True, exist_ok=True)
        self.fpath_latest = dpath_ckpt_root / FNAME_LATEST

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = RAdamScheduleFree(
            self.model.parameters(), lr=config.train.lr
        )
        print(f"Checkpoints will be saved to {self.dpath_ckpt}", flush=True)

        self.max_len = config.model.max_len
        self.n_epochs = config.train.n_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch = 0

    def train(self):
        print("Training started.", flush=True)
        self.model.to(self.device)
        self.model = torch.compile(self.model)
        torch.set_float32_matmul_precision("high")
        self.model.train()

        for epoch in range(self.n_epochs):
            self.epoch = epoch + 1
            pbar = tqdm(
                self.dataloader,
                desc=f"Epoch {self.epoch}/{self.n_epochs}",
                leave=False
            )
            for batch in pbar:
                input_ids, labels = self._unpack_batch(batch)
                pred = self.model(input_ids)
                loss = self._loss_fn(pred, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self._save_checkpoint()

    def _unpack_batch(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        input_ids = input_ids[:, :self.max_len - 1].contiguous()
        labels = labels[:, 1:self.max_len].contiguous()
        return input_ids, labels

    def _loss_fn(self, pred, labels):
        pred = pred.view(-1, pred.size(-1))
        labels = labels.view(-1)
        loss = self.criterion(pred, labels)
        return loss

    def _save_checkpoint(self):
        state_dict = self.model.state_dict()
        correct_state_dict = OrderedDict()
        for key, value in state_dict.items():
            key = key.replace("_orig_mod.", "")
            correct_state_dict[key] = value
        state_dict = {
            "model": correct_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
        }
        torch.save(state_dict, self.dpath_ckpt / f"epoch{self.epoch}.pth")
        torch.save(state_dict, self.fpath_latest)

if __name__ == "__main__":
    main()