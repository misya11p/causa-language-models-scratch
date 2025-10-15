import tomllib
from dataclasses import dataclass, asdict

from dacite import from_dict


class BaseConfig:
    def asdict(self):
        return asdict(self)


@dataclass
class ModelConfig(BaseConfig):
    arch: str
    tokenizer: str
    max_len: int
    hparams: dict

@dataclass
class TrainConfig(BaseConfig):
    n_epochs: int
    batch_size: int
    lr: float
    betas: list[float]
    grad_accum_steps: int
    max_grad_norm: float
    log_freq: int
    eval_freq: int
    wandb_name: str
    wandb_project: str

@dataclass
class Config(BaseConfig):
    model: ModelConfig
    train: TrainConfig


def load_config(fpath_toml="config.toml") -> Config:
    with open(fpath_toml, "rb") as f:
        config_dict = tomllib.load(f)
    return from_dict(Config, config_dict)
