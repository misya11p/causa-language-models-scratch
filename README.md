# Causal Language Models from Scratch

Causal言語モデルをスクラッチ実装し、理解を深める。

## 実装モデル

### Vanilla Transformer

Vaswani, Ashish, et al. "Attention Is All You Need." *Advances in Neural Information Processing Systems*, 2017, pp. 5998–6008. arXiv:1706.03762.

Implementation: [models/_vanilla_transformer.py](models/_vanilla_transformer.py)

### GPT-2

Radford, Alec, et al. "Language Models are Unsupervised Multitask Learners." OpenAI Blog, 2019.

Implementation: [models/_gpt2.py](models/_gpt2.py)

## プログラム実行

### 準備

```
uv sync
uv run python prepare.py
```

### 学習

```
uv run torchrun --nproc_per_node=1 train.py
```
