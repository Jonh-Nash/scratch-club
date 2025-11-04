# 日本語データセットでの GPT の事前学習

このディレクトリ内のコードは、日本語データセットを使って小規模な GPT モデルを学習させるためのものです。

## コードの使用方法

### 1) データセットのダウンロード

このセクションでは、[`globis-university/aozorabunko-clean`](https://huggingface.co/datasets/globis-university/aozorabunko-clean) から日本語データセットをダウンロードします。

ダウンロードした `aozorabunko-dedupe-clean.jsonl` を `dataset` に配置してください。

### 2) データセットの準備

次に、`dataset/prepare_aozora_jsonl.py` スクリプトを実行して、テキストファイルを連結し、より大きなファイルにまとめます。これにより、転送やアクセスが効率化されます。

```bash
python ./dataset/prepare_aozora_jsonl.py \
  --jsonl_path ./dataset/aozorabunko-dedupe-clean.jsonl \
  --output_dir ./dataset/001 \
  --max_size_mb 25 \
  --separator "<|endoftext|>"
```

### 3) 事前学習スクリプトの実行

```bash
python pretraining.py \
  --tokenizer hf \
  --hf_name rinna/japanese-gpt2-small \
  --data_dir ../dataset/001 \
  --use_pretokenized \
  --pretokenized_dir ../dataset/001_pretok \
  --n_epochs 1 \
  --batch_size 16 \
  --eval_freq 100 \
  --print_sample_iter 100 \
  --output_dir model_checkpoints
```

## RunPod での実行

各 exp 配下で以下を実行

```
nohup ./command.sh > training.log 2>&1 &
```

## 推論

```bash
python inference.py \
  --exp_dir exp001 \
  --prompt "吾輩は猫である。名前は" \
  --max_new_tokens 200 \
  --temperature 0.8 \
  --top_k 40
```

## 出典とライセンス

このリポジトリのコードの一部は、Sebastian Raschka 氏の著書「Build a Large Language Model From Scratch」に付属するコード（Apache License 2.0）をもとにしています。

- 書籍: [Build a Large Language Model From Scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch)
- 付属コード: [github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

本リポジトリでは、Apache License 2.0 の条件に従い再配布・改変を行っています。ライセンス全文はリポジトリ直下の `LICENSE` を参照してください。

### 改変の明示

- 2025-11-03: 日本語データセット向けの前処理・事前学習設定・推論スクリプトを追加/調整

### 引用（Citation）

本リポジトリや同梱コードを研究等で引用する場合は、原著の推奨引用に従ってください：

- 書籍（推奨引用）: Raschka, Sebastian. "Build A Large Language Model (From Scratch), Manning."（出版情報等の詳細は書籍ページ参照）
- 推奨引用メタデータ: 上流の `CITATION.cff` を参照（`https://github.com/rasbt/LLMs-from-scratch/blob/main/CITATION.cff`）

### 著作権表示

Copyright © 2025 Kotaro Fukushima

## ToDo

- `pretraining.py` スクリプトに [付録 D: 学習ループへの改良] の機能（コサイン減衰、線形ウォームアップ、勾配クリッピング）を追加する。
- 事前学習スクリプトを拡張し、オプティマイザの状態を保存できるようにする（第 5 章 \_5.4 PyTorch における重みの保存と読み込みを参照）。学習が中断された場合でも再開できるようにする。
- 自作 `MultiheadAttention` クラスを、PyTorch の `nn.functional.scaled_dot_product_attention` を利用する高速な `MHAPyTorchScaledDotProduct` クラスに置き換える（効率的なマルチヘッドアテンション実装を参照）。
- [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)（`model = torch.compile`）または [thunder](https://github.com/Lightning-AI/lightning-thunder)（`model = thunder.jit(model)`）を利用してモデルを最適化し、学習を高速化する。
- GaLore（Gradient Low-Rank Projection）を導入して、事前学習をさらに高速化する。[GaLore Python ライブラリ](https://github.com/jiaweizzhao/GaLore) の `GaLoreAdamW` オプティマイザに置き換えるだけで実現できる。
