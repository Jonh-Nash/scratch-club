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

## ToDo

- `pretraining.py` スクリプトに [付録 D: 学習ループへの改良] の機能（コサイン減衰、線形ウォームアップ、勾配クリッピング）を追加する。
- 事前学習スクリプトを拡張し、オプティマイザの状態を保存できるようにする（第 5 章 \_5.4 PyTorch における重みの保存と読み込みを参照）。学習が中断された場合でも再開できるようにする。
- 自作 `MultiheadAttention` クラスを、PyTorch の `nn.functional.scaled_dot_product_attention` を利用する高速な `MHAPyTorchScaledDotProduct` クラスに置き換える（効率的なマルチヘッドアテンション実装を参照）。
- [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)（`model = torch.compile`）または [thunder](https://github.com/Lightning-AI/lightning-thunder)（`model = thunder.jit(model)`）を利用してモデルを最適化し、学習を高速化する。
- GaLore（Gradient Low-Rank Projection）を導入して、事前学習をさらに高速化する。[GaLore Python ライブラリ](https://github.com/jiaweizzhao/GaLore) の `GaLoreAdamW` オプティマイザに置き換えるだけで実現できる。
