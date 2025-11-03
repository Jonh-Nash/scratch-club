import argparse
import os
import sys
from pathlib import Path
import torch
from typing import Optional, List


def resolve_device(device_arg: str) -> torch.device:
    if device_arg and device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # macOS Metal
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_config(vocab_size: int, debug: bool, context_length: int):
    if debug:
        return {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "emb_dim": 12,
            "n_heads": 2,
            "n_layers": 2,
            "drop_rate": 0.0,
            "qkv_bias": False,
        }
    return {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }


def load_tokenizer(kind: str, hf_name: str, spm_model: str):
    if kind == "gpt2":
        import tiktoken

        class _TkTok:
            def __init__(self):
                self._tok = tiktoken.get_encoding("gpt2")
                self.eos_token_id = None

            @property
            def vocab_size(self):
                return 50257

            def encode(self, text, allowed_special=None):
                return self._tok.encode(text, allowed_special=allowed_special or set())

            def decode(self, ids):
                return self._tok.decode(ids)

        return _TkTok()

    if kind == "spm":
        try:
            from japanese_tokenizer import SentencePieceTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError("--tokenizer spm を使うには japanese_tokenizer が必要です") from e
        tok = SentencePieceTokenizer(spm_model)
        # ラッパのインタフェースに合わせる
        tok.eos_token_id = None
        return tok

    # hf
    # exp ディレクトリ側のラッパを使う
    from hf_tokenizer import HuggingFaceTokenizer
    return HuggingFaceTokenizer(hf_name)


def main():
    parser = argparse.ArgumentParser(description="Inference for exp models")
    parser.add_argument("--exp_dir", type=str, default="exp001", help="実験ディレクトリ（ch04.py/ch05.py の在処）")
    parser.add_argument("--ckpt", type=str, default="", help="チェックポイント .pth（未指定なら <exp_dir>/model_checkpoints/model_pg_final.pth）")
    parser.add_argument("--tokenizer", type=str, default="hf", choices=["hf", "gpt2", "spm"], help="使用するトークナイザ")
    parser.add_argument("--hf_name", type=str, default="rinna/japanese-gpt2-medium", help="HF トークナイザ名（--tokenizer hf）")
    parser.add_argument("--spm_model", type=str, default="", help="SentencePiece .model（--tokenizer spm）")
    parser.add_argument("--prompt", type=str, default="吾輩は猫である。名前は", help="プロンプト")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="生成トークン数")
    parser.add_argument("--temperature", type=float, default=0.8, help="サンプリング温度 (0は貪欲)")
    parser.add_argument("--top_k", type=int, default=40, help="top-k サンプリングの k（0 で無効）")
    parser.add_argument("--context_length", type=int, default=256, help="モデルのコンテキスト長（学習時と揃える）")
    parser.add_argument("--debug", action="store_true", help="学習時 --debug 相当の極小モデル設定で復元")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="推論デバイス")

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir が見つかりません: {exp_dir}")

    # exp 内のモジュールを import できるようにパス追加
    sys.path.insert(0, str(exp_dir))

    # 依存モジュールを exp ディレクトリから import
    from ch04 import GPTModel  # type: ignore
    from ch05 import generate, text_to_token_ids, token_ids_to_text  # type: ignore

    # トークナイザ
    tokenizer = load_tokenizer(args.tokenizer, args.hf_name, args.spm_model)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if callable(vocab_size):
        vocab_size = tokenizer.vocab_size
    if not isinstance(vocab_size, int):
        raise RuntimeError("トークナイザの語彙サイズを取得できませんでした")

    # モデル構成
    cfg = build_config(vocab_size=vocab_size, debug=bool(args.debug), context_length=int(args.context_length))

    # モデルと重み
    device = resolve_device(args.device)
    model = GPTModel(cfg)
    model.to(device)

    def _try_load_state(p: Path) -> Optional[dict]:
        try:
            if not p.exists() or p.stat().st_size == 0:
                return None
            state_obj = torch.load(str(p), map_location="cpu")
            if isinstance(state_obj, dict):
                return state_obj
            return None
        except Exception:
            return None

    # 優先: 指定 ckpt -> final -> ほかの pth を新しい順
    candidates: List[Path] = []
    if args.ckpt:
        candidates.append(Path(args.ckpt))
    final_path = exp_dir / "model_checkpoints" / "model_pg_final.pth"
    if final_path not in candidates:
        candidates.append(final_path)
    # ディレクトリ内の他の .pth
    ckpt_dir = exp_dir / "model_checkpoints"
    if ckpt_dir.exists():
        others = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in others:
            if p not in candidates:
                candidates.append(p)

    loaded_state: Optional[dict] = None
    chosen_path: Optional[Path] = None
    for p in candidates:
        loaded_state = _try_load_state(p)
        if loaded_state is not None:
            chosen_path = p
            break

    if loaded_state is None or chosen_path is None:
        raise FileNotFoundError(
            f"有効なチェックポイントが見つかりませんでした。探索した候補: "
            f"{[str(p) for p in candidates]}"
        )

    print(f"[info] Loading checkpoint: {chosen_path}")
    model.load_state_dict(loaded_state)
    model.eval()

    # 生成
    encoded = text_to_token_ids(args.prompt, tokenizer).to(device)
    eos_id = None
    # HF ラッパの場合は _tok.eos_token_id がある
    if hasattr(tokenizer, "_tok") and getattr(tokenizer._tok, "eos_token_id", None) is not None:
        eos_id = tokenizer._tok.eos_token_id
    elif getattr(tokenizer, "eos_token_id", None) is not None:
        eos_id = tokenizer.eos_token_id

    top_k = None if args.top_k <= 0 else int(args.top_k)
    out_ids = generate(
        model=model,
        idx=encoded,
        max_new_tokens=int(args.max_new_tokens),
        context_size=cfg["context_length"],
        temperature=float(args.temperature),
        top_k=top_k,
        eos_id=eos_id,
    )
    text = token_ids_to_text(out_ids, tokenizer)
    print(text)


if __name__ == "__main__":
    main()


