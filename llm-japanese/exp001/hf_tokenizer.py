import os
from typing import Iterable, List, Optional

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception as e:
    raise ImportError(
        "transformers が見つかりません。`pip install transformers` を実行してください"
    ) from e


class HuggingFaceTokenizer:
    def __init__(self, pretrained_name_or_path: str):
        if not pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path が空です")
        try:
            self._tok = AutoTokenizer.from_pretrained(pretrained_name_or_path, use_fast=True)
        except Exception:
            # fast 変換（SentencePiece/Tiktoken 依存）が失敗した場合は slow にフォールバック
            self._tok = AutoTokenizer.from_pretrained(pretrained_name_or_path, use_fast=False)
        self.name_or_path = pretrained_name_or_path

        # GPT 風の特殊トークンが未設定なら最低限 EOS を確保
        if self._tok.eos_token is None:
            # 既存の special_tokens_map に無ければ追加
            self._tok.add_special_tokens({"eos_token": "<|endoftext|>"})

    @property
    def vocab_size(self) -> int:
        return int(self._tok.vocab_size + len(getattr(self._tok, "added_tokens_encoder", {})))

    def encode(self, text: str, allowed_special: Optional[set] = None) -> List[int]:
        # allowed_special は互換目的で受け取るだけ（AutoTokenizer 側で管理）
        return list(self._tok.encode(text, add_special_tokens=False))

    def decode(self, ids: Iterable[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=False)
