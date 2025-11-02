import argparse
import json
import os
import re
from typing import List

from tqdm import tqdm

# 本タスクで確認済みのテキストフィールド名をリテラルで固定
TEXT_FIELD = "text"


def normalize_text(text: str) -> str:
    # 連続空行の圧縮（prepare_dataset.py に合わせた軽い整形）
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def convert_jsonl_to_txt(
    jsonl_path: str,
    output_dir: str,
    max_size_mb: int = 500,
    separator: str = "<|endoftext|>",
) -> int:
    os.makedirs(output_dir, exist_ok=True)

    current_content: List[str] = []
    current_size_bytes = 0
    file_counter = 1
    docs_processed = 0
    docs_skipped = 0

    max_bytes = max_size_mb * 1024 * 1024
    sep_bytes = separator.encode("utf-8")

    def flush():
        nonlocal current_content, current_size_bytes, file_counter
        if not current_content:
            return
        target_path = os.path.join(output_dir, f"combined_{file_counter}.txt")
        with open(target_path, "w", encoding="utf-8") as wf:
            wf.write(separator.join(current_content))
        print(f"Saved: {os.path.abspath(target_path)}")
        file_counter += 1
        current_content = []
        current_size_bytes = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Converting JSONL"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                docs_skipped += 1
                continue

            if TEXT_FIELD not in obj or not isinstance(obj[TEXT_FIELD], str):
                docs_skipped += 1
                continue

            text = obj[TEXT_FIELD].strip()
            if not text:
                docs_skipped += 1
                continue

            text = normalize_text(text)
            encoded = text.encode("utf-8")

            # 次を追加したときの概算サイズ（区切りの分も考慮）
            estimated = (
                current_size_bytes
                + (len(sep_bytes) if current_content else 0)
                + len(encoded)
            )

            if estimated > max_bytes:
                flush()
            current_content.append(text)
            current_size_bytes += (len(sep_bytes) if len(current_content) > 1 else 0) + len(encoded)
            docs_processed += 1

    # 末尾フラッシュ
    flush()

    print(f"Processed docs: {docs_processed}")
    print(f"Skipped docs:   {docs_skipped}")
    return file_counter - 1


def main():
    parser = argparse.ArgumentParser(description="Convert Aozora JSONL to pretraining-ready .txt files")
    parser.add_argument("--jsonl_path", type=str, required=True, help="入力 JSONL ファイルパス")
    parser.add_argument("--output_dir", type=str, default="aozora_preprocessed", help="出力先ディレクトリ")
    parser.add_argument("--max_size_mb", type=int, default=500, help="各出力ファイルの最大サイズ(MB)")
    parser.add_argument("--separator", type=str, default="<|endoftext|>", help="文書区切りトークン")

    args = parser.parse_args()

    saved = convert_jsonl_to_txt(
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        max_size_mb=args.max_size_mb,
        separator=args.separator,
    )
    print(f"{saved} file(s) saved in {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()


