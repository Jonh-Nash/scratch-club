# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import argparse
import os
from pathlib import Path
import time
import tiktoken
import torch

from ch02 import create_dataloader_v1, create_dataloader_pretok
from ch04 import GPTModel
from ch05 import calc_loss_batch, evaluate_model, plot_losses, generate_and_print_sample


def _tokenize_and_chunk(text: str, tokenizer, context_length: int, stride: int):
    token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    inputs = []
    targets = []
    for i in range(0, len(token_ids) - context_length, stride):
        input_chunk = token_ids[i:i + context_length]
        target_chunk = token_ids[i + 1: i + context_length + 1]
        inputs.append(torch.tensor(input_chunk, dtype=torch.long))
        targets.append(torch.tensor(target_chunk, dtype=torch.long))

    if not inputs:
        return (
            torch.empty((0, context_length), dtype=torch.long),
            torch.empty((0, context_length), dtype=torch.long),
        )

    return torch.stack(inputs, dim=0), torch.stack(targets, dim=0)


def _pretokenize_corpus_if_needed(txt_root: str, out_root: str, context_length: int, stride: int, tokenizer):
    # 既存の .pt があればスキップ
    existing = [
        os.path.join(path, name)
        for path, subdirs, files in os.walk(out_root)
        for name in files if name.endswith((".pt"))
    ]
    if existing:
        print(f"Found {len(existing)} pretokenized files in {out_root}. Reusing.")
        return

    os.makedirs(out_root, exist_ok=True)

    txt_files = [
        os.path.join(path, name)
        for path, subdirs, files in os.walk(txt_root)
        for name in files if name.endswith((".txt"))
    ]
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under {txt_root}")

    total_start_time = time.time()
    print(f"Pretokenizing {len(txt_files)} file(s) from {txt_root} -> {out_root} ...")
    for idx, fp in enumerate(sorted(txt_files), start=1):
        per_file_start = time.time()
        print(f"[{idx}/{len(txt_files)}] {fp}")
        text = read_text_file(fp)
        inputs, targets = _tokenize_and_chunk(
            text=text,
            tokenizer=tokenizer,
            context_length=context_length,
            stride=stride,
        )

        base = os.path.splitext(os.path.basename(fp))[0]
        out_file = os.path.join(out_root, f"{base}.pt")
        torch.save({
            "input_ids": inputs,
            "target_ids": targets,
            "context_length": context_length,
            "stride": stride,
            "tokenizer": type(tokenizer).__name__,
        }, out_file)
        print(
            f"Saved: {out_file} | sequences={inputs.size(0) if inputs.ndim == 2 else 0} x {context_length}"
        )

        # Timing log: elapsed per file, total, ETA
        per_file_secs = time.time() - per_file_start
        total_secs = time.time() - total_start_time
        processed = idx
        remaining = len(txt_files) - processed
        avg_per_file = total_secs / processed if processed else 0.0
        eta_secs = avg_per_file * remaining

        pf_h, pf_m, pf_s = convert_time(per_file_secs)
        tot_h, tot_m, tot_s = convert_time(total_secs)
        eta_h, eta_m, eta_s = convert_time(eta_secs)
        print(
            f"Time: file {pf_h}h {pf_m}m {pf_s}s | total {tot_h}h {tot_m}m {tot_s}s | ETA {eta_h}h {eta_m}m {eta_s}s"
        )

    overall_secs = time.time() - total_start_time
    oh, om, os_ = convert_time(overall_secs)
    print(f"Pretokenization completed in {oh}h {om}m {os_}s for {len(txt_files)} file(s)")


def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    return text_data


def create_dataloaders(text_data, train_ratio, batch_size, max_length, stride, num_workers=0, tokenizer=None):
    split_idx = int(train_ratio * len(text_data))
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers,
        tokenizer=tokenizer,
    )
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,
        shuffle=False,
        num_workers=num_workers,
        tokenizer=tokenizer,
    )
    return train_loader, val_loader


def convert_time(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)


def print_eta(start_time, book_start_time, index, total_files):
    book_end_time = time.time()  # End time of processing this book
    elapsed_time = book_end_time - book_start_time
    total_elapsed_time = book_end_time - start_time
    books_remaining = total_files - index
    average_time_per_book = total_elapsed_time / index
    eta = average_time_per_book * books_remaining

    book_h, book_m, book_s = convert_time(elapsed_time)
    total_h, total_m, total_s = convert_time(total_elapsed_time)
    eta_h, eta_m, eta_s = convert_time(eta)

    print(f"Book processed {book_h}h {book_m}m {book_s}s"
          f"\nTotal time elapsed {total_h}h {total_m}m {total_s}s"
          f"\nETA for remaining books: {eta_h}h {eta_m}m {eta_s}s")


def train_model_simple(model, optimizer, device, n_epochs,
                       eval_freq, eval_iter, print_sample_iter, start_context,
                       output_dir, save_ckpt_freq, tokenizer,
                       batch_size=1024, train_ratio=0.90,
                       use_pretokenized: bool = False):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    start_time = time.time()

    try:
        for epoch in range(n_epochs):

            # Iterate over files (books or pretokenized shards)
            for index, file_path in enumerate(all_files, 1):
                book_start_time = time.time()
                if use_pretokenized:
                    print(f"Loading pretokenized {index}/{total_files}: {file_path}")
                    bundle = torch.load(file_path, map_location="cpu")
                    input_ids = bundle["input_ids"]  # [N, T]
                    target_ids = bundle["target_ids"]  # [N, T]
                    # 事前に分割済みなので train/val をテキスト比率で割る
                    split_idx = int(train_ratio * input_ids.size(0))
                    train_loader = create_dataloader_pretok(
                        input_ids=input_ids[:split_idx],
                        target_ids=target_ids[:split_idx],
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=0,
                    )
                    val_loader = create_dataloader_pretok(
                        input_ids=input_ids[split_idx:],
                        target_ids=target_ids[split_idx:],
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=0,
                    )
                else:
                    text_data = read_text_file(file_path) + " <|endoftext|> "
                    print(f"Tokenizing file {index} of {total_files}: {file_path}")

                    # Initialize new data loaders for each book
                    train_loader, val_loader = create_dataloaders(
                        text_data,
                        train_ratio=train_ratio,
                        batch_size=batch_size,
                        max_length=GPT_CONFIG_124M["context_length"],
                        stride=GPT_CONFIG_124M["context_length"],
                        num_workers=0,
                        tokenizer=tokenizer,
                    )
                print(f"Train loader: {len(train_loader)}")
                print(f"Val loader: {len(val_loader)}")
                print("Training ...")
                model.train()
                for input_batch, target_batch in train_loader:
                    optimizer.zero_grad()
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                    loss.backward()
                    optimizer.step()
                    tokens_seen += input_batch.numel()
                    global_step += 1

                    # Optional evaluation step
                    if global_step % eval_freq == 0:
                        train_loss, val_loss = evaluate_model(
                            model, train_loader, val_loader, device, eval_iter)
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        track_tokens_seen.append(tokens_seen)
                        print(f"Ep {epoch+1} (Step {global_step}): "
                              f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                    # Generate text passage
                    if global_step % print_sample_iter == 0:
                        generate_and_print_sample(
                            model, tokenizer, device, start_context
                        )

                if global_step % save_ckpt_freq:
                    file_name = output_dir / f"model_pg_{global_step}.pth"
                    torch.save(model.state_dict(), file_name)
                    print(f"Saved {file_name}")

                print_eta(start_time, book_start_time, index, total_files)

    except KeyboardInterrupt:
        file_name = output_dir / f"model_pg_{global_step}_interrupted.pth"
        torch.save(model.state_dict(), file_name)
        print(f"Saved {file_name}")

    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GPT Model Training Configuration")

    parser.add_argument("--data_dir", type=str, default="../dataset/001",
                        help="Directory containing the training data")
    parser.add_argument("--output_dir", type=str, default="model_checkpoints",
                        help="Directory where the model checkpoints will be saved")
    parser.add_argument("--n_epochs", type=int, default=1,
                        help="Number of epochs to train the model")
    parser.add_argument("--print_sample_iter", type=int, default=1000,
                        help="Iterations between printing sample outputs")
    parser.add_argument("--eval_freq", type=int, default=100,
                        help="Frequency of evaluations during training")
    parser.add_argument("--save_ckpt_freq", type=int, default=100_000,
                        help="Frequency of saving model checkpoints during training")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--debug", type=bool, default=False,
                        help="Uses a very small model for debugging purposes")
    parser.add_argument("--use_pretokenized", action="store_true",
                        help="事前トークナイズ(.pt)を使用して学習する")
    parser.add_argument("--pretokenized_dir", type=str, default="",
                        help=".pt を格納したディレクトリ（--use_pretokenized と併用）")

    # Tokenizer options
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                        choices=["gpt2", "spm", "hf"],
                        help="使用するトークナイザの種類: gpt2 | spm | hf")
    parser.add_argument("--spm_model", type=str, default="",
                        help="SentencePiece の .model パス（--tokenizer spm のとき必須）")
    parser.add_argument("--hf_name", type=str, default="rinna/japanese-gpt2-medium",
                        help="HuggingFace のトークナイザ名（--tokenizer hf のとき使用）")

    args = parser.parse_args()

    if args.debug:
        GPT_CONFIG_124M = {
            "vocab_size": 50257,     # Vocabulary size (後で上書き)
            "context_length": 256,    # Context length
            "emb_dim": 12,           # Embedding dimension
            "n_heads": 2,            # Number of attention heads
            "n_layers": 2,           # Number of layers
            "drop_rate": 0.0,        # Dropout rate, deactivated via 0.0 as dropout in LLMs is not recommended anymore
            "qkv_bias": False        # Query-key-value bias
        }

    else:
        GPT_CONFIG_124M = {
            "vocab_size": 50257,     # Vocabulary size (後で上書き)
            "context_length": 256,  # Context length
            "emb_dim": 768,          # Embedding dimension
            "n_heads": 12,           # Number of attention heads
            "n_layers": 12,          # Number of layers
            "drop_rate": 0.1,        # Dropout rate
            "qkv_bias": False        # Query-key-value bias
        }

    # Load tokenizer
    if args.tokenizer == "gpt2":
        tokenizer = tiktoken.get_encoding("gpt2")
        vocab_size = 50257
    elif args.tokenizer == "spm":
        from japanese_tokenizer import SentencePieceTokenizer
        if not args.spm_model:
            raise ValueError("--spm_model に SentencePiece の .model を指定してください")
        tokenizer = SentencePieceTokenizer(args.spm_model)
        vocab_size = tokenizer.vocab_size
    else:  # hf
        from hf_tokenizer import HuggingFaceTokenizer
        tokenizer = HuggingFaceTokenizer(args.hf_name)
        vocab_size = tokenizer.vocab_size

    # Reflect vocab size
    GPT_CONFIG_124M["vocab_size"] = vocab_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    if args.use_pretokenized:
        # 事前トークナイズディレクトリを決定（未指定なら data_dir + "_pretok"）
        src_dir = args.pretokenized_dir or (args.data_dir.rstrip("/") + "_pretok")
        # 必要なら自動生成（選択トークナイザで）
        _pretokenize_corpus_if_needed(
            txt_root=args.data_dir,
            out_root=src_dir,
            context_length=GPT_CONFIG_124M["context_length"],
            stride=GPT_CONFIG_124M["context_length"],
            tokenizer=tokenizer,
        )
        all_files = [
            os.path.join(path, name)
            for path, subdirs, files in os.walk(src_dir)
            for name in files if name.endswith((".pt"))
        ]
    else:
        data_dir = args.data_dir
        all_files = [os.path.join(path, name) for path, subdirs, files
                     in os.walk(data_dir) for name in files if name.endswith((".txt"))]
    total_files = len(all_files)

    if total_files == 0:
        print("No training files found. Check input directory and flags.")
        quit()
    print("Total files:", total_files)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, optimizer, device,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        eval_iter=1,
        print_sample_iter=args.print_sample_iter,
        output_dir=output_dir,
        save_ckpt_freq=args.save_ckpt_freq,
        start_context="吾輩は猫である。名前は",
        tokenizer=tokenizer,
        use_pretokenized=args.use_pretokenized
    )

    epochs_tensor = torch.linspace(0, args.n_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    torch.save(model.state_dict(), output_dir / "model_pg_final.pth")
    print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
