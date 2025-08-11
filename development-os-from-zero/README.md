## セットアップ

### QEMU

[https://www.qemu.org/download/#macos:title]

- `brew install qemu`
- `qemu-system-x86_64 --version`
  - QEMU emulator version 10.0.3

### Rust

[https://www.rust-lang.org/tools/install:title]

- `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- `cargo --version`
  - cargo 1.89.0 (c24e10642 2025-06-23)
- `rustc --version`
  - rustc 1.89.0 (29483883e 2025-08-04)

## 第 2 章

### UEFI アプリケーションを作ってみる

最初から入っていた。

```
$ rustup --version
rustup 1.28.2 (e4f3ad6f8 2025-04-28)
info: This is the version for the rustup toolchain manager, not the rustc compiler.
info: The currently active `rustc` version is `rustc 1.89.0 (29483883e 2025-08-04)`
$ cargo --version
cargo 1.89.0 (c24e10642 2025-06-23)
$ rustc --version
rustc 1.89.0 (29483883e 2025-08-04)
```

```
$ make --version
GNU Make 3.81
Copyright (C) 2006  Free Software Foundation, Inc.
This is free software; see the source for copying conditions.
There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.

This program built for i386-apple-darwin11.3.0

$ clang --version
Apple clang version 15.0.0 (clang-1500.3.9.4)
Target: arm64-apple-darwin23.5.0
Thread model: posix
InstalledDir: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin

$ nc
usage: nc [-46AacCDdEFhklMnOortUuvz] [-K tc] [-b boundif] [-i interval] [-p source_port]
          [--apple-recv-anyif] [--apple-awdl-unres]
          [--apple-boundif ifbound]
          [--apple-no-cellular] [--apple-no-expensive]
          [--apple-no-flowadv] [--apple-tcp-timeout conntimo]
          [--apple-tcp-keepalive keepidle] [--apple-tcp-keepintvl keepintvl]
          [--apple-tcp-keepcnt keepcnt] [--apple-tclass tclass]
          [--tcp-adp-rtimo num_probes] [--apple-intcoproc-allow]
          [--apple-tcp-adp-wtimo num_probes]
          [--setsockopt-later] [--apple-no-connectx]
          [--apple-delegate-pid pid] [--apple-delegate-uuid uuid]
          [--apple-kao] [--apple-ext-bk-idle]
          [--apple-netsvctype svc] [---apple-nowakefromsleep]
          [--apple-notify-ack] [--apple-sockev]
          [--apple-tos tos] [--apple-tos-cmsg]
          [-s source_ip_address] [-w timeout] [-X proxy_version]
          [-x proxy_address[:port]] [hostname] [port[s]]
```

### Rust ツールチェインのバージョンを固定する

```
[toolchain]
channel = "nightly-2024-01-01"
components = ["rustfmt", "rust-src"]
# https://doc.rust-lang.org/nightly/rustc/platform-support.html
targets = ["x86_64-apple-darwin"]
profile = "default"
```

### QEMU を利用して UEFI アプリケーションを実行する

`https://github.com/hikalium/wasabi/raw/main/third_party/ovmf/RELEASEX64_OVMF.fd` から取得して配置

本の通りに実行で良い

- `cargo build --target x86_64-unknown-uefi`
- `cp target/x86_64-unknown-uefi/debug/wasabi.efi mnt/EFI/BOOT/BOOTX64.EFI`
- `qemu-system-x86_64 -bios third_party/ovmf/RELEASEX64_OVMF.fd -drive format=raw,file=fat:rw:mnt`
