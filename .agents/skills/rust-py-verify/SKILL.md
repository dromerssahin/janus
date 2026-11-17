---
name: rust-py-verify
description: Rust ve Python/PyTorch hibrit kod tabanında her değişiklik sonrası derleme, CLI/stdio köprüsü ve istatistiksel test doğrulaması yapar.
---
Janus, PyO3/Maturin yerine Rust çalışma alanı ve `uv` tabanlı Python/PyTorch CLI/stdio köprüsü (`scripts/neural_bridge.py`) kullanır. Her değişiklik sonrası şu adımlarla doğrula:

1. Rust çalışma alanında `cargo check --workspace --all-targets --offline` ve `cargo test --workspace --all-targets --offline` komutlarını çalıştır.
2. Uyarıları hata olarak ele alarak `cargo clippy --workspace --all-targets --offline -- -D warnings` kontrolünü geçir.
3. Python/PyTorch köprü doğrulaması için `uv run pytest` komutunu çalıştır; tüm adımlar geçmeden görevi tamamlanmış sayma.
