"""Valida se o PyTorch realmente consegue executar cálculos na GPU."""

from __future__ import annotations

import platform
import time

import torch


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA compilado no PyTorch: {torch.version.cuda}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA não está disponível. Confira o driver NVIDIA e instale o wheel "
            "CUDA do PyTorch indicado em https://pytorch.org/get-started/locally/."
        )

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    print(f"GPU: {properties.name}")
    print(f"Compute capability: {properties.major}.{properties.minor}")
    print(f"VRAM: {properties.total_memory / 1024**3:.2f} GiB")

    # Multiplicação suficientemente longa para aparecer no monitor da GPU.
    size = 4096
    first = torch.randn(size, size, device=device, dtype=torch.float16)
    second = torch.randn(size, size, device=device, dtype=torch.float16)
    torch.cuda.synchronize()
    started_at = time.perf_counter()
    result = first @ second
    torch.cuda.synchronize()

    print(f"Cálculo executado em: {time.perf_counter() - started_at:.3f} s")
    print(f"Resultado na GPU: {result.device}")
    print(f"Pico de VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
