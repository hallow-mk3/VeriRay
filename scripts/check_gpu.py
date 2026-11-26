#!/usr/bin/env python
"""GPU sanity check (local only; makes NO network calls).

Reports torch/CUDA build, device name, total/free VRAM, and runs a small kernel to catch
the Blackwell "sm_120 is not compatible" failure of older CUDA builds.
"""

from __future__ import annotations

import sys

import torch


def main() -> int:
    print(f"torch {torch.__version__}  cuda-build={torch.version.cuda}  available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("NO CUDA: install the CUDA build (see requirements-gpu.txt / README.md).")
        return 1
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    print(f"device: {props.name}  capability={props.major}.{props.minor}  total={total/2**30:.2f} GiB  free={free/2**30:.2f} GiB")
    try:
        a = torch.randn(1024, 1024, device="cuda")
        torch.cuda.synchronize()
        print(f"kernel ok: {float((a @ a).sum().item()) == float((a @ a).sum().item())}")
    except Exception as exc:  # noqa: BLE001
        print(f"KERNEL FAILED: {type(exc).__name__}: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
