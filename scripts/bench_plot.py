import os
import sys
import triton

# Allow running this script from the repository's scripts/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import matplotlib.pyplot as plt
from src.efficient_toeplitz_kernel import (
    reference_toeplitz,
    reference_toeplitz_compiled,
    toeplitz_triton,
)


def time_ms(fn, *args, warmup=5, iters=50):
    # Simple wall-time benchmarking in ms
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    import time

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def main():
    device = "cuda"
    dtype = torch.float32
    batch = 16
    sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    results = {"eager": [], "compiled": [], "triton": []}

    for m in sizes:
        vals = torch.randn(batch, m, device=device, dtype=dtype)

        ms_eager = triton.testing.do_bench(lambda : reference_toeplitz(vals))
        ms_comp = triton.testing.do_bench(lambda : reference_toeplitz_compiled(vals))
        ms_triton = triton.testing.do_bench(lambda : toeplitz_triton(vals))

        results["eager"].append(ms_eager)
        results["compiled"].append(ms_comp)
        results["triton"].append(ms_triton)
        print(f"m={m:4d}  eager={ms_eager:7.3f}  compiled={ms_comp:7.3f}  triton={ms_triton:7.3f}")

    os.makedirs("../assets", exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(sizes, results["eager"], label="PyTorch", color="blue")
    plt.scatter(sizes, results["eager"], color="blue", s=24, zorder=3)
    plt.plot(sizes, results["compiled"], label="Torch Compile", color="orange")
    plt.scatter(sizes, results["compiled"], color="orange", s=24, zorder=3)
    plt.plot(sizes, results["triton"], label="Triton", color="green")
    plt.scatter(sizes, results["triton"], color="green", s=24, zorder=3)
    plt.xlabel("Size")
    plt.ylabel("Time [ms]")
    plt.title("Lower triangular Toeplitz: Pytorch vs Triton")
    plt.legend()

    
    out_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets"), "bench_forward.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()


