import torch
import os
import triton
import triton.language as tl
import math



@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 512}, num_warps=8),
    ],
    key=["M"],
)
@triton.jit
def toeplitz_fwd_kernel(
    vals_ptr,    # [B, M]
    out_ptr,     # [B, M, M]
    stride_vals_b,
    stride_vals_m,
    stride_out_b,
    stride_out_i,
    stride_out_j,
    B: tl.constexpr,
    M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    b = tl.program_id(0)  # batch id
    row_block = tl.program_id(1)  # row block id

    # Tile rows handled by this program
    row_start = row_block * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    cols = tl.arange(0, M)

    row_mask = rows < M
    col_mask = cols < M

    # Compute Toeplitz index difference: d = i - j
    d = rows[:, None] - cols[None, :]
    lower_mask = d >= 0

    # Gather vals using computed indices
    vals_b_ptr = vals_ptr + b * stride_vals_b
    vals_ptrs = vals_b_ptr + d * stride_vals_m
    v = tl.load(vals_ptrs, mask=lower_mask & row_mask[:, None] & col_mask[None, :], other=0)

    # Store into output
    out_ptrs = out_ptr + b * stride_out_b + rows[:, None] * stride_out_i + cols[None, :] * stride_out_j
    tl.store(out_ptrs, v, mask=row_mask[:, None] & col_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128}, num_warps=8, num_stages=3),
    ],
    key=["M"],
)
@triton.jit
def toeplitz_fwd_kernel_2d(
    vals_ptr,    # [B, M]
    out_ptr,     # [B, M, M]
    stride_vals_b,
    stride_vals_m,
    stride_out_b,
    stride_out_i,
    stride_out_j,
    B: tl.constexpr,
    M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    row_start = pid_m * BLOCK_SIZE_M
    col_start = pid_n * BLOCK_SIZE_N

    # skip tiles strictly above the diagonal
    if pid_n > pid_m:
        return

    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    cols = col_start + tl.arange(0, BLOCK_SIZE_N)
    row_mask = rows < M
    col_mask = cols < M

    d = rows[:, None] - cols[None, :]
    lower_mask = (d >= 0) & row_mask[:, None] & col_mask[None, :]

    vals_b_ptr = vals_ptr + pid_b * stride_vals_b
    d_safe = tl.where(lower_mask, d, 0)
    v = tl.load(vals_b_ptr + d_safe * stride_vals_m, mask=lower_mask, other=0)

    out_ptrs = (
        out_ptr
        + pid_b * stride_out_b
        + rows[:, None] * stride_out_i
        + cols[None, :] * stride_out_j
    )
    tl.store(out_ptrs, v, mask=lower_mask, eviction_policy="evict_last")


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 512}, num_warps=8),
    ],
    key=["M"],
    reset_to_zero=["grad_vals_ptr"],
)
@triton.jit
def toeplitz_bwd_kernel(
    grad_vals_ptr,  # [B, M]
    grad_out_ptr,   # [B, M, M]
    stride_gvals_b,
    stride_gvals_m,
    stride_gout_b,
    stride_gout_i,
    stride_gout_j,
    B: tl.constexpr,
    M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    b = tl.program_id(0)  # batch id
    row_block = tl.program_id(1)  # row block id

    # Tile rows handled by this program (mirror forward kernel)
    row_start = row_block * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    cols = tl.arange(0, M)

    row_mask = rows < M
    col_mask = cols < M

    # Only lower-triangular contributes: d = i - j >= 0
    d = rows[:, None] - cols[None, :]
    lower_mask = d >= 0

    # Load upstream grads for this tile
    gout_ptrs = (
        grad_out_ptr
        + b * stride_gout_b
        + rows[:, None] * stride_gout_i
        + cols[None, :] * stride_gout_j
    )
    grads = tl.load(
        gout_ptrs,
        mask=lower_mask & row_mask[:, None] & col_mask[None, :],
        other=0,
    )

    # Atomically accumulate into grad_vals at index d = i - j
    gvals_base = grad_vals_ptr + b * stride_gvals_b
    gvals_ptrs = gvals_base + d * stride_gvals_m
    tl.atomic_add(
        gvals_ptrs,
        grads,
        mask=lower_mask & row_mask[:, None] & col_mask[None, :],
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=3),
    ],
    key=["M"],
    reset_to_zero=["grad_vals_ptr"],
)
@triton.jit
def toeplitz_bwd_kernel_kstripe(
    grad_vals_ptr,  # [B, M]
    grad_out_ptr,   # [B, M, M]
    stride_gvals_b,
    stride_gvals_m,
    stride_gout_b,
    stride_gout_i,
    stride_gout_j,
    B: tl.constexpr,
    M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    row_start = pid_m * BLOCK_SIZE_M
    k_start = pid_k * BLOCK_SIZE_K

    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    ks = k_start + tl.arange(0, BLOCK_SIZE_K)

    row_mask = rows < M
    k_mask = ks < M

    # j = i - k
    j = rows[:, None] - ks[None, :]
    lower_mask = (j >= 0) & row_mask[:, None] & k_mask[None, :]
    j_safe = tl.where(lower_mask, j, 0)

    gout_ptrs = (
        grad_out_ptr
        + pid_b * stride_gout_b
        + rows[:, None] * stride_gout_i
        + j_safe * stride_gout_j
    )
    grads = tl.load(gout_ptrs, mask=lower_mask, other=0)

    # reduce across rows for each diagonal k in this tile
    partial = tl.sum(grads, axis=0)

    gvals_base = grad_vals_ptr + pid_b * stride_gvals_b
    gvals_ptrs = gvals_base + ks * stride_gvals_m
    tl.atomic_add(gvals_ptrs, partial, mask=k_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 512}, num_warps=8, num_stages=3),
    ],
    key=["M"],
    reset_to_zero=["grad_vals_ptr"],
)
@triton.jit
def toeplitz_bwd_kernel_diag(
    grad_vals_ptr,  # [B, M]
    grad_out_ptr,   # [B, M, M]
    stride_gvals_b,
    stride_gvals_m,
    stride_gout_b,
    stride_gout_i,
    stride_gout_j,
    B: tl.constexpr,
    M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_rb = tl.program_id(2)

    # diagonal index
    k = pid_k
    if k >= M:
        return

    row_start = pid_rb * BLOCK_SIZE_M + k
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < M

    # along diagonal: j = i - k
    j = rows - k
    j_mask = j >= 0
    mask = row_mask & j_mask
    j_safe = tl.where(mask, j, 0)

    gout_ptrs = (
        grad_out_ptr
        + pid_b * stride_gout_b
        + rows * stride_gout_i
        + j_safe * stride_gout_j
    )
    grads = tl.load(gout_ptrs, mask=mask, other=0)
    partial = tl.sum(grads, axis=0)

    gvals_ptr = grad_vals_ptr + pid_b * stride_gvals_b + k * stride_gvals_m
    tl.atomic_add(gvals_ptr, partial)

@triton.autotune(
    configs=[
        # k-stripe variants
        triton.Config({"USE_DIAG": 0, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"USE_DIAG": 0, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"USE_DIAG": 0, "BLOCK_SIZE_M": 256, "BLOCK_SIZE_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"USE_DIAG": 0, "BLOCK_SIZE_M": 256, "BLOCK_SIZE_K": 128}, num_warps=8, num_stages=3),
        # diagonal variants (BLOCK_SIZE_K is dummy)
        triton.Config({"USE_DIAG": 1, "BLOCK_SIZE_M": 256, "BLOCK_SIZE_K": 1}, num_warps=4, num_stages=2),
        triton.Config({"USE_DIAG": 1, "BLOCK_SIZE_M": 512, "BLOCK_SIZE_K": 1}, num_warps=8, num_stages=3),
    ],
    key=["M"],
    reset_to_zero=["grad_vals_ptr"],
)
@triton.jit
def toeplitz_bwd_kernel_auto(
    grad_vals_ptr,  # [B, M]
    grad_out_ptr,   # [B, M, M]
    stride_gvals_b,
    stride_gvals_m,
    stride_gout_b,
    stride_gout_i,
    stride_gout_j,
    B: tl.constexpr,
    M: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_DIAG: tl.constexpr,
):
    pid_b = tl.program_id(0)
    if USE_DIAG:
        # layout: (b, k, row_block)
        pid_k = tl.program_id(1)
        pid_rb = tl.program_id(2)

        k = pid_k
        if k >= M:
            return

        row_start = pid_rb * BLOCK_SIZE_M + k
        rows = row_start + tl.arange(0, BLOCK_SIZE_M)
        row_mask = rows < M

        j = rows - k
        j_mask = j >= 0
        mask = row_mask & j_mask
        j_safe = tl.where(mask, j, 0)

        gout_ptrs = (
            grad_out_ptr
            + pid_b * stride_gout_b
            + rows * stride_gout_i
            + j_safe * stride_gout_j
        )
        grads = tl.load(gout_ptrs, mask=mask, other=0)
        partial = tl.sum(grads, axis=0)

        gvals_ptr = grad_vals_ptr + pid_b * stride_gvals_b + k * stride_gvals_m
        tl.atomic_add(gvals_ptr, partial)
    else:
        # k-stripe layout: (b, row_block, k_block)
        pid_m = tl.program_id(1)
        pid_k = tl.program_id(2)

        row_start = pid_m * BLOCK_SIZE_M
        k_start = pid_k * BLOCK_SIZE_K

        rows = row_start + tl.arange(0, BLOCK_SIZE_M)
        ks = k_start + tl.arange(0, BLOCK_SIZE_K)

        row_mask = rows < M
        k_mask = ks < M

        j = rows[:, None] - ks[None, :]
        lower_mask = (j >= 0) & row_mask[:, None] & k_mask[None, :]
        j_safe = tl.where(lower_mask, j, 0)

        gout_ptrs = (
            grad_out_ptr
            + pid_b * stride_gout_b
            + rows[:, None] * stride_gout_i
            + j_safe * stride_gout_j
        )
        grads = tl.load(gout_ptrs, mask=lower_mask, other=0)
        partial = tl.sum(grads, axis=0)

        gvals_base = grad_vals_ptr + pid_b * stride_gvals_b
        gvals_ptrs = gvals_base + ks * stride_gvals_m
        tl.atomic_add(gvals_ptrs, partial, mask=k_mask)

class Toeplitz(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vals: torch.Tensor):
        vals = vals.contiguous()
        shapes = vals.shape
        m = shapes[-1]
        b = int(torch.tensor(shapes[:-1]).prod().item()) if len(shapes) > 1 else 1
        vals_2d = vals.reshape(b, m)
        out = torch.zeros((b, m, m), dtype=vals.dtype, device=vals.device)

        stride_vals_b, stride_vals_m = vals_2d.stride()
        stride_out_b, stride_out_i, stride_out_j = out.stride()

        grid_2d = lambda META: (
            b,
            triton.cdiv(m, META["BLOCK_SIZE_M"]),
            triton.cdiv(m, META["BLOCK_SIZE_N"]),
        )
        toeplitz_fwd_kernel_2d[grid_2d](
            vals_2d,
            out,
            stride_vals_b,
            stride_vals_m,
            stride_out_b,
            stride_out_i,
            stride_out_j,
            B=b,
            M=m,
        )
        ctx.m = m
        ctx.shapes = shapes
        return out.reshape(*shapes[:-1], m, m)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        m = ctx.m
        shapes = ctx.shapes
        grad_output = grad_output.contiguous()
        b = int(torch.tensor(shapes[:-1]).prod().item()) if len(shapes) > 1 else 1
        grad_output_3d = grad_output.reshape(b, m, m)
        grad_vals = torch.zeros((b, m), dtype=grad_output.dtype, device=grad_output.device)

        stride_gvals_b, stride_gvals_m = grad_vals.stride()
        stride_gout_b, stride_gout_i, stride_gout_j = grad_output_3d.stride()

        # Always use the unified autotuned kernel; it will pick diag vs k-stripe
        grid_auto = lambda META: (
            b,
            (m if META["USE_DIAG"] else triton.cdiv(m, META["BLOCK_SIZE_M"])),
            (triton.cdiv(max(m - 0, 0), META["BLOCK_SIZE_M"]) if META["USE_DIAG"] else triton.cdiv(m, META["BLOCK_SIZE_K"]))
        )
        toeplitz_bwd_kernel_auto[grid_auto](
            grad_vals,
            grad_output_3d,
            stride_gvals_b,
            stride_gvals_m,
            stride_gout_b,
            stride_gout_i,
            stride_gout_j,
            B=b,
            M=m,
        )
        return grad_vals.reshape(*shapes),


def reference_toeplitz(vals: torch.Tensor):
    """
    Build a lower-triangular Toeplitz matrix from diagonal values.
    Input: vals [..., m]
    Output: toeplitz [..., m, m], toeplitz[..., i, j] = vals[..., i-j] if i>=j else 0
    """
    m = vals.shape[-1]

    # Indices for broadcasting
    i = torch.arange(m, device=vals.device)
    j = torch.arange(m, device=vals.device)

    offsets = i[:, None] - j[None, :]
    mask = offsets >= 0
    safe_offsets = torch.where(mask, offsets, torch.zeros(1, device=vals.device, dtype=offsets.dtype))

    batch_shape = vals.shape[:-1]
    index = safe_offsets.expand(*batch_shape, m, m)
    vals_rep = vals.unsqueeze(-2).expand(*batch_shape, m, m)
    gathered = vals_rep.gather(-1, index)
    return gathered * mask

try:
    from torch._dynamo import disable as _torch_compile_disable
except Exception:
    def _torch_compile_disable(fn):
        return fn

@_torch_compile_disable
def toeplitz_triton(vals: torch.Tensor) -> torch.Tensor:
    return Toeplitz.apply(vals)


reference_toeplitz_compiled = torch.compile(reference_toeplitz, fullgraph=False)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["m"],
        x_vals=[64, 128, 256, 512, 1024],
        line_arg="provider",
        line_vals=["eager", "compiled", "triton"],
        line_names=["PyTorch", "Torch Compile", "Triton"],
        styles=[("blue", "-"), ("orange", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="Lower-tri Toeplitz: Triton vs PyTorch",
        args={"dtype": torch.float32, "batch": 16},
    )
)
def bench(m: int, provider: str, dtype: torch.dtype, batch: int):
    device = "cuda"
    vals = torch.randn(batch, m, device=device, dtype=dtype)

    if provider == "eager":
        ms = triton.testing.do_bench(lambda: reference_toeplitz(vals))
    if provider == "compiled":
        ms = triton.testing.do_bench(lambda: reference_toeplitz_compiled(vals))
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: toeplitz_triton(vals))
    return ms


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["m"],
        x_vals=[64, 128, 256, 512, 1024, 2048],
        line_arg="provider",
        line_vals=["pytorch", "triton"],
        line_names=["PyTorch Backward", "Triton Backward"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="ms",
        plot_name="Lower-tri Toeplitz Backward: Triton vs PyTorch",
        args={"dtype": torch.float32, "batch": 16},
    )
)
def bench_backward(m: int, provider: str, dtype: torch.dtype, batch: int):
    device = "cuda"
    vals = torch.randn(batch, m, device=device, dtype=dtype, requires_grad=True)
    if provider == "pytorch":
        out = reference_toeplitz(vals)
    elif provider == "triton":
        out = toeplitz_triton(vals)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    grad_out = torch.randn_like(out)

    def run_backward():
        if vals.grad is not None:
            vals.grad = None
        out.backward(grad_out, retain_graph=True)

    # Warmup once to allow any setup/compilation
    run_backward()
    ms = triton.testing.do_bench(run_backward)
    return ms


def validate_toeplitz_correctness(m: int = 128,
                                  batch: int = 4,
                                  dtype: torch.dtype = torch.float32,
                                  device: str = "cuda",
                                  rtol: float = 1e-4,
                                  atol: float = 1e-5) -> bool:
    """
    Validate Triton Toeplitz forward and backward against the reference PyTorch implementation.

    - Forward: compares `toeplitz_triton(vals)` vs `reference_toeplitz(vals)`.
    - Backward: compares gradients w.r.t. `vals` using the same upstream gradient.

    Returns True if both comparisons pass; raises AssertionError otherwise.
    """
    # Forward check
    vals = torch.randn(batch, m, device=device, dtype=dtype)
    out_triton = toeplitz_triton(vals)
    out_ref = reference_toeplitz(vals)
    torch.testing.assert_close(out_triton, out_ref, rtol=rtol, atol=atol)
    print("Forward check passed")
    # Backward check (use same upstream gradient for both paths)
    vals_t = vals.clone().detach().requires_grad_(True)
    out_t = toeplitz_triton(vals_t)
    upstream = torch.randn_like(out_t)
    out_t.backward(upstream)
    grad_triton = vals_t.grad

    vals_r = vals.clone().detach().requires_grad_(True)
    out_r = reference_toeplitz(vals_r)
    out_r.backward(upstream)
    grad_ref = vals_r.grad

    torch.testing.assert_close(grad_triton, grad_ref, rtol=rtol, atol=atol)
    print("Backward check passed")
    return True

if __name__ == "__main__":

    validate_toeplitz_correctness(m=16)
    bench.run(print_data=True)
    bench_backward.run(print_data=True)


