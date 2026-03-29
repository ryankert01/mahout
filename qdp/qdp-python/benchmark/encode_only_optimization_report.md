# Encode-Only Performance Optimization Report

## Summary

Closed the 3.5x encode-only performance gap between Mahout Rust+CUDA and PyTorch GPU
by eliminating per-batch GPU synchronizations in the CUDA tensor encode path.

| Framework | Before | After | Speedup |
|---|---|---|---|
| pytorch-cpu | 3,748 vec/s | 4,252 vec/s | baseline |
| pytorch-gpu | 228,527 vec/s | 228,410 vec/s | baseline |
| **mahout (Rust+CUDA)** | **65,511 vec/s** | **259,743 vec/s** | **4.0x faster** |

Mahout is now **1.14x faster** than PyTorch GPU in encode-only mode.
End-to-end performance is unchanged (5.6x faster than PyTorch GPU).

Benchmark: `benchmark_pytorch_ref.py --qubits 16 --batches 100 --batch-size 64`
on NVIDIA RTX 3090 Ti, CUDA 12.8, amplitude encoding.

---

## Root Cause Analysis

The encode-only benchmark calls `engine.encode(cuda_tensor, 16, "amplitude")` in a
Python loop, 100 times (once per batch of 64 samples). Each call traversed:

```
Python engine.encode(cuda_tensor)
  -> _encode_from_cuda_tensor()
    -> encode_batch_from_gpu_ptr_with_stream()
      -> GpuStateVector::new_batch()             [GPU alloc]
      -> alloc_zeros(num_samples)                [norm buffer alloc]
      -> launch_l2_norm_batch()                  [async kernel]
      -> device.dtoh_sync_copy(&inv_norms_gpu)   [*** SYNC #1: blocking D2H copy ***]
      -> launch_amplitude_encode_batch()          [async kernel]
      -> sync_cuda_stream()                       [*** SYNC #2: stream sync ***]
    -> state_vector.to_precision(Float32)
      -> device.alloc() + convert kernel
      -> device.synchronize()                     [*** SYNC #3: device sync ***]
```

**3 blocking GPU synchronizations per batch x 100 batches = 300 sync points.**

PyTorch's equivalent path queues all operations asynchronously and only calls
`torch.cuda.synchronize()` once after all 100 batches. The fundamental problem was
not kernel performance -- it was synchronization overhead dominating the wall-clock time.

### Why each sync existed

1. **Norm validation D2H copy** (`dtoh_sync_copy`): The L2 norm of each sample was
   computed on GPU, then copied back to CPU to check for zero/NaN/Inf values. This
   blocked the GPU pipeline every batch.

2. **Stream sync after encode kernel** (`sync_cuda_stream`): Ensured the encode
   kernel completed before returning the DLPack pointer. Unnecessary when the consumer
   (PyTorch) handles stream synchronization via the DLPack protocol.

3. **Precision conversion sync** (`device.synchronize`): The benchmark created the
   engine with default `float32` precision but passed `float64` tensors, triggering
   a f64->f32 conversion kernel + full device sync on every batch.

---

## Changes Made

### 1. GPU-side norm validation kernel (qdp-kernels)

**Files:** `qdp-kernels/src/amplitude.cu`, `qdp-kernels/src/lib.rs`

Added `validate_inv_norms_kernel` (f64 and f32 variants) that checks inverse norms
entirely on the GPU using `atomicOr` to set an error flag. This replaces the blocking
`dtoh_sync_copy` that copied all norms back to CPU for validation.

```
Before: launch_l2_norm_batch -> cudaMemcpyDtoH(norms) -> CPU validation  [BLOCKS]
After:  launch_l2_norm_batch -> validate_inv_norms_kernel(error_flag)     [ASYNC]
```

The error flag is a single `int` on the GPU (0 = valid, 1 = invalid norm detected).
It stays on the GPU until the consumer actually needs the result.

### 2. Async encode path in AmplitudeEncoder (qdp-core)

**File:** `qdp-core/src/gpu/encodings/amplitude.rs`

Added `encode_batch_from_gpu_ptr_no_sync` (f64 and f32). This method launches three
kernels back-to-back on the same CUDA stream with **zero synchronization**:

```
norm reduction kernel -> norm validation kernel -> amplitude encode kernel -> return
```

Returns `(GpuStateVector, CudaSlice<i32>)` immediately. The caller is responsible
for syncing the stream and checking the error flag.

### 3. Async precision conversion (qdp-core)

**File:** `qdp-core/src/gpu/memory.rs`

Added `to_precision_on_stream` to `GpuStateVector`. Identical to `to_precision` but
launches the conversion kernel on a specific stream and omits `device.synchronize()`.
CUDA stream ordering guarantees correctness without explicit sync.

### 4. QdpEngine wiring (qdp-core)

**File:** `qdp-core/src/lib.rs`

- Added `encode_batch_from_gpu_ptr_no_sync` and `encode_batch_from_gpu_ptr_f32_no_sync`
  on `QdpEngine`. These call the async encoder, then `to_precision_on_stream`, then
  `to_dlpack` -- all without blocking.
- Added `GpuErrorFlag` wrapper type for deferred norm validation. Holds the GPU
  `CudaSlice<i32>` and a device reference; provides `check()` for lazy D2H copy.

### 5. Python bindings (qdp-python)

**Files:** `qdp-python/src/engine.rs`, `qdp-python/src/tensor.rs`

- `_encode_from_cuda_tensor` now uses the no-sync path for amplitude batch encoding
  from CUDA tensors (both f32 and f64). Other encodings (angle, basis, iqp) and 1D
  tensors still use the synchronous path.
- `QuantumTensor` carries an optional `GpuErrorFlag`. When present, `__dlpack__`
  checks it after stream synchronization by D2H-copying the single int. At that point
  the stream is already synced, so the 4-byte copy is essentially free.

### 6. Benchmark precision fix

**File:** `benchmark/benchmark_pytorch_ref.py`

Changed `QdpEngine(device_id)` to `QdpEngine(device_id, precision="float64")` in
`run_mahout_encode_only`. The benchmark generates float64 tensors; the old default
`float32` precision forced a f64->f32 conversion (extra alloc + kernel + sync) on
every batch. This was a benchmark configuration bug, not a code issue.

---

## Why These Changes Lead to Acceleration

### Before: 300 sync points

```
for each of 100 batches:
  alloc state vector          ~0.1 ms
  alloc norm buffer           ~0.05 ms
  launch norm kernel          ~0.01 ms (async)
  D2H copy norms (SYNC #1)   ~0.5 ms  <- blocks GPU pipeline
  launch encode kernel        ~0.05 ms (async)
  stream sync (SYNC #2)       ~0.5 ms  <- blocks GPU pipeline
  alloc conversion buffer     ~0.1 ms
  launch convert kernel       ~0.01 ms (async)
  device sync (SYNC #3)       ~0.5 ms  <- blocks GPU pipeline
                              --------
                              ~1.8 ms per batch
Total: ~180 ms for 6,400 vectors = ~35k vec/s (estimated)
Measured: 65k vec/s (GPU was partially hiding sync latency)
```

Each `synchronize` call forces the CPU to wait until the GPU finishes all queued work.
During that wait, the CPU cannot queue the next batch's kernels. The GPU sits idle
between batches.

### After: 1 sync point per batch (in `from_dlpack`)

```
for each of 100 batches:
  alloc state vector          ~0.1 ms
  alloc norm buffer           ~0.05 ms
  alloc error flag            ~0.01 ms
  launch norm kernel          ~0.01 ms (async)
  launch validation kernel    ~0.001 ms (async)
  launch encode kernel        ~0.05 ms (async)
  from_dlpack -> __dlpack__:
    D2H 4 bytes (error flag)  ~0.15 ms (syncs implicitly)
                              --------
                              ~0.37 ms per batch
Total: ~37 ms for 6,400 vectors = ~173k vec/s (estimated)
Measured: 260k vec/s (GPU pipelining provides additional overlap)
```

The key insight: `cuMemcpyDtoH` on the legacy default stream implicitly synchronizes
with all other CUDA streams. So the single D2H copy of the 4-byte error flag serves
as both the synchronization point and the validation check. No additional sync calls
are needed.

Furthermore, while `from_dlpack` blocks on batch N's error flag copy, the GPU may
already be executing batch N+1's kernels if they were queued before the sync. This
pipelining effect explains why measured throughput (260k) exceeds the per-batch estimate
(173k).

### Precision match eliminates conversion overhead

The benchmark's `QdpEngine(device_id, precision="float64")` now matches the input
tensor dtype. When precision matches, `to_precision_on_stream` returns a clone of the
existing buffer (no alloc, no kernel, no sync) -- eliminating the third sync point
and the extra 64 MB allocation per batch entirely.

---

## Correctness Guarantees

- **Norm validation is preserved**, not removed. Invalid norms are detected on-GPU by
  `validate_inv_norms_kernel` and reported as a Python `RuntimeError` when `__dlpack__`
  is called. The check is deferred but not skipped.

- **Stream ordering** guarantees that all three kernels (norm, validation, encode)
  complete in order before `from_dlpack` reads the result. CUDA stream semantics ensure
  that operations submitted to the same stream execute in FIFO order.

- **Existing sync paths are untouched**. The no-sync path is only used for amplitude
  batch encoding from CUDA tensors. All other paths (CPU tensors, NumPy, file I/O,
  angle/basis/iqp encoding, 1D tensors) continue to use the original synchronous path.

- **102 tests pass**, including cross-validation tests that compare Rust+CUDA output
  against the PyTorch reference implementation with `atol=1e-10`.

---

## Files Modified

| File | Lines | Description |
|---|---|---|
| `qdp-kernels/src/amplitude.cu` | +66 | GPU norm validation kernel (f64/f32) + launchers |
| `qdp-kernels/src/lib.rs` | +32 | FFI declarations + no-CUDA stubs |
| `qdp-core/src/gpu/encodings/amplitude.rs` | +218 | `encode_batch_from_gpu_ptr_no_sync` (f64/f32) |
| `qdp-core/src/gpu/memory.rs` | +120 | `to_precision_on_stream` |
| `qdp-core/src/lib.rs` | +130 | `GpuErrorFlag`, engine no-sync methods |
| `qdp-python/src/tensor.rs` | +30 | `error_flag` field, lazy check in `__dlpack__` |
| `qdp-python/src/engine.rs` | +30/-20 | Use no-sync path for CUDA amplitude batches |
| `benchmark/benchmark_pytorch_ref.py` | +1 | Fix precision to match input dtype |

---

## Future Work

- **GPU memory pool**: Pre-allocate and reuse state vector / norm buffers across
  encode calls to eliminate per-batch `cudaMalloc` overhead. Estimated additional
  ~20-30% improvement.
- **Extend no-sync path** to angle, basis, and IQP encoding methods.
- **Precision conversion optimization**: When f64->f32 conversion is needed, fuse it
  with the encode kernel to avoid a separate kernel launch and buffer allocation.
