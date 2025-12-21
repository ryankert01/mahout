# CPU Warning Tests

This directory contains tests to verify that CPU-based code paths emit warnings when GPU alternatives exist.

## Running Tests

These tests require a CUDA-enabled environment. To run them:

```bash
# Run with warning output visible
RUST_LOG=warn cargo test test_cpu_l2_norm_emits_warning -- --nocapture

# Run batch warning test
RUST_LOG=warn cargo test test_cpu_batch_l2_norms_emits_warning -- --nocapture
```

## Expected Output

When running these tests with logging enabled, you should see warnings like:

```
WARN qdp_core::preprocessing: Using CPU-based L2 norm calculation instead of GPU kernel. Consider increasing data size or using GPU-accelerated methods for better performance.
```

## What's Being Tested

1. `test_cpu_l2_norm_emits_warning` - Verifies that single-sample L2 norm calculation on CPU emits a warning
2. `test_cpu_batch_l2_norms_emits_warning` - Verifies that batch L2 norm calculation on CPU emits a warning

These warnings are emitted whenever CPU-based code is used instead of available GPU kernels, helping users identify performance opportunities.
