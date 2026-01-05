# Benchmarks

This directory contains Python benchmarks for Mahout QDP. There are two main
scripts:

- `benchmark_e2e.py`: end-to-end latency from disk to GPU VRAM (includes IO,
  normalization, encoding, transfer, and a dummy forward pass).
- `benchmark_throughput.py`: DataLoader-style throughput benchmark
  that measures vectors/sec across Mahout, PennyLane, and Qiskit.

> **📋 Benchmark Roadmap**: For planned improvements including fairness, statistical
> measurements, and visualization capabilities, see the
> [Benchmark Roadmap (RFC)](../../docs/BENCHMARK_ROADMAP.md).

> **✨ NEW - Phase 2**: Both benchmarks now support **statistical mode** with warmup,
> CUDA event timing, and comprehensive statistics! Use `--statistical` flag.

## Quick Start

From the repo root:

```bash
cd qdp
make benchmark
```

This installs the QDP Python package (if needed), installs benchmark
dependencies, and runs both benchmarks.

## Manual Setup

```bash
cd qdp/qdp-python
uv sync --group benchmark
```

Then run benchmarks with `uv run python ...` or activate the virtual
environment and use `python ...`.

## E2E Benchmark (Disk -> GPU)

### Standard Mode (Single Run)

```bash
cd qdp/qdp-python/benchmark
python benchmark_e2e.py
```

### Statistical Mode (Multiple Runs with Warmup) - Phase 2

```bash
# Run with statistical analysis
python benchmark_e2e.py --statistical

# Customize warmup and repeat iterations
python benchmark_e2e.py --statistical --warmup 5 --repeat 20

# Combine with framework selection
python benchmark_e2e.py --statistical --frameworks mahout-parquet pennylane
```

### Visualization Mode (Publication-Ready Plots) - Phase 3 NEW!

```bash
# Generate visualizations with statistical mode
python benchmark_e2e.py --statistical --visualize

# Customize output directory
python benchmark_e2e.py --statistical --visualize --output-dir ./my_results

# Full example with all options
python benchmark_e2e.py --statistical --visualize \
  --warmup 5 --repeat 20 \
  --frameworks mahout-parquet pennylane \
  --output-dir ./benchmark_results
```

**Generates:**
- Bar charts with error bars (mean ± std)
- Box plots showing quartiles and outliers
- Violin plots showing full distributions
- Markdown tables with complete statistics

Additional options:

```bash
python benchmark_e2e.py --qubits 16 --samples 200 --frameworks mahout-parquet mahout-arrow
python benchmark_e2e.py --frameworks all
```

**New Flags** (Phase 2):
- `--statistical`: Enable statistical mode with warmup and multiple runs
- `--warmup N`: Number of warmup iterations (default: 3)
- `--repeat N`: Number of measurement iterations (default: 10)

**New Flags** (Phase 3):
- `--visualize`: Generate publication-ready plots (requires --statistical)
- `--output-dir PATH`: Directory to save plots (default: ./benchmark_results)

**Statistical mode provides:**
- Warmup runs to eliminate JIT compilation overhead
- CUDA event-based precise timing
- Comprehensive statistics: mean, median, std, percentiles (P25-P99), IQR, CV
- Cache clearing between runs for fair comparison

Notes:

- `--frameworks` accepts a space-separated list or `all`.
  Options: `mahout-parquet`, `mahout-arrow`, `pennylane`, `qiskit`.
- The script writes `final_benchmark_data.parquet` and
  `final_benchmark_data.arrow` in the current working directory and overwrites
  them on each run.
- If multiple frameworks run, the script compares output states for
  correctness at the end.

## DataLoader Throughput Benchmark

Simulates a typical QML training loop by continuously loading batches of 64
vectors (default). Goal: demonstrate that QDP can saturate GPU utilization and
avoid the "starvation" often seen in hybrid training loops.

See `qdp/qdp-python/benchmark/benchmark_throughput.md` for details and example
output.

### Standard Mode (Single Run)

```bash
cd qdp/qdp-python/benchmark
python benchmark_throughput.py --qubits 16 --batches 200 --batch-size 64 --prefetch 16
python benchmark_throughput.py --frameworks mahout,pennylane
```

### Statistical Mode (Multiple Runs with Warmup) - Phase 2

```bash
# Run with statistical analysis
python benchmark_throughput.py --statistical

# Customize warmup and repeat iterations
python benchmark_throughput.py --statistical --warmup 2 --repeat 15

# Combine with framework selection
python benchmark_throughput.py --statistical --frameworks mahout,pennylane
```

### Visualization Mode (Publication-Ready Plots) - Phase 3 NEW!

```bash
# Generate visualizations with statistical mode
python benchmark_throughput.py --statistical --visualize

# Customize output directory
python benchmark_throughput.py --statistical --visualize --output-dir ./my_results

# Full example with all options
python benchmark_throughput.py --statistical --visualize \
  --warmup 2 --repeat 15 \
  --frameworks mahout,pennylane \
  --output-dir ./benchmark_results
```

**Generates TWO sets of plots:**
1. **Duration plots**: Time taken for benchmark execution
2. **Throughput plots**: Vectors processed per second

**Each set includes:**
- Bar charts with error bars
- Box plots showing quartiles
- Violin plots showing distributions
- Markdown tables with statistics

**New Flags** (Phase 2):
- `--statistical`: Enable statistical mode with warmup and multiple runs
- `--warmup N`: Number of warmup iterations (default: 2 for throughput)
- `--repeat N`: Number of measurement iterations (default: 10)

**New Flags** (Phase 3):
- `--visualize`: Generate publication-ready plots (requires --statistical)
- `--output-dir PATH`: Directory to save plots (default: ./benchmark_results)

**Statistical mode provides:**
- Warmup runs to eliminate JIT compilation overhead
- CUDA event-based precise timing
- Duration statistics: mean, median, std, percentiles
- Throughput statistics: mean, median, std, percentiles
- Cache clearing between runs for fair comparison

Notes:

- `--frameworks` is a comma-separated list or `all`.
  Options: `mahout`, `pennylane`, `qiskit`.
- Throughput is reported in vectors/sec (higher is better).

## Dependency Notes

- Qiskit and PennyLane are optional. If they are not installed, their benchmark
  legs are skipped automatically.
- For Mahout-only runs, you can uninstall the competitor frameworks:
  `uv pip uninstall qiskit pennylane`.
- **Statistical mode** requires the `benchmark_utils` package (automatically available
  when running from this directory).

### We can also run benchmarks on colab notebooks(without owning a GPU)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apache/mahout/blob/dev-qdp/qdp/qdp-python/benchmark/notebooks/mahout_benchmark.ipynb)
