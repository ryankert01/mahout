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

## NumPy I/O Benchmark

Benchmarks loading quantum state data from NumPy `.npy` files and encoding on GPU.
Compares Mahout's NumPy reader against PennyLane's file loading + encoding.

### Standard Mode (Single Run)

```bash
cd qdp/qdp-python/benchmark
python benchmark_numpy_io.py --qubits 10 --samples 1000
python benchmark_numpy_io.py --frameworks mahout,pennylane
```

### Statistical Mode (Multiple Runs with Warmup) - NEW!

```bash
# Run with statistical analysis
python benchmark_numpy_io.py --statistical

# Customize warmup and repeat iterations
python benchmark_numpy_io.py --statistical --warmup 3 --repeat 15

# Combine with framework selection
python benchmark_numpy_io.py --statistical --frameworks mahout,pennylane
```

### Visualization Mode (Publication-Ready Plots) - NEW!

```bash
# Generate visualizations with statistical mode
python benchmark_numpy_io.py --statistical --visualize

# Customize output directory
python benchmark_numpy_io.py --statistical --visualize --output-dir ./my_results

# Full example with all options
python benchmark_numpy_io.py --statistical --visualize \
  --warmup 3 --repeat 15 \
  --qubits 10 --samples 1000 \
  --frameworks mahout,pennylane \
  --output-dir ./numpy_results
```

**Generates TWO sets of plots:**
1. **Duration plots**: Time taken for I/O + encoding
2. **Throughput plots**: Samples processed per second

**Each set includes:**
- Bar charts with error bars
- Box plots showing quartiles
- Violin plots showing distributions
- Markdown tables with statistics

**New Flags**:
- `--statistical`: Enable statistical mode with warmup and multiple runs
- `--warmup N`: Number of warmup iterations (default: 3)
- `--repeat N`: Number of measurement iterations (default: 10)
- `--visualize`: Generate publication-ready plots (requires --statistical)
- `--output-dir PATH`: Directory to save plots (default: ./benchmark_results)

**Statistical mode provides:**
- Warmup runs to eliminate JIT compilation overhead
- CUDA event-based precise timing
- Duration statistics: mean, median, std, percentiles
- Throughput statistics: mean, median, std, percentiles
- Cache clearing between runs for fair comparison

Notes:

- Compares NumPy file I/O performance: Mahout's Rust-based reader vs PennyLane's Python loading
- Measures end-to-end: file read + GPU encoding + transfer
- `--output` flag allows saving the `.npy` file for reuse
- `--frameworks` options: `mahout`, `pennylane`, or `all`

## Documentation and Tutorials

### 📚 Best Practices Guide

See [BEST_PRACTICES.md](./BEST_PRACTICES.md) for comprehensive guidance on:
- Fairness principles (warmup, cache clearing, CUDA events)
- Statistical rigor (distributions, repetitions, confidence intervals)
- Reproducibility (configuration, versioning, documentation)
- Publication guidelines (plots, tables, reporting)
- Common pitfalls and how to avoid them

### 📓 Tutorial Notebook

Interactive tutorial covering all features: [statistical_benchmark_tutorial.ipynb](./notebooks/statistical_benchmark_tutorial.ipynb)

Topics covered:
- Using benchmark_utils API directly
- Running statistical benchmarks
- Creating publication-ready visualizations
- Best practices with code examples

### 📊 Visualization Examples

The `--visualize` flag generates 4 types of outputs:

1. **Bar charts** (`*_bars.png`): Quick comparison with error bars (mean ± std)
2. **Box plots** (`*_box.png`): Show quartiles, median, and outliers
3. **Violin plots** (`*_violin.png`): Show full distribution shapes
4. **Markdown tables** (`*_table.md`): Complete statistics in table format

**Example output structure**:
```
benchmark_results/
├── e2e_q16_s200_bars.png        # Bar chart comparison
├── e2e_q16_s200_box.png         # Box plot
├── e2e_q16_s200_violin.png      # Violin plot
└── e2e_q16_s200_table.md        # Statistics table
```

**Throughput benchmark** generates 2 sets (8 files total):
- Duration metrics: `throughput_duration_q16_b200_*.{png|md}`
- Throughput metrics: `throughput_vecpersec_q16_b200_*.{png|md}`

All plots are 300 DPI, publication-ready PNG images suitable for papers and presentations.

### 🔗 Additional Resources

- [Benchmark Roadmap RFC](../../docs/BENCHMARK_ROADMAP.md) - Complete design and motivation
- [Benchmark Utils API](./benchmark_utils/README.md) - Detailed API documentation
- [Example Configuration](./benchmark_config.yaml) - YAML configuration template

## Dependency Notes

- Qiskit and PennyLane are optional. If they are not installed, their benchmark
  legs are skipped automatically.
- For Mahout-only runs, you can uninstall the competitor frameworks:
  `uv pip uninstall qiskit pennylane`.
- **Statistical mode** requires the `benchmark_utils` package (automatically available
  when running from this directory).
- **Visualization mode** additionally requires matplotlib, seaborn, and pandas.

## Notebooks

### Interactive Tutorials

We provide Jupyter notebooks for learning and experimentation:

1. **[Statistical Benchmark Tutorial](./notebooks/statistical_benchmark_tutorial.ipynb)** - NEW!
   - Complete walkthrough of statistical benchmarking features
   - Code examples for all APIs
   - Best practices demonstrations
   - Interactive visualization

2. **[Mahout Benchmark on Colab](./notebooks/mahout_benchmark.ipynb)**
   - Run benchmarks without owning a GPU
   - Quick start for beginners

[![Open Tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apache/mahout/blob/dev-qdp/qdp/qdp-python/benchmark/notebooks/statistical_benchmark_tutorial.ipynb)

### We can also run benchmarks on colab notebooks (without owning a GPU)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apache/mahout/blob/dev-qdp/qdp/qdp-python/benchmark/notebooks/mahout_benchmark.ipynb)

## Contributing

When contributing benchmark results or improvements:

1. **Follow best practices** - See [BEST_PRACTICES.md](./BEST_PRACTICES.md)
2. **Document configuration** - Include YAML config and system specs
3. **Use statistical mode** - For reliable, reproducible results
4. **Share visualizations** - Help others understand your results

For questions or issues, please open a GitHub issue or contact the Apache Mahout mailing list.
