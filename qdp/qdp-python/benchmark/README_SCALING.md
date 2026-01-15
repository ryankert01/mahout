# QDP Scaling Benchmark

A configurable benchmarking tool to compare quantum encoding throughput across frameworks (Mahout QDP vs PennyLane) and generate publication-quality plots.

## Quick Start

```bash
cd qdp/qdp-python

# Basic benchmark - saves CSV and PNG to benchmark/results/
uv run python benchmark/benchmark_scaling.py

# Use specific GPU
uv run python benchmark/benchmark_scaling.py --gpu 2

# Interactive mode (show plot, don't save)
uv run python benchmark/benchmark_scaling.py --no-save

# Custom output directory
uv run python benchmark/benchmark_scaling.py --output-dir ./my_results
```

## Output

By default, results are saved to `benchmark/results/` with timestamped filenames:

```
benchmark/results/
├── scaling_samples_throughput_20260115_130434.csv
└── scaling_samples_throughput_20260115_130434.png
```

## Usage

```
uv run python benchmark/benchmark_scaling.py [OPTIONS]
```

## Options

### X-Axis Configuration

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--x-axis` | `samples`, `qubits`, `batch_size` | `samples` | Parameter to vary on X-axis |
| `--samples` | list of ints | `100 250 500 1000 2000` | Sample counts to benchmark |
| `--qubits` | list of ints | `12` | Qubit counts (used as X-axis if `--x-axis qubits`) |

### Y-Axis Configuration

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--y-axis` | `throughput`, `latency` | `throughput` | Metric for Y-axis |

### Framework Selection

| Option | Values | Default | Description |
|--------|--------|---------|-------------|
| `--frameworks` | `mahout`, `pennylane`, `qiskit` | `mahout pennylane qiskit` | Frameworks to compare |

> **Note**: Qiskit is significantly slower (~100x) than other frameworks. For faster benchmarks, use `--frameworks mahout pennylane`.

### Benchmark Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--warmup` | int | `1` | Warmup runs per configuration |
| `--runs` | int | `3` | Measurement runs per configuration |
| `--gpu` | int | `0` | GPU device ID to use |

### Output Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output-dir`, `-o` | path | `benchmark/results` | Output directory for CSV and PNG |
| `--no-save` | flag | False | Don't save files, show plot interactively |
| `--title` | string | None | Custom plot title |
| `--log-x` | flag | False | Use log scale for X-axis |
| `--log-y` | flag | False | Use log scale for Y-axis |

## Examples

### 1. Basic Throughput vs Samples

```bash
uv run python benchmark/benchmark_scaling.py \
    --samples 500 1000 2000 5000 10000 \
    --qubits 12 \
    --output scaling_throughput.png
```

### 2. Throughput vs Qubits

```bash
uv run python benchmark/benchmark_scaling.py \
    --x-axis qubits \
    --qubits 8 10 12 14 16 \
    --samples 1000 \
    --output scaling_qubits.png
```

### 3. Latency Comparison

```bash
uv run python benchmark/benchmark_scaling.py \
    --y-axis latency \
    --samples 100 500 1000 \
    --output scaling_latency.png
```

### 4. High-Precision Benchmark

```bash
uv run python benchmark/benchmark_scaling.py \
    --warmup 3 \
    --runs 10 \
    --samples 1000 5000 10000 \
    --output scaling_precise.png
```

### 5. Skip Qiskit (Faster Runs)

```bash
uv run python benchmark/benchmark_scaling.py \
    --frameworks mahout pennylane \
    --samples 500 1000 2000 5000 10000
```

### 6. Single Framework (Mahout Only)

```bash
uv run python benchmark/benchmark_scaling.py \
    --frameworks mahout \
    --samples 500 1000 2000 5000
```

### 7. Using Specific GPU

```bash
# Use GPU 2
uv run python benchmark/benchmark_scaling.py \
    --gpu 2 \
    --samples 1000 5000 10000 \
    --output scaling_gpu2.png
```

### 8. Log Scale Plot

```bash
uv run python benchmark/benchmark_scaling.py \
    --samples 100 500 1000 5000 10000 50000 \
    --log-x \
    --output scaling_logx.png
```

### 9. Custom Title

```bash
uv run python benchmark/benchmark_scaling.py \
    --title "QDP vs PennyLane: 12-Qubit Encoding" \
    --samples 1000 5000 10000 \
    --output comparison.png
```

## Output

### Console Output

```
============================================================
SCALING BENCHMARK
============================================================
X-axis:     samples = [500, 1000, 2000, 5000, 10000]
Y-axis:     throughput
Qubits:     12
Frameworks: ['mahout', 'pennylane']
GPU:        0
Warmup:     1, Runs: 3
============================================================

============================================================
Benchmarking: samples=500, qubits=12, samples=500
============================================================

  Running mahout...
    Throughput: 780.30 samples/sec
    Latency: 1.2816 ms/sample

  Running pennylane...
    Throughput: 1547.52 samples/sec
    Latency: 0.6462 ms/sample

...

============================================================
RESULTS SUMMARY
============================================================
Framework    X-Value    Throughput (s/s)   Latency (ms)
------------------------------------------------------------
Mahout       500        780.30             1.2816
PennyLane    500        1547.52            0.6462
...
```

### Plot Output

The tool generates a line plot with:
- **X-axis**: Configured parameter (samples, qubits, or batch_size)
- **Y-axis**: Throughput (samples/sec) or latency (ms/sample)
- **Lines**: One colored line per framework
- **Markers**: Data points at each measurement

## Understanding Results

### Throughput Characteristics

| Framework | Behavior | Best For |
|-----------|----------|----------|
| **Mahout** | Scales linearly with sample count | Large batches (1000+ samples) |
| **PennyLane** | Constant throughput (~1500 s/s) | Small batches (<1000 samples) |
| **Qiskit** | Constant, very slow (~10 s/s) | Correctness verification only |

### Crossover Point

At approximately **1000-1500 samples**, Mahout's throughput matches PennyLane. Beyond this point, Mahout is faster due to GPU batch processing that amortizes initialization overhead.

### Typical Results (12 qubits)

| Samples | Mahout | PennyLane | Qiskit | Winner |
|---------|--------|-----------|--------|--------|
| 500 | ~780 s/s | ~1,500 s/s | ~11 s/s | PennyLane 2x |
| 1,000 | ~1,400 s/s | ~1,500 s/s | ~11 s/s | Equal |
| 5,000 | ~4,500 s/s | ~1,500 s/s | ~11 s/s | Mahout 3x |
| 10,000 | ~6,500 s/s | ~1,500 s/s | ~11 s/s | Mahout 4x |

### Relative Performance

| Comparison | Speedup |
|------------|---------|
| PennyLane vs Qiskit | ~130x faster |
| Mahout vs Qiskit (at 10k samples) | ~590x faster |
| Mahout vs PennyLane (at 10k samples) | ~4x faster |

## Tips

1. **For accurate results**: Use `--warmup 3 --runs 10` or higher
2. **For quick testing**: Use `--warmup 1 --runs 2` with fewer samples
3. **For publication**: Use `--warmup 5 --runs 30` with comprehensive sample range
4. **GPU selection**: Always specify `--gpu` to avoid resource conflicts
5. **Large sample counts**: May require significant GPU memory at higher qubit counts

## Troubleshooting

### "Mahout not available"
- Ensure QDP is built: `uv run maturin develop`

### "PennyLane not available"
- Install PennyLane: `uv add pennylane`

### Out of GPU memory
- Reduce sample count or qubit count
- Use a GPU with more VRAM

### Inconsistent results
- Increase warmup runs (`--warmup 3`)
- Increase measurement runs (`--runs 10`)
- Ensure no other GPU processes are running
