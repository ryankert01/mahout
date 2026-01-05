# Best Practices for Reproducible Benchmarks

This document outlines best practices for running fair, reproducible benchmarks with Apache Mahout QDP's statistical benchmarking infrastructure.

## Table of Contents

1. [Fairness Principles](#fairness-principles)
2. [Statistical Rigor](#statistical-rigor)
3. [Reproducibility](#reproducibility)
4. [Publication Guidelines](#publication-guidelines)
5. [Common Pitfalls](#common-pitfalls)

## Fairness Principles

### 1. Always Use Warmup Iterations

**Why**: The first few executions include JIT compilation, GPU kernel compilation, and cache warmup overhead.

**How**: Use 3-5 warmup iterations for most workloads.

```bash
# GOOD: With warmup
python benchmark_e2e.py --statistical --warmup 5 --repeat 20

# BAD: No warmup (first run includes compilation overhead)
python benchmark_e2e.py  # Standard mode has no warmup
```

**Code example**:
```python
from benchmark_utils import benchmark_with_cuda_events

# Proper warmup
timings = benchmark_with_cuda_events(
    my_function,
    warmup_iters=5,    # Eliminate JIT overhead
    repeat=20
)
```

### 2. Clear Caches Between Framework Comparisons

**Why**: One framework may benefit from data cached by a previous framework.

**How**: Use `clear_all_caches()` between different frameworks.

```python
from benchmark_utils import clear_all_caches, benchmark_with_cuda_events

# Benchmark Framework A
clear_all_caches()
timings_framework_a = benchmark_with_cuda_events(func_a, warmup_iters=3, repeat=20)

# CRITICAL: Clear before Framework B
clear_all_caches()
timings_framework_b = benchmark_with_cuda_events(func_b, warmup_iters=3, repeat=20)
```

**Command line**:
```bash
# Statistical mode automatically clears caches between frameworks
python benchmark_e2e.py --statistical --frameworks mahout-parquet pennylane
```

### 3. Use CUDA Events for GPU Timing

**Why**: `time.perf_counter()` includes CPU-GPU synchronization overhead and is less precise.

**How**: The statistical mode automatically uses CUDA events.

```python
# GOOD: CUDA events (millisecond precision)
timings = benchmark_with_cuda_events(gpu_function, warmup_iters=3, repeat=20)

# BAD: CPU timing for GPU operations
import time
start = time.perf_counter()
gpu_function()
duration = time.perf_counter() - start  # Inaccurate for GPU!
```

### 4. Synchronize at Measurement Boundaries

**Why**: GPU operations are asynchronous; must ensure completion before timing.

**How**: Use `torch.cuda.synchronize()` or CUDA events (which handle this automatically).

```python
import torch

# GOOD: Proper synchronization
torch.cuda.synchronize()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
gpu_operation()
end_event.record()

torch.cuda.synchronize()
duration_ms = start_event.elapsed_time(end_event)

# BAD: No synchronization
start = time.time()
gpu_operation()  # Returns immediately, GPU still running!
duration = time.time() - start  # Wrong!
```

## Statistical Rigor

### 1. Report Full Distributions, Not Just Means

**Why**: Mean alone hides variability and can be misleading.

**How**: Always report mean, median, std, and percentiles.

```python
from benchmark_utils import compute_statistics, format_statistics

stats = compute_statistics(timings)

# GOOD: Full distribution
print(format_statistics(stats))
# Shows: mean, median, std, min, max, P95, CV, n_runs

# BAD: Only mean
print(f"Mean: {sum(timings)/len(timings):.2f} ms")
```

**Key metrics to report**:
- **Mean**: Average performance
- **Median (P50)**: Typical performance
- **P95/P99**: Worst-case performance (important for latency-sensitive apps)
- **Std**: Variability
- **CV (Coefficient of Variation)**: Normalized variability (std/mean)

### 2. Use Sufficient Repetitions

**Why**: More repetitions → more reliable statistics.

**Guidelines**:
- **Fast operations (< 1ms)**: 100+ repetitions
- **Medium operations (1-100ms)**: 20-50 repetitions
- **Slow operations (> 100ms)**: 10-20 repetitions

```bash
# Adjust based on operation speed
python benchmark_e2e.py --statistical --repeat 20  # For ~50ms operations
python benchmark_throughput.py --statistical --repeat 10  # For ~200ms operations
```

### 3. Consider Outlier Filtering

**When**: If you have sporadic system interference (background tasks, thermal throttling).

**How**: Use IQR-based filtering with threshold=1.5 (standard) or 3.0 (conservative).

```python
from benchmark_utils import filter_outliers, compute_statistics

# Run benchmark
timings = benchmark_with_cuda_events(my_func, warmup_iters=5, repeat=50)

# Remove outliers
filtered_timings = filter_outliers(timings, method='iqr', threshold=1.5)

# Compute statistics on filtered data
stats = compute_statistics(filtered_timings)
print(f"Removed {len(timings) - len(filtered_timings)} outliers")
```

**Caution**: Only filter if you understand why outliers occur. Don't filter to make results look better!

### 4. Report Confidence Intervals

**Why**: Shows uncertainty in measurements.

**How**: Use 95% confidence intervals for the mean.

```python
from benchmark_utils import compute_confidence_interval

lower, upper = compute_confidence_interval(timings, confidence=0.95)
print(f"Mean: {stats['mean']:.2f} ms")
print(f"95% CI: [{lower:.2f}, {upper:.2f}] ms")
```

## Reproducibility

### 1. Save Benchmark Configuration

**Why**: Others need to know exact settings to reproduce results.

**How**: Use YAML configuration files.

```python
from benchmark_utils.config import BenchmarkConfig

# Create and customize configuration
config = BenchmarkConfig.default()
config.fairness.warmup_iters = 5
config.fairness.repeat_runs = 20
config.visualization.output_dir = "./results"

# Save for reproducibility
config.to_yaml('my_benchmark_config.yaml')
```

Include in your repository:
```yaml
# benchmark_config.yaml
fairness:
  warmup_iters: 5
  repeat_runs: 20
  clear_cache_between_runs: false
  use_cuda_events: true

statistics:
  collect_percentiles: [25, 50, 75, 90, 95, 99]
  outlier_detection: 'iqr'
  outlier_threshold: 1.5

visualization:
  output_dir: './benchmark_results'
  dpi: 300
  style: 'seaborn-v0_8-paper'
```

### 2. Document System Configuration

Always report:
- **Hardware**: GPU model, CPU, RAM
- **Software**: CUDA version, PyTorch version, driver version
- **Data**: Dataset size, characteristics
- **Settings**: Warmup, repeat, batch size, etc.

**Example**:
```markdown
## Benchmark Configuration

**Hardware:**
- GPU: NVIDIA A100 40GB
- CPU: AMD EPYC 7742 (64 cores)
- RAM: 512GB DDR4

**Software:**
- CUDA: 11.8
- PyTorch: 2.0.1
- Driver: 520.61.05
- Mahout QDP: 0.1.0

**Settings:**
- Warmup: 5 iterations
- Repeat: 20 measurements
- Dataset: 200 samples, 16 qubits
- Frameworks: Mahout (Parquet), PennyLane

**Results:** See [benchmark_results/](./benchmark_results/)
```

### 3. Version Control Results

**Do**:
- ✅ Save configuration files
- ✅ Save generated plots (PNG, PDF)
- ✅ Save statistics tables (Markdown)
- ✅ Document git commit hash

**Don't**:
- ❌ Commit raw timing data (too large)
- ❌ Commit generated Python cache (`__pycache__`)
- ❌ Commit temporary benchmark data files

Use `.gitignore`:
```
# Benchmark artifacts
*.parquet
*.arrow
final_benchmark_data.*

# Generated data
__pycache__/
*.pyc

# Keep plots and tables
!benchmark_results/*.png
!benchmark_results/*.md
```

## Publication Guidelines

### 1. Choose Appropriate Plots

**For papers**:
- Bar charts with error bars (mean ± std)
- Box plots (show quartiles)
- Use 300 DPI for print quality

**For blog posts**:
- Violin plots (show full distribution)
- Include comparison tables
- Can use lower DPI (150-200)

```bash
# Generate publication-quality plots
python benchmark_e2e.py --statistical --visualize \
  --warmup 5 --repeat 20 \
  --output-dir ./paper_figures
```

### 2. Report Effect Sizes, Not Just Speedups

**Bad**: "Framework A is 3.2x faster"

**Good**: 
```
Framework A: 12.5ms ± 0.8ms (P95: 14.2ms)
Framework B: 40.1ms ± 2.1ms (P95: 44.8ms)
Speedup: 3.2x (95% CI: [2.9x, 3.5x])
```

### 3. Include Visualization Examples

When writing papers/blogs, include:

1. **Bar chart**: Quick comparison
2. **Box or violin plot**: Show variability
3. **Statistics table**: Complete data

```bash
# Generate all three
python benchmark_e2e.py --statistical --visualize \
  --frameworks mahout-parquet pennylane qiskit \
  --output-dir ./figures
```

The command generates:
- `e2e_q16_s200_bars.png` → Use in introduction
- `e2e_q16_s200_box.png` → Use in results section
- `e2e_q16_s200_table.md` → Include in appendix

## Common Pitfalls

### ❌ Pitfall 1: No Warmup

**Problem**: First run includes JIT compilation overhead.

**Solution**: Always use `--warmup` or `warmup_iters`.

### ❌ Pitfall 2: Comparing Cached vs Uncached

**Problem**: Framework A runs first, loads data into cache. Framework B benefits from cached data.

**Solution**: Clear caches between frameworks (automatic in statistical mode).

### ❌ Pitfall 3: Using CPU Timing for GPU Operations

**Problem**: `time.time()` or `time.perf_counter()` don't account for GPU async execution.

**Solution**: Use CUDA events (automatic in statistical mode).

### ❌ Pitfall 4: Cherry-Picking Results

**Problem**: Running benchmark multiple times and reporting best result.

**Solution**: Run once with sufficient repetitions, report full distribution.

### ❌ Pitfall 5: Hiding Variability

**Problem**: Only reporting mean, hiding high variance.

**Solution**: Report mean ± std and percentiles.

### ❌ Pitfall 6: Insufficient Repetitions

**Problem**: With 3-5 runs, statistics are unreliable.

**Solution**: Use 20+ repetitions for stable statistics.

### ❌ Pitfall 7: Not Documenting Configuration

**Problem**: Others can't reproduce your results.

**Solution**: Save configuration to YAML, document system specs.

## Quick Reference

### Command Line Cheat Sheet

```bash
# Minimal statistical benchmark
python benchmark_e2e.py --statistical

# Full publication-ready benchmark
python benchmark_e2e.py --statistical --visualize \
  --warmup 5 --repeat 20 \
  --frameworks mahout-parquet pennylane \
  --output-dir ./paper_results

# Throughput with visualization
python benchmark_throughput.py --statistical --visualize \
  --warmup 2 --repeat 15 \
  --frameworks mahout,pennylane
```

### Python API Cheat Sheet

```python
from benchmark_utils import (
    benchmark_with_cuda_events,
    compute_statistics,
    format_statistics,
    BenchmarkVisualizer,
)

# 1. Benchmark
timings = benchmark_with_cuda_events(
    my_func,
    warmup_iters=5,
    repeat=20
)

# 2. Statistics
stats = compute_statistics(timings)
print(format_statistics(stats))

# 3. Visualize
visualizer = BenchmarkVisualizer()
visualizer.create_all_plots(
    results={'My Method': stats},
    results_raw={'My Method': timings},
    output_dir='./results'
)
```

## Further Reading

- [Benchmark Roadmap RFC](../../docs/BENCHMARK_ROADMAP.md) - Complete design document
- [Benchmark Utils API](../benchmark_utils/README.md) - Detailed API documentation
- [Tutorial Notebook](./notebooks/statistical_benchmark_tutorial.ipynb) - Interactive examples
- [PyTorch Helion Benchmarking](https://github.com/pytorch/helion/blob/main/helion/autotuner/benchmarking.py) - Reference implementation
- [Triton Testing](https://github.com/triton-lang/triton/blob/main/python/triton/testing.py) - Best practices from Triton

## Contributing

Found an issue or have suggestions? Please:
1. Open an issue on GitHub
2. Include your configuration and system specs
3. Provide reproducible example

---

**Last Updated**: Phase 4 Documentation (2026-01-05)

**Authors**: Apache Mahout QDP Team
