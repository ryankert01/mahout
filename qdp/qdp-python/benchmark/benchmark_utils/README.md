# Benchmark Utils API Documentation

This directory contains utilities for fair, reproducible, and statistically rigorous benchmarking of Apache Mahout QDP.

## Quick Start

```python
from benchmark_utils import (
    benchmark_with_cuda_events,
    compute_statistics,
    BenchmarkVisualizer,
    load_config
)

# Define your operation to benchmark
def my_operation():
    # Your quantum operation here
    pass

# Run benchmark with CUDA events
timings = benchmark_with_cuda_events(
    my_operation,
    warmup_iters=3,
    repeat=50
)

# Compute statistics
stats = compute_statistics(timings)
print(f"Mean: {stats['mean']:.2f} ms ± {stats['std']:.2f} ms")

# Create visualizations
visualizer = BenchmarkVisualizer()
visualizer.plot_comparison_bars({
    'My Operation': stats
}, output_path='results.png')
```

## Modules

### `timing.py`

Provides CUDA event-based timing and warmup utilities.

**Key Functions:**
- `warmup(func, warmup_iters=3)`: Warm up JIT and GPU kernels
- `clear_all_caches()`: Clear Python GC and GPU caches
- `benchmark_with_cuda_events(func, warmup=3, repeat=100)`: Precise GPU timing
- `benchmark_cpu_function(func, warmup=3, repeat=100)`: CPU timing

**Example:**
```python
from benchmark_utils.timing import benchmark_with_cuda_events

def my_kernel():
    x = torch.randn(1000, 1000, device='cuda')
    return x @ x.T

timings = benchmark_with_cuda_events(my_kernel, warmup=5, repeat=100)
print(f"Average time: {sum(timings)/len(timings):.2f} ms")
```

### `statistics.py`

Statistical computations for benchmark results.

**Key Functions:**
- `compute_statistics(timings)`: Compute mean, median, std, percentiles, etc.
- `filter_outliers(timings, method='iqr', threshold=1.5)`: Remove outliers
- `compute_confidence_interval(timings, confidence=0.95)`: Compute CI
- `format_statistics(stats)`: Pretty-print statistics

**Example:**
```python
from benchmark_utils.statistics import compute_statistics, filter_outliers

timings = [10.1, 10.2, 10.3, 50.0, 10.2]  # 50.0 is outlier
filtered = filter_outliers(timings)
stats = compute_statistics(filtered)

print(f"Mean: {stats['mean']:.2f} ms")
print(f"P95: {stats['p95']:.2f} ms")
print(f"CV: {stats['cv']*100:.2f}%")
```

### `visualization.py`

Publication-ready plot generation.

**Key Class:**
- `BenchmarkVisualizer`: Create bar charts, box plots, violin plots, tables

**Example:**
```python
from benchmark_utils.visualization import BenchmarkVisualizer

visualizer = BenchmarkVisualizer(style='seaborn-v0_8-paper')

# Create all plots at once
visualizer.create_all_plots(
    results={'Framework A': stats_a, 'Framework B': stats_b},
    results_raw={'Framework A': timings_a, 'Framework B': timings_b},
    output_dir='./results',
    prefix='my_benchmark'
)
```

### `config.py`

Configuration loading and management.

**Key Classes:**
- `BenchmarkConfig`: Complete benchmark configuration
- `FairnessConfig`: Warmup, cache clearing settings
- `StatisticsConfig`: Percentiles, outlier detection settings
- `VisualizationConfig`: Plot output settings

**Example:**
```python
from benchmark_utils.config import BenchmarkConfig

# Load from YAML
config = BenchmarkConfig.from_yaml('benchmark_config.yaml')

# Or use defaults
config = BenchmarkConfig.default()
config.fairness.warmup_iters = 5
config.statistics.outlier_detection = 'zscore'
```

## Complete Example

Here's a complete example benchmarking two frameworks:

```python
#!/usr/bin/env python3
import torch
from benchmark_utils import (
    benchmark_with_cuda_events,
    compute_statistics,
    BenchmarkVisualizer,
    load_config
)

# Load configuration
config = load_config('benchmark_config.yaml')

# Define operations to benchmark
def framework_a_operation():
    x = torch.randn(1000, 1000, device='cuda')
    return x @ x.T

def framework_b_operation():
    x = torch.randn(1000, 1000, device='cuda')
    return torch.mm(x, x.T)

# Run benchmarks
print("Benchmarking Framework A...")
timings_a = benchmark_with_cuda_events(
    framework_a_operation,
    warmup_iters=config.fairness.warmup_iters,
    repeat=config.fairness.repeat_runs
)

print("Benchmarking Framework B...")
timings_b = benchmark_with_cuda_events(
    framework_b_operation,
    warmup_iters=config.fairness.warmup_iters,
    repeat=config.fairness.repeat_runs
)

# Compute statistics
stats_a = compute_statistics(timings_a)
stats_b = compute_statistics(timings_b)

# Print results
print(f"\nFramework A: {stats_a['mean']:.2f} ms ± {stats_a['std']:.2f} ms")
print(f"Framework B: {stats_b['mean']:.2f} ms ± {stats_b['std']:.2f} ms")

# Create visualizations
visualizer = BenchmarkVisualizer()
visualizer.create_all_plots(
    results={'Framework A': stats_a, 'Framework B': stats_b},
    results_raw={'Framework A': timings_a, 'Framework B': timings_b},
    output_dir='./benchmark_results'
)

print("\nVisualizations saved to ./benchmark_results/")
```

## Configuration File Example

Create `benchmark_config.yaml`:

```yaml
fairness:
  warmup_iters: 3
  repeat_runs: 20
  clear_cache_between_runs: false
  use_cuda_events: true

statistics:
  collect_percentiles: [25, 50, 75, 90, 95, 99]
  outlier_detection: 'iqr'
  outlier_threshold: 1.5

visualization:
  output_dir: './benchmark_results'
  plot_formats: ['png', 'pdf']
  dpi: 300
  style: 'seaborn-v0_8-paper'
  plots:
    - 'bar_chart'
    - 'box_plot'
    - 'violin_plot'
    - 'comparison_table'

workloads:
  e2e:
    qubits: 16
    samples: 200
    frameworks: ['mahout-parquet', 'pennylane']
```

## Best Practices

1. **Always use warmup**: Run 3-5 warmup iterations before measurements
2. **Use CUDA events for GPU**: More accurate than `time.perf_counter()`
3. **Collect multiple runs**: At least 20 runs for statistical significance
4. **Report distributions**: Mean alone is insufficient, include P95, std
5. **Clear cache between frameworks**: Ensure fair comparison
6. **Filter outliers carefully**: Use IQR method with threshold=1.5
7. **Save raw data**: Keep timings_raw for post-processing

## Dependencies

Required:
- numpy

Optional (for full functionality):
- torch (for CUDA timing)
- matplotlib (for visualization)
- seaborn (for visualization)
- pandas (for visualization)
- pyyaml (for configuration)
- scipy (for confidence intervals)

Install with:
```bash
pip install numpy torch matplotlib seaborn pandas pyyaml scipy
```

## Testing

Run unit tests:
```bash
cd qdp/qdp-python
pytest tests/test_benchmark_utils.py -v

# Run only non-GPU tests
pytest tests/test_benchmark_utils.py -v -m "not gpu"

# Run GPU tests (requires CUDA)
pytest tests/test_benchmark_utils.py -v -m "gpu"
```

## License

Apache License 2.0 - See LICENSE file for details.
