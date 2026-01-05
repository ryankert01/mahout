# Benchmark Roadmap (RFC) for Apache Mahout QDP

**Status:** Draft  
**Authors:** Apache Mahout QDP Team  
**Created:** 2026-01-05  
**Last Updated:** 2026-01-05

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State](#current-state)
3. [Problem Statement](#problem-statement)
4. [Goals and Objectives](#goals-and-objectives)
5. [Proposed Improvements](#proposed-improvements)
   - [5.1 Fairness Improvements](#51-fairness-improvements)
   - [5.2 Statistical Measurements](#52-statistical-measurements)
   - [5.3 Visualization for Publications](#53-visualization-for-publications)
6. [Implementation Details](#implementation-details)
7. [Best Practices and References](#best-practices-and-references)
8. [Roadmap and Timeline](#roadmap-and-timeline)
9. [Open Questions](#open-questions)

---

## 1. Executive Summary

This RFC proposes a comprehensive improvement to Apache Mahout QDP's benchmarking infrastructure to ensure **fairness**, **reproducibility**, and **publication-ready visualization**. The roadmap focuses on:

1. **Fairness**: Implementing proper warmup, cache clearing, and CUDA event synchronization
2. **Statistical rigor**: Collecting multiple runs with comprehensive statistics (mean, std, percentiles, min, max)
3. **Visualization**: Creating publication-quality graphs for blog posts and academic papers

The improvements are inspired by best practices from PyTorch Helion, Triton, and GPU-mode reference kernels.

---

## 2. Current State

### 2.1 Existing Benchmarks

Apache Mahout QDP currently has two main benchmark suites located in `qdp/qdp-python/benchmark/`:

1. **`benchmark_e2e.py`**: End-to-end latency benchmark (Disk → GPU VRAM)
   - Measures: IO, normalization, encoding, transfer, forward pass
   - Compares: Mahout (Parquet & Arrow) vs PennyLane vs Qiskit
   - Current implementation includes basic cache clearing via `clean_cache()`

2. **`benchmark_throughput.py`**: DataLoader throughput benchmark
   - Measures: vectors/sec for streaming batches
   - Uses prefetched batches to simulate real training loops
   - Compares: Mahout vs PennyLane vs Qiskit

### 2.2 Current Strengths

- ✅ Multi-framework comparison (Mahout, PennyLane, Qiskit)
- ✅ Real-world workloads (simulates actual QML training scenarios)
- ✅ Basic cache clearing with `gc.collect()` and `torch.cuda.empty_cache()`
- ✅ CUDA synchronization at benchmark boundaries
- ✅ Correctness verification (state comparison)

### 2.3 Current Limitations

- ❌ No warmup runs before timing
- ❌ Single-run measurements (no statistical analysis)
- ❌ No detailed profiling metrics (percentiles, std deviation, etc.)
- ❌ Limited visualization output
- ❌ No systematic event-based timing (relies on `time.perf_counter()`)
- ❌ Inconsistent cache clearing strategy

---

## 3. Problem Statement

Current benchmarks may produce **unfair** or **unreproducible** results due to:

1. **Cold start effects**: First run includes JIT compilation, library initialization
2. **Cache effects**: GPU L2 cache, CPU cache not systematically cleared
3. **Timing precision**: `time.perf_counter()` doesn't account for GPU async execution
4. **Statistical variance**: Single runs don't capture performance distribution
5. **Publication needs**: No automated graph generation for papers/blogs

These issues make it difficult to:
- Confidently compare frameworks
- Identify performance regressions
- Publish results in academic papers
- Generate blog post visualizations

---

## 4. Goals and Objectives

### Primary Goals

1. **Fairness**: Ensure all frameworks are measured under identical conditions
2. **Reproducibility**: Enable consistent results across runs and environments
3. **Statistical rigor**: Provide confidence intervals and distribution metrics
4. **Publication-ready**: Generate graphs suitable for academic papers and blog posts

### Non-Goals

- Gradient computation benchmarks (not needed for quantum state preparation)
- Multi-GPU benchmarks (out of scope for initial implementation)
- Distributed benchmarks (future work)

---

## 5. Proposed Improvements

### 5.1 Fairness Improvements

#### 5.1.1 Warmup Runs

**Problem**: First run includes JIT compilation and library initialization overhead.

**Solution**: Implement warmup phase before actual measurements.

```python
def warmup(func, warmup_iters=3):
    """Run function multiple times to warm up JIT, caches, etc."""
    for _ in range(warmup_iters):
        func()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
```

**Recommendation**: 
- Use 3-5 warmup iterations for small workloads
- Use 1-2 warmup iterations for large workloads (>1 minute per run)

#### 5.1.2 Cache Clearing

**Problem**: Inconsistent cache state affects reproducibility.

**Enhancement**: Comprehensive cache clearing between runs.

```python
def clear_all_caches():
    """Clear all caches to ensure fair benchmarking."""
    # Python garbage collection
    gc.collect()
    
    if torch.cuda.is_available():
        # Clear PyTorch GPU cache
        torch.cuda.empty_cache()
        
        # Synchronize all CUDA operations
        torch.cuda.synchronize()
        
        # Reset peak memory stats (useful for memory profiling)
        torch.cuda.reset_peak_memory_stats()
        
        # Optional: Clear L2 cache by allocating and freeing large tensor
        # This is more invasive and may not be needed for all benchmarks
        # cache_clear_tensor = torch.empty(
        #     (1024, 1024, 256), dtype=torch.float32, device='cuda'
        # )
        # del cache_clear_tensor
        # torch.cuda.empty_cache()
```

**When to clear**:
- Before warmup runs
- Between different frameworks
- Between repeated runs (for statistical measurements)

#### 5.1.3 CUDA Event-Based Timing

**Problem**: `time.perf_counter()` doesn't account for GPU async execution.

**Solution**: Use CUDA events for precise GPU timing.

```python
def benchmark_with_cuda_events(func, warmup=3, repeat=100):
    """
    Benchmark a function using CUDA events for precise timing.
    
    Args:
        func: Callable to benchmark
        warmup: Number of warmup iterations
        repeat: Number of measurement iterations
    
    Returns:
        List of execution times in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        func()
        torch.cuda.synchronize()
    
    clear_all_caches()
    
    # Measurement
    timings = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        func()
        end_event.record()
        
        torch.cuda.synchronize()
        timings.append(start_event.elapsed_time(end_event))  # Returns ms
        
        # Optional: clear cache between runs for more conservative measurement
        # clear_all_caches()
    
    return timings
```

**Note**: For operations that don't involve gradients (like quantum state encoding), we can safely ignore gradient-related considerations mentioned in the original suggestion.

### 5.2 Statistical Measurements

#### 5.2.1 Metrics to Collect

Collect comprehensive statistics as suggested by Triton and Helion:

```python
import numpy as np

def compute_statistics(timings):
    """
    Compute comprehensive statistics from benchmark timings.
    
    Args:
        timings: List of timing measurements in milliseconds
    
    Returns:
        Dictionary with statistical metrics
    """
    timings_arr = np.array(timings)
    
    return {
        'mean': np.mean(timings_arr),
        'median': np.median(timings_arr),
        'std': np.std(timings_arr),
        'min': np.min(timings_arr),
        'max': np.max(timings_arr),
        'p25': np.percentile(timings_arr, 25),
        'p50': np.percentile(timings_arr, 50),  # Same as median
        'p75': np.percentile(timings_arr, 75),
        'p90': np.percentile(timings_arr, 90),
        'p95': np.percentile(timings_arr, 95),
        'p99': np.percentile(timings_arr, 99),
        'iqr': np.percentile(timings_arr, 75) - np.percentile(timings_arr, 25),
        'cv': np.std(timings_arr) / np.mean(timings_arr) if np.mean(timings_arr) > 0 else 0,  # Coefficient of variation
        'n_runs': len(timings_arr),
    }
```

#### 5.2.2 Recommended Number of Runs

Based on Triton's approach:
- **Fast operations** (<10ms): 100+ runs
- **Medium operations** (10ms-1s): 50-100 runs
- **Slow operations** (>1s): 10-20 runs

For our benchmarks:
- **Throughput benchmark**: 20-50 runs (each run is ~seconds)
- **E2E benchmark**: 10-20 runs (each run is ~seconds to minutes)

#### 5.2.3 Outlier Detection

Implement outlier filtering to remove anomalous measurements:

```python
def filter_outliers(timings, method='iqr', threshold=1.5):
    """
    Remove outliers from timing measurements.
    
    Args:
        timings: List of measurements
        method: 'iqr' (interquartile range) or 'zscore'
        threshold: Multiplier for IQR or z-score threshold
    
    Returns:
        Filtered timings
    """
    timings_arr = np.array(timings)
    
    if method == 'iqr':
        q1 = np.percentile(timings_arr, 25)
        q3 = np.percentile(timings_arr, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return timings_arr[(timings_arr >= lower) & (timings_arr <= upper)]
    
    elif method == 'zscore':
        z_scores = np.abs((timings_arr - np.mean(timings_arr)) / np.std(timings_arr))
        return timings_arr[z_scores < threshold]
    
    return timings_arr
```

### 5.3 Visualization for Publications

#### 5.3.1 Required Plots

Generate publication-ready visualizations using matplotlib:

1. **Bar charts with error bars** (mean ± std)
2. **Box plots** (showing quartiles and outliers)
3. **Violin plots** (showing full distribution)
4. **Performance comparison tables**

#### 5.3.2 Implementation

```python
import matplotlib.pyplot as plt
import seaborn as sns

class BenchmarkVisualizer:
    """Create publication-ready benchmark visualizations."""
    
    def __init__(self, style='seaborn-v0_8-paper'):
        """Initialize with publication style."""
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 8)
    
    def plot_comparison_bars(self, results, metric='mean', output_path='benchmark_bars.png'):
        """
        Create bar chart comparing frameworks.
        
        Args:
            results: Dict of {framework_name: statistics_dict}
            metric: Which metric to plot (mean, median, etc.)
            output_path: Where to save the plot
        """
        frameworks = list(results.keys())
        values = [results[fw][metric] for fw in frameworks]
        errors = [results[fw]['std'] for fw in frameworks]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(frameworks))
        
        bars = ax.bar(x_pos, values, yerr=errors, capsize=5, 
                     alpha=0.8, color=self.colors[:len(frameworks)])
        
        ax.set_xlabel('Framework', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title(f'Benchmark Comparison ({metric.capitalize()})', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(frameworks, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved bar chart to {output_path}")
        plt.close()
    
    def plot_box_comparison(self, results_raw, output_path='benchmark_box.png'):
        """
        Create box plot showing distributions.
        
        Args:
            results_raw: Dict of {framework_name: [list of timings]}
            output_path: Where to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        frameworks = list(results_raw.keys())
        data = [results_raw[fw] for fw in frameworks]
        
        bp = ax.boxplot(data, labels=frameworks, patch_artist=True,
                       showmeans=True, meanline=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], self.colors[:len(frameworks)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Framework', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Benchmark Distribution Comparison (Box Plot)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved box plot to {output_path}")
        plt.close()
    
    def plot_violin_comparison(self, results_raw, output_path='benchmark_violin.png'):
        """
        Create violin plot showing full distributions.
        
        Args:
            results_raw: Dict of {framework_name: [list of timings]}
            output_path: Where to save the plot
        """
        # Prepare data for seaborn
        import pandas as pd
        
        data_list = []
        for framework, timings in results_raw.items():
            for timing in timings:
                data_list.append({'Framework': framework, 'Time (ms)': timing})
        
        df = pd.DataFrame(data_list)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=df, x='Framework', y='Time (ms)', 
                      palette=self.colors, ax=ax)
        
        ax.set_xlabel('Framework', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Benchmark Distribution Comparison (Violin Plot)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved violin plot to {output_path}")
        plt.close()
    
    def create_comparison_table(self, results, output_path='benchmark_table.md'):
        """
        Create markdown table with all statistics.
        
        Args:
            results: Dict of {framework_name: statistics_dict}
            output_path: Where to save the table
        """
        frameworks = sorted(results.keys())
        
        # Create markdown table
        lines = []
        lines.append("# Benchmark Results\n")
        lines.append("| Framework | Mean (ms) | Median (ms) | Std (ms) | Min (ms) | Max (ms) | P95 (ms) | Runs |")
        lines.append("|-----------|-----------|-------------|----------|----------|----------|----------|------|")
        
        for fw in frameworks:
            stats = results[fw]
            lines.append(
                f"| {fw} | {stats['mean']:.2f} | {stats['median']:.2f} | "
                f"{stats['std']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} | "
                f"{stats['p95']:.2f} | {stats['n_runs']} |"
            )
        
        table_content = '\n'.join(lines)
        
        with open(output_path, 'w') as f:
            f.write(table_content)
        
        print(f"Saved comparison table to {output_path}")
        print(table_content)
```

---

## 6. Implementation Details

### 6.1 Refactored Benchmark Structure

Proposed new structure for benchmarks:

```
qdp/qdp-python/benchmark/
├── benchmark_e2e.py          # Existing E2E benchmark
├── benchmark_throughput.py   # Existing throughput benchmark
├── benchmark_utils.py        # NEW: Shared utilities
│   ├── timing.py            # CUDA event timing, warmup
│   ├── statistics.py        # Statistical computations
│   └── visualization.py     # Plot generation
└── benchmark_fair.py         # NEW: Fair benchmark runner
```

### 6.2 Minimal Changes to Existing Benchmarks

To maintain backward compatibility, we'll:

1. **Create new utility modules** without modifying existing benchmarks
2. **Add optional flags** to existing benchmarks for statistical mode
3. **Provide standalone fair benchmark runner** that wraps existing benchmarks

Example CLI enhancement:
```bash
# Existing behavior (backward compatible)
python benchmark_e2e.py --qubits 16 --samples 200

# New statistical mode
python benchmark_e2e.py --qubits 16 --samples 200 \
  --statistical --warmup 3 --repeat 20 --visualize
```

### 6.3 Configuration File

Create `benchmark_config.yaml` for reproducibility:

```yaml
# Benchmark configuration for fair comparison
fairness:
  warmup_iters: 3
  repeat_runs: 20
  clear_cache_between_runs: false  # Conservative: false, Aggressive: true
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
  
  throughput:
    qubits: 16
    batches: 200
    batch_size: 64
    prefetch: 16
    frameworks: ['mahout', 'pennylane']
```

---

## 7. Best Practices and References

### 7.1 Referenced Implementations

1. **PyTorch Helion**
   - Autotuner benchmarking: https://github.com/pytorch/helion/blob/main/helion/autotuner/benchmarking.py
   - Search statistics: https://github.com/pytorch/helion/blob/main/helion/autotuner/base_search.py#L974

2. **Triton**
   - Testing utilities: https://github.com/triton-lang/triton/blob/main/python/triton/testing.py#L127
   - Statistical collection: https://github.com/triton-lang/triton/blob/main/python/triton/testing.py#L42

3. **GPU-mode Reference Kernels**
   - Better benchmarking: https://github.com/gpu-mode/reference-kernels/blob/main/problems/nvidia/eval_better_bench.py#L109

### 7.2 Key Takeaways

From the references:

1. **Always warmup**: 3-5 iterations before timing
2. **Use CUDA events**: More precise than CPU-side timing
3. **Collect distributions**: Mean alone is insufficient
4. **Report percentiles**: P50, P95, P99 are critical
5. **Clear cache thoughtfully**: Between frameworks (always), between runs (optional)
6. **No gradients needed**: For inference/encoding benchmarks, skip gradient-related code

### 7.3 Common Pitfalls to Avoid

- ❌ Measuring first run (includes compilation)
- ❌ Single measurement (doesn't capture variance)
- ❌ Ignoring outliers (can skew results)
- ❌ Not synchronizing CUDA (async execution)
- ❌ Comparing apples to oranges (different cache states)

---

## 8. Roadmap and Timeline

### Phase 1: Foundation (Weeks 1-2)

- [ ] Create `benchmark_utils/` module structure
  - [ ] `timing.py`: CUDA event timing, warmup
  - [ ] `statistics.py`: Statistical computations
  - [ ] `visualization.py`: Plot generation
  - [ ] `config.py`: Configuration loading
- [ ] Write unit tests for utilities
- [ ] Document API and usage examples

### Phase 2: Integration (Weeks 3-4)

- [ ] Add `--statistical` flag to existing benchmarks
- [ ] Integrate CUDA event timing
- [ ] Implement warmup mechanism
- [ ] Add statistical output to console

### Phase 3: Visualization (Week 5)

- [ ] Implement bar chart generation
- [ ] Implement box plot generation
- [ ] Implement violin plot generation
- [ ] Create markdown table export
- [ ] Add `--visualize` flag to benchmarks

### Phase 4: Documentation (Week 6)

- [ ] Update `qdp/qdp-python/benchmark/README.md`
- [ ] Add visualization examples
- [ ] Create tutorial notebook
- [ ] Document best practices for reproducibility

### Phase 5: Validation (Week 7)

- [ ] Run benchmarks with new infrastructure
- [ ] Compare old vs new methodology
- [ ] Generate sample blog post graphs
- [ ] Create sample academic paper figures

### Phase 6: Publication (Week 8)

- [ ] Blog post with results and visualizations
- [ ] Update documentation website
- [ ] Submit to Apache Mahout mailing list for feedback

---

## 9. Open Questions

### 9.1 Technical Questions

1. **Cache clearing aggressiveness**: Should we clear cache between every run or only between frameworks?
   - **Recommendation**: Between frameworks (always), between runs (optional, configurable)

2. **Number of runs**: What's the right balance between accuracy and runtime?
   - **Recommendation**: Start with 20 runs, adjust based on coefficient of variation

3. **Outlier handling**: Should we automatically filter outliers or report them?
   - **Recommendation**: Report all data, but also provide filtered statistics

### 9.2 Process Questions

1. **Backward compatibility**: Should we maintain old CLI interface?
   - **Recommendation**: Yes, add new flags as optional

2. **Default behavior**: Should statistical mode be default or opt-in?
   - **Recommendation**: Opt-in initially, default after validation

3. **CI integration**: Should we run statistical benchmarks in CI?
   - **Recommendation**: No (too slow), but run quick sanity checks

### 9.3 Future Enhancements

1. **Profiling integration**: Integrate with NSight Systems/Compute
2. **Automated regression detection**: Alert on significant performance changes
3. **Multi-GPU benchmarks**: Extend to distributed scenarios
4. **Power consumption metrics**: Measure energy efficiency
5. **Memory bandwidth analysis**: Add memory-focused benchmarks

---

## 10. Appendix: Example Output

### 10.1 Console Output with Statistics

```
======================================================================
E2E BENCHMARK: 16 Qubits, 200 Samples (Statistical Mode)
======================================================================

[Mahout-Parquet] Warmup (3 iterations)...
[Mahout-Parquet] Running 20 measurement iterations...
[Mahout-Parquet] Statistics:
  Mean:     125.34 ms
  Median:   124.89 ms
  Std:        3.21 ms
  Min:      120.12 ms
  Max:      132.45 ms
  P95:      130.23 ms
  P99:      131.87 ms
  CV:         2.56%
  Runs:          20

[PennyLane] Warmup (3 iterations)...
[PennyLane] Running 20 measurement iterations...
[PennyLane] Statistics:
  Mean:     456.78 ms
  Median:   455.23 ms
  Std:        8.45 ms
  Min:      442.34 ms
  Max:      478.90 ms
  P95:      471.23 ms
  P99:      476.12 ms
  CV:         1.85%
  Runs:          20

======================================================================
RESULTS SUMMARY
======================================================================
Framework        Mean (ms)   Median (ms)   Std (ms)   P95 (ms)
----------------------------------------------------------------------
Mahout-Parquet     125.34       124.89        3.21      130.23
PennyLane          456.78       455.23        8.45      471.23
----------------------------------------------------------------------
Speedup vs PennyLane: 3.64x (3.50x - 3.92x at 95% CI)

Visualizations saved to ./benchmark_results/
  - benchmark_bars.png
  - benchmark_box.png
  - benchmark_violin.png
  - benchmark_table.md
```

---

## 11. Conclusion

This roadmap provides a comprehensive plan for improving Apache Mahout QDP's benchmarking infrastructure with a focus on **fairness**, **statistical rigor**, and **publication-quality visualization**. The proposed changes are:

- **Minimal and incremental**: New utilities without breaking existing benchmarks
- **Well-referenced**: Based on best practices from PyTorch, Triton, and GPU-mode
- **Publication-ready**: Generates graphs suitable for academic papers and blog posts
- **Reproducible**: Clear methodology and configuration

**Next Steps**: Begin Phase 1 implementation and gather community feedback on this RFC.

---

## 12. References

1. PyTorch Helion Benchmarking: https://github.com/pytorch/helion/blob/main/helion/autotuner/benchmarking.py
2. Triton Testing Utilities: https://github.com/triton-lang/triton/blob/main/python/triton/testing.py
3. GPU-mode Better Benchmarking: https://github.com/gpu-mode/reference-kernels/blob/main/problems/nvidia/eval_better_bench.py
4. CUDA Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
5. "How to Benchmark Neural Networks" (blog): Various best practices for ML benchmarking

---

**Document Version**: 1.0  
**License**: Apache License 2.0  
**Maintainers**: Apache Mahout QDP Team
