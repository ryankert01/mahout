# QDP Benchmark Suite Roadmap

This roadmap outlines the development of a comprehensive, publication-quality benchmark suite for QDP quantum encoding performance.

## Overview

### Goals
1. **Fairness**: Proper warmup, cache clearing, CUDA event synchronization
2. **Statistical Rigor**: Multiple runs with comprehensive statistics (mean, std, percentiles, min, max)
3. **Real Datasets**: Test with respected ML datasets (MNIST, Iris, etc.)
4. **Hyperparameter Exploration**: Systematic testing across qubits, samples, batch sizes
5. **Publication-Quality Visualization**: Professional graphs for blogs, presentations, papers

### Current State
- 4 benchmark scripts exist (E2E, latency, throughput, NumPy I/O)
- Console-based reporting with basic comparisons
- Synthetic data generation only
- Single-run measurements

---

## Phase 1: Benchmark Infrastructure Foundation

### Issue 1.1: Create Statistical Measurement Framework
**Scope**: Build a reusable statistics collection module

**Acceptance Criteria**:
- [ ] Create `benchmark/core/statistics.py` module
- [ ] Implement `BenchmarkStats` dataclass with: mean, std, min, max, median, p5, p25, p75, p95, p99
- [ ] Implement `StatisticalRunner` class that:
  - Accepts a callable benchmark function
  - Runs configurable warmup iterations (default: 3)
  - Runs configurable measurement iterations (default: 10)
  - Returns `BenchmarkStats` object
- [ ] Add outlier detection (IQR-based) with optional filtering
- [ ] Unit tests for statistics calculations

**Files to Create/Modify**:
- `benchmark/core/__init__.py`
- `benchmark/core/statistics.py`
- `benchmark/tests/test_statistics.py`

**Example API**:
```python
from benchmark.core.statistics import StatisticalRunner, BenchmarkStats

runner = StatisticalRunner(warmup_runs=3, measurement_runs=10)
stats: BenchmarkStats = runner.run(my_benchmark_fn, *args)
print(f"Mean: {stats.mean:.3f} ± {stats.std:.3f} ms")
print(f"P95: {stats.p95:.3f} ms")
```

---

### Issue 1.2: Implement CUDA Synchronization & Cache Management
**Scope**: Ensure fair GPU benchmarking with proper synchronization

**Acceptance Criteria**:
- [ ] Create `benchmark/core/cuda_utils.py` module
- [ ] Implement `CUDATimer` context manager using CUDA events (not wall-clock time)
- [ ] Implement `clear_gpu_caches()` function that:
  - Calls `torch.cuda.empty_cache()`
  - Calls `torch.cuda.synchronize()`
  - Optionally triggers Python garbage collection
- [ ] Implement `warmup_gpu()` function with dummy kernel launches
- [ ] Add `@cuda_benchmark` decorator that wraps functions with sync/clear logic
- [ ] Unit tests verifying synchronization behavior

**Files to Create/Modify**:
- `benchmark/core/cuda_utils.py`
- `benchmark/tests/test_cuda_utils.py`

**Example API**:
```python
from benchmark.core.cuda_utils import CUDATimer, clear_gpu_caches, cuda_benchmark

with CUDATimer() as timer:
    result = encode_data(batch)
print(f"GPU time: {timer.elapsed_ms:.3f} ms")

@cuda_benchmark(warmup=True, clear_cache=True)
def benchmark_encoding(data):
    return qdp_engine.extract(data)
```

---

### Issue 1.3: Create Benchmark Configuration System
**Scope**: YAML/JSON-based configuration for reproducible benchmarks

**Acceptance Criteria**:
- [ ] Create `benchmark/core/config.py` module
- [ ] Implement `BenchmarkConfig` dataclass with:
  - `warmup_runs: int`
  - `measurement_runs: int`
  - `clear_cache: bool`
  - `sync_cuda: bool`
  - `seed: int`
  - `output_dir: Path`
  - `export_formats: List[str]`  # ["csv", "json", "parquet"]
- [ ] Support YAML config file loading
- [ ] Support CLI argument overrides
- [ ] Implement config validation with helpful error messages
- [ ] Create default config file `benchmark/configs/default.yaml`

**Files to Create/Modify**:
- `benchmark/core/config.py`
- `benchmark/configs/default.yaml`
- `benchmark/configs/quick.yaml` (fewer runs for development)
- `benchmark/configs/publication.yaml` (many runs for papers)

---

## Phase 2: Dataset Infrastructure

### Issue 2.1: Create Dataset Abstraction Layer
**Scope**: Unified interface for all benchmark datasets

**Acceptance Criteria**:
- [ ] Create `benchmark/datasets/base.py` with abstract `BenchmarkDataset` class
- [ ] Define interface:
  - `name: str` - Dataset identifier
  - `category: str` - "image", "tabular", "scale"
  - `load() -> Tuple[np.ndarray, np.ndarray]` - Returns (features, labels)
  - `get_metadata() -> DatasetMetadata` - Returns shape, dtype, description
  - `prepare_for_qubits(n_qubits: int) -> np.ndarray` - Resize/pad for encoding
- [ ] Implement `DatasetRegistry` for dataset discovery
- [ ] Unit tests for base functionality

**Files to Create/Modify**:
- `benchmark/datasets/__init__.py`
- `benchmark/datasets/base.py`
- `benchmark/datasets/registry.py`
- `benchmark/tests/test_datasets_base.py`

**Example API**:
```python
from benchmark.datasets import DatasetRegistry

registry = DatasetRegistry()
mnist = registry.get("mnist-binary")
X, y = mnist.load()
X_quantum = mnist.prepare_for_qubits(n_qubits=10)
```

---

### Issue 2.2: Implement Image Datasets
**Scope**: Add MNIST and synthetic image datasets

**Acceptance Criteria**:
- [ ] Create `benchmark/datasets/image.py`
- [ ] Implement `MNISTBinaryDataset`:
  - Binary classification (0 vs 1, configurable)
  - Configurable downsampling (default: 4x4)
  - L2 normalization for amplitude encoding
  - Caching of downloaded data
- [ ] Implement `SyntheticImageDataset`:
  - Configurable resolution (default: 8x8)
  - Grayscale generation with patterns (gradients, noise, shapes)
  - Deterministic generation with seed
- [ ] Integration tests verifying quantum encoding compatibility
- [ ] Documentation with example usage

**Files to Create/Modify**:
- `benchmark/datasets/image.py`
- `benchmark/tests/test_datasets_image.py`

**Dependencies**: `scikit-learn` (for MNIST via fetch_openml or similar)

---

### Issue 2.3: Implement Tabular Datasets
**Scope**: Add Iris and synthetic tabular datasets

**Acceptance Criteria**:
- [ ] Create `benchmark/datasets/tabular.py`
- [ ] Implement `IrisBinaryDataset`:
  - Binary classification (2 classes, configurable)
  - Feature selection (default: 2 features for visualization)
  - Normalization options (L2, min-max, standard)
- [ ] Implement `SyntheticBlobsDataset`:
  - Configurable clusters, features, samples
  - Linear and non-linear separability options
  - Deterministic generation with seed
- [ ] Implement `CSVDataset`:
  - Load arbitrary CSV files
  - Configurable feature/label columns
  - Automatic normalization
- [ ] Integration tests

**Files to Create/Modify**:
- `benchmark/datasets/tabular.py`
- `benchmark/tests/test_datasets_tabular.py`

---

### Issue 2.4: Implement Scale Test Datasets
**Scope**: Large-scale datasets for stress testing

**Acceptance Criteria**:
- [ ] Create `benchmark/datasets/scale.py`
- [ ] Implement `LargeSyntheticDataset`:
  - Configurable sample count (10k, 100k, 1M)
  - Memory-efficient generation (chunked/streamed)
  - Option to persist to disk (Parquet/Arrow)
- [ ] Implement `LargeImageDataset`:
  - Synthetic large image batches
  - Measures memory + pipeline throughput
- [ ] Memory usage tracking during data loading
- [ ] Stress tests with memory profiling

**Files to Create/Modify**:
- `benchmark/datasets/scale.py`
- `benchmark/tests/test_datasets_scale.py`

---

## Phase 3: Framework Comparison Infrastructure

### Issue 3.1: Create Framework Abstraction Layer
**Scope**: Unified interface for comparing QML frameworks

**Acceptance Criteria**:
- [ ] Create `benchmark/frameworks/base.py` with abstract `QuantumFramework` class
- [ ] Define interface:
  - `name: str` - Framework identifier
  - `is_available() -> bool` - Check if framework is installed
  - `encode(data: np.ndarray, n_qubits: int) -> Any` - Amplitude encoding
  - `get_statevector() -> np.ndarray` - Extract quantum state
  - `cleanup()` - Release resources
- [ ] Implement `FrameworkRegistry` for framework discovery
- [ ] Graceful degradation when frameworks not installed

**Files to Create/Modify**:
- `benchmark/frameworks/__init__.py`
- `benchmark/frameworks/base.py`
- `benchmark/frameworks/registry.py`

---

### Issue 3.2: Implement Framework Adapters
**Scope**: Adapters for Mahout QDP, PennyLane, Qiskit

**Acceptance Criteria**:
- [ ] Create `benchmark/frameworks/mahout.py`:
  - QdpEngine wrapper with DLPack output
  - GPU tensor extraction
- [ ] Create `benchmark/frameworks/pennylane.py`:
  - AmplitudeEmbedding with state extraction
  - CPU and GPU (lightning.gpu) variants
- [ ] Create `benchmark/frameworks/qiskit.py`:
  - Statevector simulator
  - Initialize gate variant
  - qiskit-aer GPU variant if available
- [ ] Correctness verification between frameworks
- [ ] Integration tests comparing outputs

**Files to Create/Modify**:
- `benchmark/frameworks/mahout.py`
- `benchmark/frameworks/pennylane.py`
- `benchmark/frameworks/qiskit.py`
- `benchmark/tests/test_frameworks.py`

---

## Phase 4: Hyperparameter Sweep Infrastructure

### Issue 4.1: Create Hyperparameter Grid System
**Scope**: Systematic exploration of benchmark parameters

**Acceptance Criteria**:
- [ ] Create `benchmark/core/sweep.py`
- [ ] Implement `ParameterGrid` class:
  - Define parameter ranges (qubits, batch_size, samples, etc.)
  - Generate all combinations or random samples
  - Support linear, log-scale, and discrete ranges
- [ ] Implement `SweepRunner`:
  - Execute benchmark across parameter grid
  - Parallel execution option
  - Progress tracking with ETA
  - Checkpoint/resume capability
- [ ] Results stored in structured format (DataFrame)

**Files to Create/Modify**:
- `benchmark/core/sweep.py`
- `benchmark/tests/test_sweep.py`

**Example API**:
```python
from benchmark.core.sweep import ParameterGrid, SweepRunner

grid = ParameterGrid({
    "n_qubits": [8, 10, 12, 14, 16],
    "batch_size": [32, 64, 128, 256],
    "samples": [1000, 10000],
})

runner = SweepRunner(benchmark_fn, grid, parallel=True)
results = runner.run()  # Returns DataFrame with all results
```

---

### Issue 4.2: Create Benchmark Presets
**Scope**: Pre-defined hyperparameter configurations for common scenarios

**Acceptance Criteria**:
- [ ] Create `benchmark/presets/` directory
- [ ] Implement preset configurations:
  - `quick.yaml` - Fast sanity check (2-3 configs)
  - `standard.yaml` - Balanced coverage (10-20 configs)
  - `comprehensive.yaml` - Full exploration (50+ configs)
  - `scale.yaml` - Focus on large-scale performance
  - `accuracy.yaml` - Focus on encoding fidelity
- [ ] CLI support: `python run_benchmark.py --preset standard`
- [ ] Documentation of each preset's purpose

**Files to Create/Modify**:
- `benchmark/presets/quick.yaml`
- `benchmark/presets/standard.yaml`
- `benchmark/presets/comprehensive.yaml`
- `benchmark/presets/scale.yaml`
- `benchmark/presets/accuracy.yaml`

---

## Phase 5: Visualization & Reporting

### Issue 5.1: Create Results Data Model
**Scope**: Structured storage of benchmark results

**Acceptance Criteria**:
- [ ] Create `benchmark/results/model.py`
- [ ] Implement `BenchmarkResult` dataclass:
  - Timestamp, git commit hash, hardware info
  - Configuration used
  - Per-framework statistics
  - Raw timing data
- [ ] Implement `ResultsStore`:
  - Save to Parquet/JSON/CSV
  - Load historical results
  - Query/filter by parameters
  - Append new results to existing store
- [ ] Schema versioning for backwards compatibility

**Files to Create/Modify**:
- `benchmark/results/__init__.py`
- `benchmark/results/model.py`
- `benchmark/results/store.py`
- `benchmark/tests/test_results.py`

---

### Issue 5.2: Implement Publication-Quality Plotting
**Scope**: Matplotlib-based visualization for papers/presentations

**Acceptance Criteria**:
- [ ] Create `benchmark/visualization/plots.py`
- [ ] Implement plot types:
  - **Bar charts**: Framework comparison with error bars
  - **Line plots**: Performance vs. qubits/samples with confidence bands
  - **Heatmaps**: Parameter sweep results
  - **Box plots**: Distribution visualization
  - **Speedup charts**: Relative performance vs. baseline
- [ ] Publication styling:
  - Configurable figure size (single/double column)
  - Consistent color palette
  - LaTeX-compatible fonts
  - High DPI export (300+)
- [ ] Export formats: PNG, PDF, SVG

**Files to Create/Modify**:
- `benchmark/visualization/__init__.py`
- `benchmark/visualization/plots.py`
- `benchmark/visualization/style.py`
- `benchmark/tests/test_plots.py`

**Example API**:
```python
from benchmark.visualization import plot_framework_comparison, plot_scaling

fig = plot_framework_comparison(
    results,
    metric="latency_ms",
    error_bars="std",
    title="Encoding Latency Comparison",
    style="publication"
)
fig.savefig("latency_comparison.pdf", dpi=300)
```

---

### Issue 5.3: Create Interactive Dashboard
**Scope**: HTML/Jupyter dashboard for exploration

**Acceptance Criteria**:
- [ ] Create `benchmark/visualization/dashboard.py`
- [ ] Implement `BenchmarkDashboard` class:
  - Load results from store
  - Interactive filtering (framework, dataset, parameters)
  - Multiple linked visualizations
- [ ] Jupyter notebook integration:
  - Widget-based parameter selection
  - Live plot updates
- [ ] Static HTML export option
- [ ] Template notebook for analysis

**Files to Create/Modify**:
- `benchmark/visualization/dashboard.py`
- `benchmark/notebooks/analysis_template.ipynb`

**Dependencies**: `plotly` or `altair` for interactivity (optional)

---

### Issue 5.4: Implement Report Generation
**Scope**: Automated report generation for benchmarks

**Acceptance Criteria**:
- [ ] Create `benchmark/reports/generator.py`
- [ ] Implement `ReportGenerator`:
  - Markdown report with embedded plots
  - Summary statistics table
  - Configuration documentation
  - Hardware/software environment info
- [ ] Templates for:
  - Quick summary (1 page)
  - Full technical report
  - Blog post format
- [ ] CLI: `python generate_report.py results.parquet -o report.md`

**Files to Create/Modify**:
- `benchmark/reports/__init__.py`
- `benchmark/reports/generator.py`
- `benchmark/reports/templates/summary.md.j2`
- `benchmark/reports/templates/full.md.j2`

---

## Phase 6: Main Benchmark Runner

### Issue 6.1: Create Unified Benchmark CLI
**Scope**: Single entry point for all benchmarks

**Acceptance Criteria**:
- [ ] Create `benchmark/run_benchmark.py` as main entry point
- [ ] CLI arguments:
  - `--benchmark`: latency, throughput, e2e, all
  - `--dataset`: mnist, iris, synthetic, all
  - `--framework`: mahout, pennylane, qiskit, all
  - `--preset`: quick, standard, comprehensive
  - `--config`: path to YAML config
  - `--output`: results directory
  - `--plot`: generate plots after run
  - `--report`: generate report after run
- [ ] Progress reporting with ETA
- [ ] Graceful error handling and partial results
- [ ] Integration with all Phase 1-5 components

**Files to Create/Modify**:
- `benchmark/run_benchmark.py`
- `benchmark/cli.py`

**Example Usage**:
```bash
# Quick sanity check
python run_benchmark.py --preset quick --framework mahout

# Full comparison for publication
python run_benchmark.py --preset comprehensive --plot --report

# Specific configuration
python run_benchmark.py --benchmark latency --dataset mnist \
    --framework mahout,pennylane --config custom.yaml
```

---

### Issue 6.2: Create CI/CD Integration
**Scope**: Automated benchmark runs in CI

**Acceptance Criteria**:
- [ ] Create GitHub Actions workflow for benchmarks
- [ ] Nightly benchmark runs on GPU runner
- [ ] Performance regression detection:
  - Compare against baseline
  - Alert on >10% regression
- [ ] Results published to GitHub Pages or artifact storage
- [ ] Badge generation for README

**Files to Create/Modify**:
- `.github/workflows/benchmark.yml`
- `benchmark/ci/regression_check.py`

---

## Phase 7: Documentation & Examples

### Issue 7.1: Create Comprehensive Documentation
**Scope**: User guide and API documentation

**Acceptance Criteria**:
- [ ] Update `benchmark/README.md` with new architecture
- [ ] Create `benchmark/docs/user_guide.md`:
  - Getting started
  - Running benchmarks
  - Interpreting results
  - Adding new datasets/frameworks
- [ ] Create `benchmark/docs/api_reference.md`:
  - All public classes and functions
  - Examples for each module
- [ ] Create `benchmark/docs/contributing.md`:
  - How to add new benchmarks
  - Code style guidelines
  - Testing requirements

**Files to Create/Modify**:
- `benchmark/README.md`
- `benchmark/docs/user_guide.md`
- `benchmark/docs/api_reference.md`
- `benchmark/docs/contributing.md`

---

### Issue 7.2: Create Example Notebooks
**Scope**: Jupyter notebooks demonstrating benchmark usage

**Acceptance Criteria**:
- [ ] Create `benchmark/notebooks/01_quick_start.ipynb`:
  - Basic benchmark execution
  - Results interpretation
- [ ] Create `benchmark/notebooks/02_dataset_exploration.ipynb`:
  - Loading and visualizing datasets
  - Preparing data for encoding
- [ ] Create `benchmark/notebooks/03_framework_comparison.ipynb`:
  - Comparing Mahout vs PennyLane vs Qiskit
  - Statistical analysis
- [ ] Create `benchmark/notebooks/04_custom_benchmark.ipynb`:
  - Adding custom datasets
  - Creating custom visualizations
- [ ] All notebooks runnable in Google Colab

**Files to Create/Modify**:
- `benchmark/notebooks/01_quick_start.ipynb`
- `benchmark/notebooks/02_dataset_exploration.ipynb`
- `benchmark/notebooks/03_framework_comparison.ipynb`
- `benchmark/notebooks/04_custom_benchmark.ipynb`

---

## Implementation Priority

### Tier 1: Core Infrastructure (Start Here)
1. **Issue 1.1**: Statistical Measurement Framework
2. **Issue 1.2**: CUDA Synchronization & Cache Management
3. **Issue 2.1**: Dataset Abstraction Layer

### Tier 2: Data & Frameworks
4. **Issue 2.2**: Image Datasets (MNIST)
5. **Issue 2.3**: Tabular Datasets (Iris)
6. **Issue 3.1**: Framework Abstraction Layer
7. **Issue 3.2**: Framework Adapters

### Tier 3: Automation & Sweeps
8. **Issue 1.3**: Benchmark Configuration System
9. **Issue 4.1**: Hyperparameter Grid System
10. **Issue 4.2**: Benchmark Presets

### Tier 4: Visualization
11. **Issue 5.1**: Results Data Model
12. **Issue 5.2**: Publication-Quality Plotting
13. **Issue 5.4**: Report Generation

### Tier 5: Integration & Polish
14. **Issue 6.1**: Unified Benchmark CLI
15. **Issue 2.4**: Scale Test Datasets
16. **Issue 5.3**: Interactive Dashboard
17. **Issue 6.2**: CI/CD Integration
18. **Issue 7.1**: Comprehensive Documentation
19. **Issue 7.2**: Example Notebooks

---

## Dataset Matrix Summary

| Category | Dataset | Class | Priority |
|----------|---------|-------|----------|
| Image | MNIST (0 vs 1, 4×4) | `MNISTBinaryDataset` | High |
| Image | Synthetic images | `SyntheticImageDataset` | High |
| Tabular | Iris (binary, 2 features) | `IrisBinaryDataset` | High |
| Tabular | Synthetic blobs | `SyntheticBlobsDataset` | High |
| Tabular | Generic CSV | `CSVDataset` | Medium |
| Scale | 100k synthetic samples | `LargeSyntheticDataset` | Medium |
| Scale | Large image batches | `LargeImageDataset` | Low |

---

## Success Metrics

1. **Statistical Rigor**: All measurements include mean ± std, with p95/p99 for latency
2. **Reproducibility**: Same config produces <5% variance across runs
3. **Framework Coverage**: Mahout, PennyLane, Qiskit all benchmarked
4. **Dataset Coverage**: At least 5 datasets (2 image, 2 tabular, 1 scale)
5. **Visualization Quality**: Plots ready for academic papers (PDF, 300 DPI)
6. **Documentation**: All public APIs documented with examples
7. **CI Integration**: Automated nightly benchmarks with regression detection
