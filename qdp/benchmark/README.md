# QDP Benchmarks

This directory contains performance benchmarks comparing Mahout QDP against other quantum computing frameworks (PennyLane, Qiskit).

## Benchmarks

### 1. DataLoader Throughput (`benchmark_dataloader_throughput.py`)

Compares full pipeline throughput across frameworks when loading batches of quantum state vectors from memory.

**Run:**
```bash
python benchmark_dataloader_throughput.py --qubits 16 --batches 200 --batch-size 64
```

**Options:**
- `--qubits`: Number of qubits (default: 16)
- `--batches`: Number of batches to process (default: 200)
- `--batch-size`: Vectors per batch (default: 64)
- `--prefetch`: CPU-side prefetch depth (default: 16)
- `--frameworks`: Comma-separated list: pennylane,qiskit,mahout or 'all' (default: all)

### 2. NumPy I/O Benchmark (`benchmark_numpy_io.py`) ⭐ NEW

Compares performance of loading quantum state data from NumPy .npy files and encoding on GPU.

**Run:**
```bash
python benchmark_numpy_io.py --qubits 10 --samples 1000
```

**Options:**
- `--qubits`: Number of qubits (default: 10)
- `--samples`: Number of samples to generate (default: 1000)
- `--output`: Path to save .npy file (default: temp file)
- `--frameworks`: Comma-separated list: mahout,pennylane or 'all' (default: all)

**What it measures:**
- File I/O: Loading .npy file from disk
- Encoding: Converting to quantum states on GPU
- Total throughput: Samples processed per second
- Average time per sample

**Example output:**
```
SUMMARY
Framework       Time (s)     Throughput           Avg/Sample     
----------------------------------------------------------------------
Mahout          2.3456       426.4                5.50           
PennyLane       15.2341      65.6                 15.23          

SPEEDUP COMPARISON
----------------------------------------------------------------------
Mahout vs PennyLane: 6.50x
Time reduction: 6.50x faster
```

### 3. End-to-End Test (`benchmark_e2e.py`)

Complete workflow test including data generation, encoding, and circuit execution.

**Run:**
```bash
python benchmark_e2e.py
```

## Setup

Install required dependencies:

```bash
# From qdp/benchmark directory
pip install -r requirements.txt

# Or using uv
uv pip install -r requirements.txt
```

**Required:**
- numpy
- torch (with CUDA support)
- mahout_qdp (installed via `maturin develop`)

**Optional (for comparisons):**
- pennylane
- qiskit
- qiskit-aer

## Running Without Optional Frameworks

If you only want to benchmark Mahout without installing PennyLane or Qiskit:

```bash
# Uninstall optional frameworks
pip uninstall pennylane qiskit qiskit-aer

# Run with only Mahout
python benchmark_numpy_io.py --frameworks mahout
python benchmark_dataloader_throughput.py --frameworks mahout
```

## Performance Tips

1. **GPU Warmup**: First run may be slower due to GPU initialization
2. **File I/O**: For NumPy benchmark, SSD vs HDD affects I/O times
3. **Memory**: Large sample counts may require more RAM
4. **Batch Size**: Larger batches may improve throughput but increase memory usage

## Interpreting Results

**Throughput (samples/sec)**: Higher is better
- Measures how many quantum state vectors can be processed per second
- Includes both I/O and encoding time

**Average per sample (ms)**: Lower is better
- Time to load and encode one quantum state vector
- Useful for understanding latency

**Speedup**: Ratio comparing frameworks
- Values > 1.0 mean Mahout is faster
- Example: 6.5x means Mahout is 6.5 times faster

## Common Issues

**CUDA errors:**
- Ensure GPU is available: `nvidia-smi`
- Check CUDA environment: `echo $CUDA_VISIBLE_DEVICES`
- Specify GPU: `CUDA_VISIBLE_DEVICES=0 python benchmark_numpy_io.py`

**Out of memory:**
- Reduce `--samples` or `--qubits`
- Monitor with `nvidia-smi`

**Import errors:**
- Install mahout_qdp: `cd ../qdp-python && uv run maturin develop`
- Check Python environment: `which python`

## Adding New Benchmarks

To add a new benchmark:

1. Create `benchmark_<name>.py`
2. Follow the structure of existing benchmarks
3. Include clear documentation in docstring
4. Add command-line arguments for flexibility
5. Print summary table at the end
6. Update this README

## License

Apache License 2.0 - See repository LICENSE file
