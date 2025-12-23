# Adding New Input Format Support

This document explains how to add support for new input formats to the QDP library using the refactored reader architecture.

## Overview

The QDP library uses a trait-based architecture for reading quantum data from various sources. This makes it easy to add new input formats without modifying the core library code.

## Architecture

The reader system is based on two main traits:

- **`DataReader`**: Basic interface for batch reading (read all data at once)
- **`StreamingDataReader`**: Extended interface for chunk-by-chunk streaming (for large files)

## Adding a New Format

### Step 1: Implement the `DataReader` Trait

Create a new file in `qdp-core/src/readers/` for your format. For example, to add NumPy support:

```rust
// qdp-core/src/readers/numpy.rs

use std::path::Path;
use crate::error::{MahoutError, Result};
use crate::reader::DataReader;

pub struct NumpyReader {
    path: std::path::PathBuf,
    read: bool,
}

impl NumpyReader {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(Self {
            path: path.as_ref().to_path_buf(),
            read: false,
        })
    }
}

impl DataReader for NumpyReader {
    fn read_batch(&mut self) -> Result<(Vec<f64>, usize, usize)> {
        if self.read {
            return Err(MahoutError::InvalidInput("Reader already consumed".to_string()));
        }
        self.read = true;

        // TODO: Implement NumPy file reading logic
        // 1. Open and parse .npy file
        // 2. Extract shape information (num_samples, sample_size)
        // 3. Read data as Vec<f64>
        // 4. Return (flattened_data, num_samples, sample_size)
        
        unimplemented!("NumPy reading not yet implemented")
    }

    fn get_sample_size(&self) -> Option<usize> {
        // Return sample size if known before reading
        None
    }

    fn get_num_samples(&self) -> Option<usize> {
        // Return number of samples if known before reading
        None
    }
}
```

### Step 2: (Optional) Implement `StreamingDataReader` for Large Files

If your format needs to support streaming for large files:

```rust
use crate::reader::StreamingDataReader;

impl StreamingDataReader for NumpyReader {
    fn read_chunk(&mut self, buffer: &mut [f64]) -> Result<usize> {
        // Implement chunk-by-chunk reading
        // Return number of elements written to buffer
        // Return 0 when no more data
        unimplemented!("Streaming not yet implemented")
    }

    fn total_rows(&self) -> usize {
        // Return total number of samples
        0
    }
}
```

### Step 3: Register Your Reader

Add your reader to `qdp-core/src/readers/mod.rs`:

```rust
pub mod parquet;
pub mod arrow_ipc;
pub mod numpy;  // Add this line

pub use parquet::{ParquetReader, ParquetStreamingReader};
pub use arrow_ipc::ArrowIPCReader;
pub use numpy::NumpyReader;  // Add this line
```

### Step 4: Add Dependencies (if needed)

If your format requires external crates, add them to `qdp-core/Cargo.toml`:

```toml
[dependencies]
# ... existing dependencies ...
ndarray-npy = "0.8"  # Example for NumPy support
```

### Step 5: Add Tests

Create tests for your new reader in `qdp-core/tests/numpy_io.rs`:

```rust
use qdp_core::reader::DataReader;
use qdp_core::readers::NumpyReader;

#[test]
fn test_read_numpy_batch() {
    // Create test .npy file
    // ...
    
    let mut reader = NumpyReader::new("test.npy").unwrap();
    let (data, num_samples, sample_size) = reader.read_batch().unwrap();
    
    assert_eq!(num_samples, 10);
    assert_eq!(sample_size, 16);
    assert_eq!(data.len(), num_samples * sample_size);
}
```

### Step 6: Add Convenience Functions (Optional)

You can add convenience functions to `qdp-core/src/io.rs` for backward compatibility or ease of use:

```rust
pub fn read_numpy_batch<P: AsRef<Path>>(path: P) -> Result<(Vec<f64>, usize, usize)> {
    use crate::reader::DataReader;
    let mut reader = crate::readers::NumpyReader::new(path)?;
    reader.read_batch()
}
```

### Step 7: Add Integration with QdpEngine (Optional)

Add a high-level API method to `QdpEngine` in `qdp-core/src/lib.rs`:

```rust
impl QdpEngine {
    // ... existing methods ...
    
    pub fn encode_from_numpy(
        &self,
        path: &str,
        num_qubits: usize,
        encoding_method: &str,
    ) -> Result<*mut DLManagedTensor> {
        use crate::reader::DataReader;
        let mut reader = crate::readers::NumpyReader::new(path)?;
        let (batch_data, num_samples, sample_size) = reader.read_batch()?;
        self.encode_batch(&batch_data, num_samples, sample_size, num_qubits, encoding_method)
    }
}
```

## Examples of Supported Formats

### Parquet (Implemented)
- **Reader**: `ParquetReader`, `ParquetStreamingReader`
- **Format**: List<Float64> or FixedSizeList<Float64> columns
- **Use case**: Efficient columnar storage, large datasets

### Arrow IPC (Implemented)
- **Reader**: `ArrowIPCReader`  
- **Format**: FixedSizeList<Float64> or List<Float64> columns
- **Use case**: Fast zero-copy data exchange, interoperability

### NumPy (Placeholder)
- **Reader**: `NumpyReader` (to be implemented)
- **Format**: `.npy` files with 2D arrays
- **Use case**: Python ecosystem integration

### PyTorch (Placeholder)
- **Reader**: `TorchReader` (to be implemented)
- **Format**: `.pt` tensor files
- **Use case**: Deep learning workflows

## Performance Considerations

1. **Zero-copy**: Use `extend_from_slice()` instead of `extend()` when possible
2. **Pre-allocation**: Reserve capacity when you know the total size
3. **Streaming**: Implement `StreamingDataReader` for files > 1GB
4. **Buffering**: Use appropriate buffer sizes (typically 1-4MB chunks)

## Best Practices

1. **Error handling**: Use descriptive error messages with context
2. **Validation**: Check data types and shapes early
3. **Documentation**: Document expected format and limitations
4. **Testing**: Include tests for edge cases (empty files, inconsistent shapes, etc.)
5. **Backward compatibility**: Maintain existing APIs when possible

## Questions?

See existing implementations in:
- `qdp-core/src/readers/parquet.rs`
- `qdp-core/src/readers/arrow_ipc.rs`
