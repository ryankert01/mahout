# NumPy File Size Check Implementation

## Summary
Added file size validation to the NumPy reader to prevent Out-Of-Memory (OOM) errors when loading large .npy files.

## Problem
The NumPy reader (`NumpyReader`) loads entire .npy files into memory without checking file size first. This can cause OOM errors if users attempt to load very large files (e.g., multi-GB quantum state files).

## Solution
Added a file size check in `NumpyReader::new()` that:
1. Reads file metadata using `std::fs::metadata()`
2. Compares file size against `MAX_NUMPY_FILE_SIZE` constant (2GB)
3. Returns a descriptive error if file exceeds the limit
4. Suggests using Parquet or Arrow IPC formats for larger datasets

## Changes Made

### File Modified
- `qdp/qdp-core/src/readers/numpy.rs`

### Code Changes
```rust
// Added constant
const MAX_NUMPY_FILE_SIZE: u64 = 2 * 1024 * 1024 * 1024; // 2GB

// Added file size check in NumpyReader::new()
let metadata = fs::metadata(path)?;
let file_size = metadata.len();
if file_size > MAX_NUMPY_FILE_SIZE {
    return Err(MahoutError::InvalidInput(...));
}
```

### Test Added
```rust
#[test]
fn test_numpy_reader_file_size_limit() {
    // Verifies size check doesn't reject valid small files
    // Documents the size limit behavior
}
```

## Testing Results
All tests pass (7/7):
- `test_numpy_reader_basic` ✓
- `test_numpy_reader_fortran_order` ✓
- `test_numpy_reader_file_not_found` ✓
- `test_numpy_reader_invalid_dimensions` ✓
- `test_numpy_reader_already_consumed` ✓
- `test_numpy_reader_empty_dimensions` ✓
- `test_numpy_reader_file_size_limit` ✓ (NEW)

## Comparison with Other Formats

| Format | Size Handling | Streaming Support |
|--------|---------------|-------------------|
| NumPy | **Now**: 2GB limit check | No (loads all into memory) |
| Parquet | Metadata-based | Yes (batch/streaming) |
| Arrow IPC | Batch iteration | Yes (batch processing) |

## Why 2GB Limit?
1. NumPy files must be fully loaded into memory (no streaming)
2. 2GB is a reasonable limit for most systems
3. Prevents accidental OOM from very large files
4. Consistent with typical file size limits in similar tools
5. Large quantum datasets should use streaming formats anyway

## Error Message Example
```
NumPy file is too large: 3221225472 bytes (max: 2147483648 bytes / 2 GB).
For larger datasets, consider using Parquet or Arrow IPC formats which support streaming.
```

## Branch Status
- **Implementation completed on**: `dev-qdp` branch (commit `8954fb7db`)
- **Also available on**: `numpy-io` branch (commit `958a39b36`)
- **Reference copy on**: `copilot/add-file-size-check` branch (for PR)

The implementation is production-ready on `dev-qdp` where all dependencies exist.

## Future Considerations
- Could make the limit configurable via environment variable
- Could add a `--force` flag to bypass the check if needed
- Could implement streaming NumPy reader (more complex)
