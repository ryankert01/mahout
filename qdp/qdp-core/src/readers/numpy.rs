//
// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements.  See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License.  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! NumPy format reader implementation (placeholder).
//!
//! This is a placeholder implementation showing how to add new input formats.
//! To fully implement:
//! 1. Add `ndarray-npy` or similar crate to dependencies
//! 2. Implement proper .npy file parsing
//! 3. Add comprehensive tests

use std::path::Path;

use crate::error::{MahoutError, Result};
use crate::reader::DataReader;

/// Reader for NumPy `.npy` files containing 2D float64 arrays.
///
/// # Expected Format
/// - 2D array with shape `[num_samples, sample_size]`
/// - Data type: `float64`
/// - Fortran (column-major) or C (row-major) order supported
///
/// # Example
///
/// ```rust,ignore
/// use qdp_core::reader::DataReader;
/// use qdp_core::readers::NumpyReader;
///
/// let mut reader = NumpyReader::new("data.npy").unwrap();
/// let (data, num_samples, sample_size) = reader.read_batch().unwrap();
/// println!("Read {} samples of size {}", num_samples, sample_size);
/// ```
pub struct NumpyReader {
    #[allow(dead_code)]
    path: std::path::PathBuf,
}

impl NumpyReader {
    /// Create a new NumPy reader.
    ///
    /// # Arguments
    /// * `path` - Path to the `.npy` file
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(Self {
            path: path.as_ref().to_path_buf(),
        })
    }
}

impl DataReader for NumpyReader {
    fn read_batch(&mut self) -> Result<(Vec<f64>, usize, usize)> {
        // TODO: Implement actual NumPy file reading
        // Suggested implementation:
        //
        // 1. Use ndarray-npy crate to read the file:
        //    let array: Array2<f64> = ndarray_npy::read_npy(&self.path)?;
        //
        // 2. Extract shape:
        //    let (num_samples, sample_size) = array.dim();
        //
        // 3. Flatten to Vec<f64>:
        //    let data = if array.is_standard_layout() {
        //        array.into_raw_vec()
        //    } else {
        //        array.iter().copied().collect()
        //    };
        //
        // 4. Return:
        //    Ok((data, num_samples, sample_size))

        Err(MahoutError::NotImplemented(
            "NumPy reader not yet implemented. See docs/ADDING_INPUT_FORMATS.md".to_string()
        ))
    }

    fn get_sample_size(&self) -> Option<usize> {
        // Could be determined by reading just the header
        None
    }

    fn get_num_samples(&self) -> Option<usize> {
        // Could be determined by reading just the header
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "not yet implemented")]
    fn test_numpy_reader_placeholder() {
        // This test documents that the feature is not yet implemented
        let mut reader = NumpyReader::new("test.npy").unwrap();
        let _ = reader.read_batch().unwrap();
    }
}
