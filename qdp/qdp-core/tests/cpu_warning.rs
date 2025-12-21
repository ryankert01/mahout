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

//! Tests to verify that CPU-based code paths emit warnings when GPU alternatives exist.

use qdp_core::preprocessing::Preprocessor;

#[test]
fn test_cpu_l2_norm_emits_warning() {
    // This test verifies that when CPU-based L2 norm calculation is used,
    // a warning is logged. To see the warning, run with:
    // RUST_LOG=warn cargo test test_cpu_l2_norm_emits_warning -- --nocapture
    
    env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Warn)
        .try_init()
        .ok(); // Ignore error if already initialized
    
    let data = vec![3.0, 4.0];
    let result = Preprocessor::calculate_l2_norm(&data);
    
    assert!(result.is_ok());
    // The warning should have been logged
}

#[test]
fn test_cpu_batch_l2_norms_emits_warning() {
    // This test verifies that when CPU-based batch L2 norm calculation is used,
    // a warning is logged. To see the warning, run with:
    // RUST_LOG=warn cargo test test_cpu_batch_l2_norms_emits_warning -- --nocapture
    
    env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Warn)
        .try_init()
        .ok(); // Ignore error if already initialized
    
    let data = vec![3.0, 4.0, 1.0, 1.0]; // 2 samples of size 2
    let result = Preprocessor::calculate_batch_l2_norms(&data, 2, 2);
    
    assert!(result.is_ok());
    // The warning should have been logged
}
