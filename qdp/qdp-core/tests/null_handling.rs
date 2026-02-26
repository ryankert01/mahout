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

//! Tests for unified null value handling between batch and streaming modes (MAHOUT-765).

use arrow::array::{FixedSizeListArray, Float64Array, ListBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::FileWriter as ArrowFileWriter;
use std::fs;
use std::sync::Arc;

use qdp_core::io::{arrow_to_vec_with_null_handling, arrow_to_vec_chunked_with_null_handling};
use qdp_core::reader::{DataReader, NullHandling, handle_float64_nulls};
use qdp_core::readers::{ArrowIPCReader, ParquetReader};

// ---------------------------------------------------------------------------
// Unit tests for handle_float64_nulls helper
// ---------------------------------------------------------------------------

#[test]
fn test_handle_nulls_no_nulls() {
    let array = Float64Array::from(vec![1.0, 2.0, 3.0]);
    let mut output = Vec::new();

    // Both strategies produce the same result when there are no nulls
    handle_float64_nulls(&mut output, &array, NullHandling::FillZero).unwrap();
    assert_eq!(output, vec![1.0, 2.0, 3.0]);

    output.clear();
    handle_float64_nulls(&mut output, &array, NullHandling::Reject).unwrap();
    assert_eq!(output, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_handle_nulls_fill_zero() {
    let array = Float64Array::from(vec![Some(1.0), None, Some(3.0), None]);
    let mut output = Vec::new();

    handle_float64_nulls(&mut output, &array, NullHandling::FillZero).unwrap();
    assert_eq!(output, vec![1.0, 0.0, 3.0, 0.0]);
}

#[test]
fn test_handle_nulls_reject() {
    let array = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
    let mut output = Vec::new();

    let result = handle_float64_nulls(&mut output, &array, NullHandling::Reject);
    assert!(result.is_err());

    match result {
        Err(qdp_core::MahoutError::InvalidInput(msg)) => {
            assert!(msg.contains("Null value encountered"));
        }
        other => panic!("Expected InvalidInput error, got: {:?}", other),
    }
}

#[test]
fn test_null_handling_default_is_fill_zero() {
    assert_eq!(NullHandling::default(), NullHandling::FillZero);
}

// ---------------------------------------------------------------------------
// Tests for io.rs utility functions with null handling
// ---------------------------------------------------------------------------

#[test]
fn test_arrow_to_vec_with_null_handling_fill_zero() {
    let array = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
    let result = arrow_to_vec_with_null_handling(&array, NullHandling::FillZero).unwrap();
    assert_eq!(result, vec![1.0, 0.0, 3.0]);
}

#[test]
fn test_arrow_to_vec_with_null_handling_reject() {
    let array = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
    let result = arrow_to_vec_with_null_handling(&array, NullHandling::Reject);
    assert!(result.is_err());
}

#[test]
fn test_arrow_to_vec_chunked_with_null_handling_fill_zero() {
    let a1 = Float64Array::from(vec![Some(1.0), None]);
    let a2 = Float64Array::from(vec![Some(3.0), None]);
    let result =
        arrow_to_vec_chunked_with_null_handling(&[a1, a2], NullHandling::FillZero).unwrap();
    assert_eq!(result, vec![1.0, 0.0, 3.0, 0.0]);
}

#[test]
fn test_arrow_to_vec_chunked_with_null_handling_reject() {
    let a1 = Float64Array::from(vec![1.0, 2.0]);
    let a2 = Float64Array::from(vec![Some(3.0), None]); // second chunk has nulls
    let result = arrow_to_vec_chunked_with_null_handling(&[a1, a2], NullHandling::Reject);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Helper: write a Parquet file with List<Float64> that contains null elements
// ---------------------------------------------------------------------------

fn write_parquet_list_with_nulls(path: &str, sample_size: usize) {
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use std::fs::File;

    let num_samples = 3;

    let mut list_builder =
        ListBuilder::new(Float64Array::builder(num_samples * sample_size));

    // Sample 0: all valid
    for j in 0..sample_size {
        list_builder.values().append_value(j as f64);
    }
    list_builder.append(true);

    // Sample 1: contains a null at position 0
    list_builder.values().append_null();
    for j in 1..sample_size {
        list_builder.values().append_value(j as f64 + 100.0);
    }
    list_builder.append(true);

    // Sample 2: all valid
    for j in 0..sample_size {
        list_builder.values().append_value(j as f64 + 200.0);
    }
    list_builder.append(true);

    let list_array = list_builder.finish();

    let schema = Arc::new(Schema::new(vec![Field::new(
        "data",
        DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
        false,
    )]));

    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(list_array)]).unwrap();

    let file = File::create(path).unwrap();
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

// ---------------------------------------------------------------------------
// Helper: write an Arrow IPC file with FixedSizeList<Float64> containing nulls
// ---------------------------------------------------------------------------

fn write_arrow_ipc_fixed_with_nulls(path: &str, sample_size: usize) {
    use arrow::record_batch::RecordBatch;
    use std::fs::File;

    let num_samples: usize = 3;

    // Build flat values with a null in sample 1, position 0
    let mut values: Vec<Option<f64>> = Vec::with_capacity(num_samples * sample_size);
    // Sample 0: all valid
    for j in 0..sample_size {
        values.push(Some(j as f64));
    }
    // Sample 1: null at position 0
    values.push(None);
    for j in 1..sample_size {
        values.push(Some(j as f64 + 100.0));
    }
    // Sample 2: all valid
    for j in 0..sample_size {
        values.push(Some(j as f64 + 200.0));
    }

    let values_array = Float64Array::from(values);
    let field = Arc::new(Field::new("item", DataType::Float64, true));
    let list_array =
        FixedSizeListArray::new(field, sample_size as i32, Arc::new(values_array), None);

    let schema = Arc::new(Schema::new(vec![Field::new(
        "data",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float64, true)),
            sample_size as i32,
        ),
        false,
    )]));

    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(list_array)]).unwrap();

    let file = File::create(path).unwrap();
    let mut writer = ArrowFileWriter::try_new(file, &schema).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();
}

// ---------------------------------------------------------------------------
// ParquetReader null handling tests (batch mode)
// ---------------------------------------------------------------------------

#[test]
fn test_parquet_reader_fill_zero_on_nulls() {
    let path = "/tmp/test_null_handling_parquet_fill.parquet";
    let sample_size = 4;
    write_parquet_list_with_nulls(path, sample_size);

    let mut reader = ParquetReader::new(path, None).unwrap();
    // Default is FillZero
    let (data, num_samples, ss) = reader.read_batch().unwrap();

    assert_eq!(num_samples, 3);
    assert_eq!(ss, sample_size);

    // Sample 1 had a null at position 0 -> should be 0.0
    let sample1_start = sample_size;
    assert_eq!(data[sample1_start], 0.0);

    fs::remove_file(path).unwrap();
}

#[test]
fn test_parquet_reader_reject_on_nulls() {
    let path = "/tmp/test_null_handling_parquet_reject.parquet";
    let sample_size = 4;
    write_parquet_list_with_nulls(path, sample_size);

    let mut reader = ParquetReader::new(path, None)
        .unwrap()
        .with_null_handling(NullHandling::Reject);
    let result = reader.read_batch();

    assert!(result.is_err(), "Reject should error on nulls");

    fs::remove_file(path).unwrap();
}

// ---------------------------------------------------------------------------
// ParquetStreamingReader null handling tests (streaming mode)
// ---------------------------------------------------------------------------

#[test]
fn test_parquet_streaming_reader_fill_zero_on_nulls() {
    use qdp_core::readers::ParquetStreamingReader;

    let path = "/tmp/test_null_handling_streaming_fill.parquet";
    let sample_size = 4;
    write_parquet_list_with_nulls(path, sample_size);

    let mut reader = ParquetStreamingReader::new(path, Some(2)).unwrap();
    // Default is FillZero -> should NOT error
    let (data, num_samples, ss) = reader.read_batch().unwrap();

    assert_eq!(num_samples, 3);
    assert_eq!(ss, sample_size);

    // Sample 1 null at position 0 -> 0.0
    let sample1_start = sample_size;
    assert_eq!(data[sample1_start], 0.0);

    fs::remove_file(path).unwrap();
}

#[test]
fn test_parquet_streaming_reader_reject_on_nulls() {
    use qdp_core::readers::ParquetStreamingReader;

    let path = "/tmp/test_null_handling_streaming_reject.parquet";
    let sample_size = 4;
    write_parquet_list_with_nulls(path, sample_size);

    let mut reader = ParquetStreamingReader::new(path, Some(2))
        .unwrap()
        .with_null_handling(NullHandling::Reject);
    let result = reader.read_batch();

    assert!(result.is_err(), "Reject should error on nulls in streaming mode");

    fs::remove_file(path).unwrap();
}

// ---------------------------------------------------------------------------
// ArrowIPCReader null handling tests
// ---------------------------------------------------------------------------

#[test]
fn test_arrow_ipc_reader_fill_zero_on_nulls() {
    let path = "/tmp/test_null_handling_arrow_fill.arrow";
    let sample_size = 4;
    write_arrow_ipc_fixed_with_nulls(path, sample_size);

    let mut reader = ArrowIPCReader::new(path).unwrap();
    // Default is FillZero
    let (data, num_samples, ss) = reader.read_batch().unwrap();

    assert_eq!(num_samples, 3);
    assert_eq!(ss, sample_size);

    // Sample 1, position 0 was null -> 0.0
    let sample1_start = sample_size;
    assert_eq!(data[sample1_start], 0.0);

    fs::remove_file(path).unwrap();
}

#[test]
fn test_arrow_ipc_reader_reject_on_nulls() {
    let path = "/tmp/test_null_handling_arrow_reject.arrow";
    let sample_size = 4;
    write_arrow_ipc_fixed_with_nulls(path, sample_size);

    let mut reader = ArrowIPCReader::new(path)
        .unwrap()
        .with_null_handling(NullHandling::Reject);
    let result = reader.read_batch();

    assert!(result.is_err(), "Reject should error on nulls");

    fs::remove_file(path).unwrap();
}

// ---------------------------------------------------------------------------
// Consistency test: batch and streaming produce identical results
// ---------------------------------------------------------------------------

#[test]
fn test_batch_and_streaming_produce_same_output_with_nulls() {
    use qdp_core::readers::ParquetStreamingReader;

    let path = "/tmp/test_null_handling_consistency.parquet";
    let sample_size = 4;
    write_parquet_list_with_nulls(path, sample_size);

    // Batch mode with FillZero
    let mut batch_reader = ParquetReader::new(path, None).unwrap();
    let (batch_data, batch_n, batch_ss) = batch_reader.read_batch().unwrap();

    // Streaming mode with FillZero (default)
    let mut streaming_reader = ParquetStreamingReader::new(path, Some(2)).unwrap();
    let (stream_data, stream_n, stream_ss) = streaming_reader.read_batch().unwrap();

    assert_eq!(batch_n, stream_n, "Sample count must match");
    assert_eq!(batch_ss, stream_ss, "Sample size must match");
    assert_eq!(batch_data.len(), stream_data.len(), "Data length must match");
    assert_eq!(batch_data, stream_data, "Data values must be identical between batch and streaming modes");

    fs::remove_file(path).unwrap();
}
