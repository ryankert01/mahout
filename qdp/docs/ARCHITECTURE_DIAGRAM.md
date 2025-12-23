# QDP Reader Architecture Diagram

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         QDP Engine                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  encode_from_parquet(), encode_from_arrow_ipc()          │  │
│  │  encode_batch()                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Reader Trait Layer                              │
│  ┌──────────────────┐         ┌──────────────────────┐         │
│  │   DataReader     │         │ StreamingDataReader  │         │
│  │  Trait (basic)   │◄────────│   Trait (advanced)   │         │
│  └──────────────────┘         └──────────────────────┘         │
└──────┬────────────────────────────┬─────────────────────────────┘
       │                            │
       ▼                            ▼
┌──────────────────────────────────────────────────────────────────┐
│              Format-Specific Implementations                      │
│                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ ParquetReader   │  │ ArrowIPCReader  │  │  NumpyReader    │ │
│  │  (batch mode)   │  │                 │  │  (placeholder)  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐                      │
│  │   Parquet       │  │  TorchReader    │                      │
│  │StreamingReader  │  │  (placeholder)  │                      │
│  └─────────────────┘  └─────────────────┘                      │
└──────┬───────────────────┬──────────────────┬───────────────────┘
       │                   │                  │
       ▼                   ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   .parquet   │  │    .arrow    │  │     .npy     │
│    files     │  │   .feather   │  │     .pt      │
└──────────────┘  └──────────────┘  └──────────────┘
```

## Trait Hierarchy

```
DataReader (trait)
    ├── read_batch() -> (Vec<f64>, usize, usize)
    ├── get_sample_size() -> Option<usize>
    └── get_num_samples() -> Option<usize>

StreamingDataReader (trait): DataReader
    ├── read_chunk(&mut [f64]) -> usize
    └── total_rows() -> usize
```

## Implementation Matrix

```
┌──────────────────────┬──────────────┬────────────────┬──────────┐
│ Format               │ DataReader   │ Streaming      │ Status   │
├──────────────────────┼──────────────┼────────────────┼──────────┤
│ Parquet              │ ✓            │ ✓              │ Complete │
│ Arrow IPC            │ ✓            │ -              │ Complete │
│ NumPy (.npy)         │ Placeholder  │ -              │ Future   │
│ PyTorch (.pt)        │ Placeholder  │ -              │ Future   │
│ HDF5 (.h5)           │ -            │ -              │ Future   │
│ JSON                 │ -            │ -              │ Future   │
│ CSV                  │ -            │ -              │ Future   │
└──────────────────────┴──────────────┴────────────────┴──────────┘
```

## Data Flow

### Batch Reading (Small Files)

```
File (.arrow/.parquet)
    │
    ▼
ArrowIPCReader::new() / ParquetReader::new()
    │
    ▼
reader.read_batch()
    │
    ▼
(data: Vec<f64>, num_samples: usize, sample_size: usize)
    │
    ▼
QdpEngine::encode_batch()
    │
    ▼
DLPack Tensor (GPU)
```

### Streaming Reading (Large Files)

```
Large File (.parquet)
    │
    ▼
ParquetStreamingReader::new()
    │
    ▼
loop {
    reader.read_chunk(&mut buffer)  ← Constant memory
        │
        ▼
    Process chunk
}
    │
    ▼
Complete
```

## Extensibility Pattern

### Adding a New Format

```
1. Create Reader
   ┌──────────────────────────────────┐
   │ src/readers/myformat.rs          │
   │                                  │
   │ pub struct MyFormatReader { }    │
   │                                  │
   │ impl DataReader for ... {        │
   │     fn read_batch() { ... }      │
   │ }                                │
   └──────────────────────────────────┘
                 │
                 ▼
2. Register
   ┌──────────────────────────────────┐
   │ src/readers/mod.rs               │
   │                                  │
   │ pub mod myformat;                │
   │ pub use myformat::MyFormatReader;│
   └──────────────────────────────────┘
                 │
                 ▼
3. Use
   ┌──────────────────────────────────┐
   │ User Code                        │
   │                                  │
   │ let r = MyFormatReader::new()?;  │
   │ let data = r.read_batch()?;      │
   └──────────────────────────────────┘
```

## Performance Characteristics

```
┌────────────────┬──────────────┬──────────────┬─────────────┐
│ Operation      │ Time         │ Memory       │ Notes       │
├────────────────┼──────────────┼──────────────┼─────────────┤
│ Parquet Batch  │ O(n)         │ O(n)         │ Fast        │
│ Parquet Stream │ O(n)         │ O(1)         │ Const mem   │
│ Arrow IPC      │ O(n)         │ O(n)         │ Zero-copy   │
│ Trait overhead │ O(1)         │ O(1)         │ Static      │
└────────────────┴──────────────┴──────────────┴─────────────┘

n = file size
Trait calls use static dispatch (no vtable overhead)
```

## API Layers

```
High Level (User-Friendly)
    ├── QdpEngine::encode_from_parquet()
    └── QdpEngine::encode_from_arrow_ipc()
            │
            ▼
Mid Level (Convenience)
    ├── read_parquet_batch()
    └── read_arrow_ipc_batch()
            │
            ▼
Low Level (Direct)
    ├── ParquetReader::new().read_batch()
    ├── ParquetStreamingReader::new().read_chunk()
    └── ArrowIPCReader::new().read_batch()
```

## Backward Compatibility

```
Old Code:                         New Implementation:
ParquetBlockReader::new()  ───►   ParquetStreamingReader::new()
      │                                  │
      │ (type alias)                     │
      └──────────────────────────────────┘

read_parquet_batch()       ───►   ParquetReader::new().read_batch()
      │                                  │
      │ (wrapper)                        │
      └──────────────────────────────────┘

All existing code continues to work!
```
