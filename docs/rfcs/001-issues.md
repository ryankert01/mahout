# Issues for RFC-001: Python Infrastructure Consolidation

Parent tracking issue and 5 implementation issues for the Python infrastructure consolidation effort.

---

## Issue #1: [Tracking] Python Infrastructure Consolidation

**Labels:** `enhancement`, `infrastructure`, `tracking`

### Description

This is the tracking issue for consolidating the Python project infrastructure to provide a unified developer experience.

### Motivation

Currently, the project has fragmented test infrastructure:
- Tests split between `testing/` and `qdp/qdp-python/tests/`
- Duplicate pytest configurations with different versions
- Complex multi-step development workflow
- Inconsistent dev dependencies

### Goals

- [ ] Single `pytest` command runs all tests from repository root
- [ ] Unified pytest configuration with consistent markers
- [ ] Consolidated dev dependencies in root `pyproject.toml`
- [ ] Auto-skip QDP tests when extension isn't built
- [ ] Updated documentation

### Implementation Issues

- [ ] #X: Add root conftest.py with QDP auto-skip logic
- [ ] #X: Consolidate dev dependencies in root pyproject.toml
- [ ] #X: Reorganize test directory structure
- [ ] #X: Clean up qdp-python pyproject.toml
- [ ] #X: Update documentation for new test workflow

### Design Document

See [RFC-001: Python Infrastructure Consolidation](./001-python-infra-consolidation.md)

---

## Issue #2: Add root conftest.py with QDP auto-skip logic

**Labels:** `enhancement`, `infrastructure`, `good first issue`

### Description

Add a root-level `conftest.py` that automatically skips QDP-dependent tests when the `_qdp` native extension is not built.

### Context

Currently, running `pytest` from the repository root only runs tests in `testing/`. Once we consolidate tests, we need a mechanism to gracefully skip QDP tests for contributors who don't have CUDA or haven't built the extension.

### Acceptance Criteria

- [ ] Create `/conftest.py` at repository root
- [ ] Register `gpu` and `slow` pytest markers
- [ ] Auto-detect if `_qdp` extension is importable
- [ ] Skip tests marked with `@pytest.mark.gpu` if extension unavailable
- [ ] Skip all tests in `testing/qdp/` directory if extension unavailable
- [ ] Provide helpful skip message with build instructions
- [ ] Add `qdp_available` fixture for tests that need conditional QDP access

### Implementation

```python
"""Root pytest configuration for Apache Mahout."""
import pytest

_QDP_AVAILABLE = False
_QDP_IMPORT_ERROR = None

try:
    import _qdp
    _QDP_AVAILABLE = True
except ImportError as e:
    _QDP_IMPORT_ERROR = str(e)


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: tests that require GPU and _qdp extension")
    config.addinivalue_line("markers", "slow: tests that take a long time to run")


def pytest_collection_modifyitems(config, items):
    if _QDP_AVAILABLE:
        return

    skip_marker = pytest.mark.skip(
        reason=f"QDP extension not available: {_QDP_IMPORT_ERROR}. "
        "Build with: cd qdp/qdp-python && maturin develop"
    )

    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_marker)
        if "testing/qdp" in str(item.fspath):
            item.add_marker(skip_marker)


@pytest.fixture
def qdp_available():
    if not _QDP_AVAILABLE:
        pytest.skip(f"QDP extension not available: {_QDP_IMPORT_ERROR}")
    return True
```

### Testing

1. Run `pytest` without building `_qdp` - QDP tests should be skipped with helpful message
2. Build `_qdp` with `maturin develop` and run `pytest` - all tests should run
3. Verify `pytest -m "not gpu"` works correctly

### Dependencies

None - this can be done first.

---

## Issue #3: Consolidate dev dependencies in root pyproject.toml

**Labels:** `enhancement`, `infrastructure`

### Description

Move all development dependencies from `qdp/qdp-python/pyproject.toml` to the root `pyproject.toml` and create separate dependency groups for different development scenarios.

### Context

Currently we have:
- Root: `dev = ["pytest>=8.1.1", "ruff>=0.13.1", "pre-commit>=3.0.0"]`
- qdp-python: `dev = ["maturin>=1.10.2", "patchelf>=0.17.2.4", "pytest>=9.0.1", "torch>=2.2", "numpy>=1.24,<2.0"]`

These have different pytest versions and require separate `uv sync` commands.

### Acceptance Criteria

- [ ] Update root `pyproject.toml` with consolidated dependencies
- [ ] Create `qdp-dev` dependency group for QDP-specific tools
- [ ] Align pytest version to `>=9.0.1`
- [ ] Keep `benchmark` group in qdp-python for benchmark-specific heavy dependencies
- [ ] Verify `uv sync --group dev --group qdp-dev` installs everything needed
- [ ] Update any CI workflows that reference dependency groups

### Changes to root pyproject.toml

```toml
[dependency-groups]
dev = [
    "pytest>=9.0.1",
    "ruff>=0.13.1",
    "pre-commit>=3.0.0",
]
qdp-dev = [
    "maturin>=1.10.2",
    "patchelf>=0.17.2.4",
    "torch>=2.2",
    "numpy>=1.24,<2.0",
]
```

### Testing

1. `uv sync --group dev` - should install base dev tools
2. `uv sync --group dev --group qdp-dev` - should install all dev tools including maturin
3. Verify CI passes with updated dependency groups

### Dependencies

None - can be done in parallel with Issue #2.

---

## Issue #4: Reorganize test directory structure

**Labels:** `enhancement`, `infrastructure`

### Description

Move tests from `qdp/qdp-python/tests/` to `testing/qdp/` and reorganize existing tests into `testing/qumat/` subdirectory.

### Context

This is the main structural change that enables a unified test experience. After this change, all tests will be under `testing/` with clear separation between qumat and qdp tests.

### Current Structure

```
testing/
├── test_single_qubit_gates.py
├── test_rotation_gates.py
├── test_multi_qubit_gates.py
├── test_create_circuit.py
├── test_final_quantum_states.py
├── test_overlap_measurement.py
├── test_parameter_binding.py
├── test_swap_test.py
└── utils/

qdp/qdp-python/tests/
├── test_bindings.py
├── test_numpy.py
└── test_high_fidelity.py
```

### Target Structure

```
testing/
├── conftest.py              # Shared fixtures
├── qumat/
│   ├── __init__.py
│   ├── test_single_qubit_gates.py
│   ├── test_rotation_gates.py
│   ├── test_multi_qubit_gates.py
│   ├── test_create_circuit.py
│   ├── test_final_quantum_states.py
│   ├── test_overlap_measurement.py
│   ├── test_parameter_binding.py
│   └── test_swap_test.py
├── qdp/
│   ├── __init__.py
│   ├── test_bindings.py
│   ├── test_numpy.py
│   └── test_high_fidelity.py
└── utils/
    └── (unchanged)
```

### Acceptance Criteria

- [ ] Create `testing/qumat/` directory
- [ ] Move existing `testing/test_*.py` files to `testing/qumat/`
- [ ] Create `testing/qdp/` directory
- [ ] Move `qdp/qdp-python/tests/*.py` to `testing/qdp/`
- [ ] Add `__init__.py` files to new directories
- [ ] Update any relative imports in moved test files
- [ ] Ensure tests in `testing/qdp/` have `@pytest.mark.gpu` where appropriate
- [ ] Delete empty `qdp/qdp-python/tests/` directory
- [ ] All tests pass from repository root with `pytest`

### File Moves

| From | To |
|------|-----|
| `testing/test_single_qubit_gates.py` | `testing/qumat/test_single_qubit_gates.py` |
| `testing/test_rotation_gates.py` | `testing/qumat/test_rotation_gates.py` |
| `testing/test_multi_qubit_gates.py` | `testing/qumat/test_multi_qubit_gates.py` |
| `testing/test_create_circuit.py` | `testing/qumat/test_create_circuit.py` |
| `testing/test_final_quantum_states.py` | `testing/qumat/test_final_quantum_states.py` |
| `testing/test_overlap_measurement.py` | `testing/qumat/test_overlap_measurement.py` |
| `testing/test_parameter_binding.py` | `testing/qumat/test_parameter_binding.py` |
| `testing/test_swap_test.py` | `testing/qumat/test_swap_test.py` |
| `qdp/qdp-python/tests/test_bindings.py` | `testing/qdp/test_bindings.py` |
| `qdp/qdp-python/tests/test_numpy.py` | `testing/qdp/test_numpy.py` |
| `qdp/qdp-python/tests/test_high_fidelity.py` | `testing/qdp/test_high_fidelity.py` |

### Testing

1. `pytest testing/qumat` - all qumat tests pass
2. `pytest testing/qdp` - qdp tests run (or skip if extension not built)
3. `pytest` - all tests discoverable and run correctly

### Dependencies

- Issue #2 (conftest.py) should be merged first for auto-skip to work

---

## Issue #5: Clean up qdp-python pyproject.toml

**Labels:** `enhancement`, `infrastructure`, `good first issue`

### Description

Remove redundant configuration from `qdp/qdp-python/pyproject.toml` now that tests and dev dependencies are consolidated in the root.

### Context

After Issues #3 and #4 are complete, the qdp-python `pyproject.toml` will have orphaned configuration that should be removed.

### Acceptance Criteria

- [ ] Remove `[dependency-groups].dev` section (moved to root)
- [ ] Remove `[tool.pytest.ini_options]` section (consolidated in root)
- [ ] Keep `[dependency-groups].benchmark` (heavy deps specific to benchmarking)
- [ ] Keep `[build-system]`, `[project]`, `[tool.maturin]` sections
- [ ] Verify `maturin develop` still works
- [ ] Verify benchmarks can still be run with `uv sync --group benchmark`

### Before

```toml
[build-system]
requires = ["maturin>=1.10,<2.0"]
build-backend = "maturin"

[project]
name = "qumat-qdp"
requires-python = ">=3.10,<3.13"
# ...

[dependency-groups]
dev = [                          # REMOVE
    "maturin>=1.10.2",
    "patchelf>=0.17.2.4",
    "pytest>=9.0.1",
    "torch>=2.2",
    "numpy>=1.24,<2.0",
]
benchmark = [                    # KEEP
    "numpy>=1.24,<2.0",
    # ...
]

[tool.pytest.ini_options]        # REMOVE
testpaths = ["tests"]
markers = [...]

[tool.maturin]                   # KEEP
module-name = "_qdp"
```

### After

```toml
[build-system]
requires = ["maturin>=1.10,<2.0"]
build-backend = "maturin"

[project]
name = "qumat-qdp"
requires-python = ">=3.10,<3.13"
classifiers = [
    "Programming Language :: Rust",
    "Programming Language :: Python :: Implementation :: CPython",
]
dynamic = ["version"]

[dependency-groups]
benchmark = [
    "numpy>=1.24,<2.0",
    "pandas>=2.0",
    "pyarrow>=14.0",
    "tensorflow>=2.20",
    "torch>=2.2",
    "qiskit>=1.0",
    "qiskit-aer>=0.17.2",
    "pennylane>=0.35",
    "scikit-learn>=1.3",
    "tqdm",
    "matplotlib",
]

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu122"
explicit = true

[tool.maturin]
module-name = "_qdp"
```

### Testing

1. `cd qdp/qdp-python && maturin develop` - builds successfully
2. `cd qdp/qdp-python && uv sync --group benchmark` - installs benchmark deps
3. Root-level `pytest` still works

### Dependencies

- Issue #3 (consolidate dependencies) must be complete
- Issue #4 (reorganize tests) must be complete

---

## Issue #6: Update documentation for new test workflow

**Labels:** `documentation`, `infrastructure`

### Description

Update all documentation to reflect the new unified test workflow and project structure.

### Acceptance Criteria

- [ ] Update `README.md` with new development setup instructions
- [ ] Update `testing/README.md` with new directory structure
- [ ] Create or update `CONTRIBUTING.md` with full development workflow
- [ ] Add inline comments to `conftest.py` explaining the auto-skip logic
- [ ] Remove outdated instructions from `qdp/qdp-python/README.md`

### README.md Changes

Add/update development section:

```markdown
## Development

### Setup

```bash
# Clone and install dependencies
git clone https://github.com/apache/mahout.git
cd mahout
uv sync --group dev

# Optional: Build QDP extension (requires CUDA)
uv sync --group qdp-dev
uv run maturin develop -m qdp/qdp-python/Cargo.toml
```

### Running Tests

```bash
# Run all tests (QDP tests auto-skip if extension not built)
pytest

# Run specific test suites
pytest testing/qumat      # Quantum circuit tests
pytest testing/qdp        # QDP encoding tests (requires extension)

# Skip GPU-dependent tests
pytest -m "not gpu"
```
```

### testing/README.md Changes

Update to reflect new structure:

```markdown
# Apache Mahout Testing Suite

## Directory Structure

```
testing/
├── conftest.py       # Shared fixtures and auto-skip logic
├── qumat/            # Quantum circuit abstraction tests
│   └── test_*.py
├── qdp/              # QDP encoding tests (GPU required)
│   └── test_*.py
└── utils/            # Test utilities and backend helpers
```

## Running Tests

```bash
pytest                    # All tests
pytest testing/qumat      # Only qumat tests
pytest testing/qdp        # Only qdp tests
pytest -m "not gpu"       # Skip GPU tests
```

Note: QDP tests are automatically skipped if the `_qdp` extension is not built.
```

### Dependencies

- All previous issues should be complete before documentation is finalized

---

## Summary: Issue Dependency Graph

```
                    ┌─────────────────────────┐
                    │  #1 Tracking Issue      │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 │
    ┌─────────────────┐ ┌─────────────────┐      │
    │ #2 conftest.py  │ │ #3 Consolidate  │      │
    │ (auto-skip)     │ │ dependencies    │      │
    └────────┬────────┘ └────────┬────────┘      │
             │                   │               │
             └─────────┬─────────┘               │
                       │                         │
                       ▼                         │
             ┌─────────────────┐                 │
             │ #4 Reorganize   │                 │
             │ test structure  │                 │
             └────────┬────────┘                 │
                      │                          │
                      ▼                          │
             ┌─────────────────┐                 │
             │ #5 Clean up     │                 │
             │ qdp pyproject   │                 │
             └────────┬────────┘                 │
                      │                          │
                      ▼                          │
             ┌─────────────────┐                 │
             │ #6 Update docs  │◄────────────────┘
             └─────────────────┘
```

## Suggested PR Strategy

| PR | Issues | Description |
|----|--------|-------------|
| PR 1 | #2, #3 | Add conftest.py and consolidate deps (can be combined) |
| PR 2 | #4 | Reorganize test directory structure |
| PR 3 | #5 | Clean up qdp-python pyproject.toml |
| PR 4 | #6 | Documentation updates |

Each PR is independently reviewable and can be merged sequentially.
