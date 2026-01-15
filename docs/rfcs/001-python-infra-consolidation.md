# RFC-001: Python Infrastructure Consolidation

## Status
**Draft** | Author: @ryankert | Created: 2026-01-15

## Summary

Consolidate the Python project infrastructure to provide a unified developer experience for the `qumat` package and its `qdp` submodule. This includes unifying test locations, consolidating dependencies, and simplifying the development workflow.

## Motivation

The current project structure has evolved organically, resulting in:

1. **Fragmented test structure** - Tests are split between `testing/` (qumat) and `qdp/qdp-python/tests/` (qdp), requiring different working directories and commands to run.

2. **Duplicate configurations** - Two separate `pyproject.toml` files with overlapping but inconsistent pytest configurations and dev dependencies.

3. **Complex development workflow** - Contributors must run multiple commands from different directories to set up the development environment and run all tests.

4. **Inconsistent dependency versions** - pytest `>=8.1.1` in root vs `>=9.0.1` in qdp-python.

These issues create friction for new contributors and increase maintenance burden.

## Goals

- **Single command to run all tests** from repository root
- **Unified pytest configuration** with consistent markers and settings
- **Consolidated dev dependencies** in one location
- **Clear separation** between qumat core tests and qdp (GPU-dependent) tests
- **Graceful degradation** - tests should skip appropriately when qdp extension isn't built

## Non-Goals

- Changing the package structure of `qumat` or `_qdp`
- Modifying the Rust/maturin build process
- Changing how `qumat.qdp` is exposed to end users

## Current State

```
mahout/
├── pyproject.toml              # qumat package, testpaths=["testing"]
├── qumat/
│   ├── __init__.py
│   ├── qumat.py
│   ├── qdp.py                  # Bridge to _qdp extension
│   └── *_backend.py
├── testing/                    # qumat tests only
│   ├── test_*.py
│   └── utils/
└── qdp/
    └── qdp-python/
        ├── pyproject.toml      # qumat-qdp package, testpaths=["tests"]
        └── tests/              # qdp tests only
            └── test_*.py
```

**Current workflow:**
```bash
# Setup
uv sync --group dev
cd qdp/qdp-python && uv sync --group dev && maturin develop && cd ../..

# Run tests (two separate commands)
pytest                                    # Only testing/
cd qdp/qdp-python && pytest && cd ../..   # Only qdp tests
```

## Proposed Design

### Directory Structure

```
mahout/
├── pyproject.toml              # Unified configuration
├── conftest.py                 # Root pytest configuration
├── qumat/
│   └── (unchanged)
├── testing/
│   ├── conftest.py             # Test fixtures and helpers
│   ├── qumat/                  # Renamed from flat structure
│   │   ├── test_single_qubit_gates.py
│   │   ├── test_rotation_gates.py
│   │   ├── test_multi_qubit_gates.py
│   │   ├── test_create_circuit.py
│   │   ├── test_final_quantum_states.py
│   │   ├── test_overlap_measurement.py
│   │   ├── test_parameter_binding.py
│   │   └── test_swap_test.py
│   ├── qdp/                    # Moved from qdp/qdp-python/tests/
│   │   ├── test_bindings.py
│   │   ├── test_numpy.py
│   │   └── test_high_fidelity.py
│   └── utils/
│       └── (unchanged)
└── qdp/
    └── qdp-python/
        ├── pyproject.toml      # Build config only (no test config)
        └── (no tests/ directory)
```

### Unified pyproject.toml

```toml
[project]
name = "qumat"
# ... existing config ...

[project.optional-dependencies]
qdp = ["qumat-qdp"]

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

[tool.pytest.ini_options]
testpaths = ["testing"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = ["-v", "--tb=short"]
markers = [
    "gpu: tests that require GPU and _qdp extension",
    "slow: tests that take a long time to run",
]
```

### Root conftest.py

```python
"""Root pytest configuration for Apache Mahout."""
import pytest

# Check if QDP extension is available
_QDP_AVAILABLE = False
_QDP_IMPORT_ERROR = None

try:
    import _qdp
    _QDP_AVAILABLE = True
except ImportError as e:
    _QDP_IMPORT_ERROR = str(e)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: tests that require GPU and _qdp extension")
    config.addinivalue_line("markers", "slow: tests that take a long time to run")


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU tests if _qdp extension is not available."""
    if _QDP_AVAILABLE:
        return

    skip_marker = pytest.mark.skip(
        reason=f"QDP extension not available: {_QDP_IMPORT_ERROR}. "
        "Build with: cd qdp/qdp-python && maturin develop"
    )

    for item in items:
        # Skip tests marked with @pytest.mark.gpu
        if "gpu" in item.keywords:
            item.add_marker(skip_marker)
        # Skip all tests in testing/qdp/ directory
        if "testing/qdp" in str(item.fspath) or "testing\\qdp" in str(item.fspath):
            item.add_marker(skip_marker)


@pytest.fixture
def qdp_available():
    """Fixture that skips test if QDP is not available."""
    if not _QDP_AVAILABLE:
        pytest.skip(f"QDP extension not available: {_QDP_IMPORT_ERROR}")
    return True
```

### Simplified qdp-python/pyproject.toml

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

# Remove [dependency-groups] - consolidated in root
# Remove [tool.pytest.ini_options] - consolidated in root

[tool.maturin]
module-name = "_qdp"
```

### New Workflow

```bash
# Setup (one command)
uv sync --group dev --group qdp-dev

# Build QDP extension (if needed)
uv run maturin develop -m qdp/qdp-python/Cargo.toml

# Run all tests
pytest

# Run specific test subsets
pytest testing/qumat           # Only qumat tests
pytest testing/qdp             # Only qdp tests (skipped if extension not built)
pytest -m "not gpu"            # Skip GPU-dependent tests
pytest -m "not slow"           # Skip slow tests
```

## Migration Path

The migration will be done in phases to ensure reviewability:

1. **Phase 1**: Add root `conftest.py` with auto-skip logic
2. **Phase 2**: Consolidate dev dependencies in root `pyproject.toml`
3. **Phase 3**: Reorganize test directory structure
4. **Phase 4**: Clean up qdp-python `pyproject.toml`
5. **Phase 5**: Update documentation and CI

## Alternatives Considered

### Alternative A: Keep tests in place, update pytest discovery

```toml
[tool.pytest.ini_options]
testpaths = ["testing", "qdp/qdp-python/tests"]
```

**Pros:**
- Minimal file moves
- Less disruptive

**Cons:**
- Tests still conceptually split
- Harder to understand project structure
- qdp-python tests mixed with qdp-python source

**Decision:** Rejected. The consolidated approach provides better long-term maintainability.

### Alternative B: Monorepo with separate packages

Use tools like `uv workspaces` to treat qumat and qdp as completely separate packages with their own test suites.

**Pros:**
- Clear separation of concerns
- Independent versioning

**Cons:**
- More complex tooling setup
- Overkill for current project size
- qdp is semantically a submodule of qumat, not a peer

**Decision:** Rejected. The submodule relationship (`qumat.qdp`) should be reflected in the project structure.

## Testing Strategy

- All existing tests must pass after migration
- CI should run `pytest` from root to validate unified configuration
- CI should have jobs that test with and without qdp extension built

## Documentation Updates

- Update `README.md` quick start section
- Update `testing/README.md` with new structure
- Update `CONTRIBUTING.md` (if exists) with development workflow

## Open Questions

1. Should we add a `Makefile` or `justfile` for common development tasks?
2. Should benchmark files in `qdp/qdp-python/benchmark/` also be consolidated?
3. Do we need a `pytest-gpu` marker that checks for actual GPU availability vs just extension availability?

## References

- [pytest documentation on conftest.py](https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files)
- [uv workspaces](https://docs.astral.sh/uv/concepts/workspaces/)
- [maturin documentation](https://www.maturin.rs/)
