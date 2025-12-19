# DeepIsles Test Suite

This directory contains the test suite for the DeepIsles ensemble algorithm for ischemic stroke lesion segmentation.

## Overview

The test suite uses [pytest](https://docs.pytest.org/) for testing and includes:

- Unit tests for core functionality (`test_isles22_ensemble.py`)
- CLI interface tests (`test_main.py`)
- Majority voting tests (`test_majority_voting.py`)
- Utility function tests (`test_utils.py`)
- Shared fixtures and test utilities (`conftest.py`, `fixtures/`)

## Running Tests

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test File

```bash
pytest tests/test_isles22_ensemble.py
```

### Run Specific Test Class or Test Function

```bash
pytest tests/test_isles22_ensemble.py::TestIslesEnsembleInit
pytest tests/test_isles22_ensemble.py::TestIslesEnsembleInit::test_init
```

### Run with Coverage

```bash
pytest --cov=src tests/
```

### Run with Verbose Output

```bash
pytest -v tests/
```

### Run Tests Matching a Pattern

```bash
pytest -k "test_check_images" tests/
```

### Exclude Slow Tests

```bash
pytest -m "not slow" tests/
```

## Test Markers

The following pytest markers are available:

- `@pytest.mark.slow` - Marks tests as slow (may take longer to execute)
- `@pytest.mark.requires_gpu` - Marks tests that require GPU
- `@pytest.mark.integration` - Marks integration tests

Example usage:
```python
@pytest.mark.slow
def test_long_running_function():
    # Test code
    pass
```

## Test Fixtures

The `conftest.py` file provides shared fixtures for all tests:

- `tmp_dir` - Creates a temporary directory for test files
- `sample_dwi_path` - Creates a sample DWI NIfTI file
- `sample_adc_path` - Creates a sample ADC NIfTI file
- `sample_flair_path` - Creates a sample FLAIR NIfTI file
- `sample_mask_path` - Creates a sample binary mask file
- `ensemble_path` - Creates a mock ensemble directory structure
- `mock_subprocess_run`, `mock_subprocess_call` - Mock subprocess calls
- `mock_gpu_available`, `mock_gpu_unavailable` - Mock GPU availability checks
- And more...

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── test_isles22_ensemble.py # Main ensemble class tests
├── test_main.py             # CLI interface tests
├── test_majority_voting.py  # Majority voting logic tests
├── test_utils.py            # Utility function tests
├── fixtures/                # Test fixture utilities
│   ├── __init__.py
│   └── sample_images.py     # Image generation utilities
└── README.md                # This file
```

## Writing New Tests

When writing new tests:

1. Follow the existing test structure and naming conventions
2. Use fixtures from `conftest.py` when possible
3. Use appropriate markers (`@pytest.mark.slow`, etc.)
4. Mock external dependencies (file system, GPU, subprocess calls)
5. Keep tests isolated and independent
6. Use descriptive test names that explain what is being tested

Example:
```python
def test_new_feature_with_valid_input(tmp_dir, sample_dwi_path):
    """Test new feature with valid input parameters."""
    # Arrange
    ensemble = IslesEnsemble()
    ensemble.input_dwi_path = sample_dwi_path
    
    # Act
    result = ensemble.new_feature()
    
    # Assert
    assert result is not None
    assert result.shape == expected_shape
```

## Continuous Integration

These tests are designed to run in CI/CD environments. They:

- Use temporary directories that are automatically cleaned up
- Mock GPU availability checks for environments without GPU
- Mock external dependencies (SimpleITK, subprocess calls, etc.)
- Use deterministic test data (synthetic NIfTI images)

## Troubleshooting

### Tests Fail Due to Missing Dependencies

Ensure all test dependencies are installed:
```bash
pip install -e .[dev]  # If dev dependencies are specified
# or
pytest --fixtures  # Check available fixtures
```

### Tests Fail Due to Path Issues

Ensure the `src/` directory is in the Python path. The test files add it automatically:
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
```

### GPU Tests Fail

GPU-dependent tests are marked with `@pytest.mark.requires_gpu` and use mocked GPU checks. To skip them:
```bash
pytest -m "not requires_gpu" tests/
```

## Contributing

When contributing new code:

1. Write tests for new functionality
2. Ensure all tests pass: `pytest tests/`
3. Maintain or improve test coverage
4. Update this README if adding new test markers or fixtures

