# Contributing to OPENtransformer

Thank you for your interest in contributing to OPENtransformer! This guide will help you get started with the contribution process.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Style](#code-style)
4. [Testing](#testing)
5. [Documentation](#documentation)
6. [Pull Requests](#pull-requests)
7. [Release Process](#release-process)

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/OPENtransformer.git
   cd OPENtransformer
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/original/OPENtransformer.git
   ```
4. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip
- virtualenv or conda

### Setup Development Environment

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_file.py

# Run with coverage
pytest --cov=OPENtransformer
```

## Code Style

### Python Style Guide

We follow PEP 8 style guide with some additional rules:

1. Maximum line length: 88 characters (Black formatter)
2. Use type hints for all function parameters and return values
3. Use docstrings for all public functions and classes
4. Use meaningful variable and function names

### Code Formatting

We use the following tools for code formatting:

1. Black for code formatting
2. isort for import sorting
3. flake8 for linting
4. mypy for type checking

Run the formatters:

```bash
# Format code
black .

# Sort imports
isort .

# Run linter
flake8

# Check types
mypy .
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit hooks manually
pre-commit run --all-files
```

## Testing

### Writing Tests

1. Place tests in the `tests` directory
2. Use pytest for testing
3. Follow the naming convention: `test_*.py`
4. Use fixtures for common setup

Example test:

```python
import pytest
from OPENtransformer import EasyDiffusionAPI

def test_diffusion_api_initialization():
    api = EasyDiffusionAPI()
    assert api is not None
    assert api.config is not None

@pytest.mark.parametrize("model_name", [
    "sd-v1-5",
    "sd-v2-1"
])
def test_model_loading(model_name):
    api = EasyDiffusionAPI()
    model = api.load_model(model_name)
    assert model is not None
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_file.py

# Run with coverage
pytest --cov=OPENtransformer

# Run specific test
pytest tests/test_file.py::test_function
```

## Documentation

### Writing Documentation

1. Use Markdown for documentation
2. Follow the existing documentation structure
3. Include code examples
4. Add type hints and docstrings

Example docstring:

```python
def process_image(image: Image, config: Optional[Dict] = None) -> Dict[str, Any]:
    """Process an image with the given configuration.

    Args:
        image (Image): Input image to process
        config (Optional[Dict]): Configuration dictionary

    Returns:
        Dict[str, Any]: Processing results

    Raises:
        ValueError: If image is invalid
        ResourceError: If processing fails
    """
    pass
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# Serve documentation locally
python -m http.server -d _build/html
```

## Pull Requests

### Creating a Pull Request

1. Update your fork:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Push your changes:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Create a pull request on GitHub

### Pull Request Guidelines

1. Use a descriptive title
2. Reference related issues
3. Include a detailed description
4. Add tests for new features
5. Update documentation
6. Ensure all tests pass
7. Follow the code style guide

## Release Process

### Versioning

We follow semantic versioning (MAJOR.MINOR.PATCH):

- MAJOR: Breaking changes
- MINOR: New features
- PATCH: Bug fixes

### Creating a Release

1. Update version in `setup.py`
2. Update changelog
3. Create release branch
4. Run tests
5. Build documentation
6. Create GitHub release
7. Publish to PyPI

### Publishing to PyPI

```bash
# Build distribution
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*
```

## Support

For help with contributing:
1. Check the [documentation](README.md)
2. Search [existing issues](https://github.com/yourusername/OPENtransformer/issues)
3. Create a new issue if needed 