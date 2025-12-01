# Contributing to Multimodal Data Pipeline

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional, for GPU features)
- Ray cluster (local or remote)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/nvidia-demo.git
cd nvidia-demo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write docstrings for all public functions
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
# Run unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=pipeline --cov-report=html

# Run integration tests
pytest tests/integration/ -v

# Run linting
ruff check pipeline/
mypy pipeline/

# Run formatting
ruff format pipeline/
```

### 4. Commit Changes

```bash
# Pre-commit hooks will run automatically
git add .
git commit -m "feat: add new feature"
```

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

## Code Style

- Use `ruff` for linting and formatting
- Follow type hints best practices
- Use descriptive variable names
- Add docstrings to all public functions
- Keep functions focused and small

## Testing Guidelines

- Write unit tests for all new functionality
- Aim for >80% code coverage
- Use fixtures for test data
- Mock external dependencies
- Test edge cases and error conditions

## Documentation

- Update README.md for user-facing changes
- Add docstrings to all public APIs
- Update CHANGELOG.md for user-visible changes
- Add examples for new features

## Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add changelog entry
4. Request review from maintainers
5. Address review comments
6. Squash commits if requested

## Questions?

Feel free to open an issue for questions or discussions.

