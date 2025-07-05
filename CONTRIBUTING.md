# Contributing to ML Master

Thank you for your interest in contributing to ML Master! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style and Standards](#code-style-and-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)
- [Issue Reporting](#issue-reporting)
- [Documentation](#documentation)
- [Community Guidelines](#community-guidelines)

## Getting Started

Before contributing, please:

1. Read the [README.md](README.md) to understand the project
2. Review the research paper: "ML-Master: Towards AI-for-AI via Integration of Exploration and Reasoning" (arXiv:2506.16499v1)
3. Check existing issues and pull requests
4. Join our community discussions

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Azure OpenAI API access (for testing)

### Local Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd ml_master

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt  # If available
pip install black flake8 pytest pytest-cov

# Set up environment variables
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

### Project Structure

```
ml_master/
├── core/                    # Core framework components
│   ├── exploration/         # Multi-trajectory exploration
│   ├── reasoning/          # Steerable reasoning
│   ├── memory/             # Adaptive memory system
│   └── integration/        # Framework integration
├── agents/                 # LLM agent implementations
├── environments/           # Execution environments
├── benchmarks/            # Benchmark integration
├── utils/                 # Utility functions
├── configs/               # Configuration files
├── examples/              # Usage examples
├── tests/                 # Test suite
└── docs/                  # Documentation
```

## Code Style and Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Import organization**: Standard library, third-party, local imports
- **Docstrings**: Google style docstrings
- **Type hints**: Required for all public functions

### Code Formatting

We use Black for code formatting:

```bash
# Format code
black .

# Check formatting
black --check .
```

### Linting

We use flake8 for linting:

```bash
# Run linter
flake8 .

# Run with specific configuration
flake8 --config=.flake8 .
```

### Type Checking

We use mypy for type checking:

```bash
# Run type checker
mypy core/ tests/
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov-report=html

# Run specific test file
pytest tests/test_exploration.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Mock external dependencies
- Test both success and failure cases

### Test Structure

```python
def test_function_name():
    """Test description."""
    # Arrange
    input_data = "test"
    
    # Act
    result = function_to_test(input_data)
    
    # Assert
    assert result == expected_output
```

### Integration Tests

For integration tests involving Azure OpenAI:

```python
@pytest.mark.integration
def test_azure_openai_integration():
    """Test Azure OpenAI integration (requires API key)."""
    # Skip if no API key
    if not os.getenv('AZURE_OPENAI_API_KEY'):
        pytest.skip("Azure OpenAI API key not available")
    
    # Test implementation
```

## Submitting Changes

### Workflow

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Write/update tests**
5. **Run tests and linting**
   ```bash
   pytest
   black --check .
   flake8 .
   mypy core/
   ```
6. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```
7. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Create a pull request**

### Commit Message Format

We follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Example:
```
feat: add adaptive memory decay mechanism

- Implement exponential decay for memory entries
- Add configurable decay parameters
- Update memory relevance scoring
```

### Pull Request Guidelines

- **Title**: Clear, descriptive title
- **Description**: Detailed description of changes
- **Related issues**: Link to related issues
- **Testing**: Describe how you tested the changes
- **Breaking changes**: Note any breaking changes
- **Screenshots**: Include screenshots for UI changes

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** on different environments
4. **Documentation** updates if needed
5. **Approval** from at least one maintainer

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

- **Environment**: OS, Python version, dependencies
- **Steps to reproduce**: Clear, step-by-step instructions
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full error traceback
- **Screenshots**: If applicable

### Feature Requests

When requesting features, please include:

- **Problem description**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Use cases**: Examples of how it would be used
- **Alternatives**: Other solutions you've considered

### Issue Templates

Use the provided issue templates:
- Bug report template
- Feature request template
- Documentation request template

## Documentation

### Code Documentation

- **Docstrings**: Required for all public functions and classes
- **Type hints**: Required for all public APIs
- **Comments**: Explain complex logic, not obvious code

### API Documentation

- **Function signatures**: Clear parameter descriptions
- **Return values**: Document return types and values
- **Examples**: Include usage examples
- **Error handling**: Document exceptions and error conditions

### User Documentation

- **README.md**: Project overview and quick start
- **Examples**: Working code examples
- **Configuration**: Configuration options and examples
- **Troubleshooting**: Common issues and solutions

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Be respectful and inclusive
- Use welcoming and inclusive language
- Be collaborative and constructive
- Focus on what is best for the community
- Show empathy towards other community members

### Communication

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for general questions
- **Pull requests**: Use pull requests for code contributions
- **Email**: Use email for sensitive or private matters

### Recognition

Contributors will be recognized in:

- **Contributors list**: GitHub contributors page
- **Release notes**: Mentioned in release announcements
- **Documentation**: Listed in documentation
- **Acknowledgments**: Listed in README.md

## Getting Help

If you need help:

1. **Check documentation**: README.md, docstrings, examples
2. **Search issues**: Look for similar issues
3. **Ask questions**: Use GitHub Discussions
4. **Contact maintainers**: For urgent or private matters

## Development Roadmap

### Current Focus Areas

- **Performance optimization**: Improve execution speed
- **Memory efficiency**: Reduce memory usage
- **Error handling**: Better error recovery
- **Testing coverage**: Increase test coverage
- **Documentation**: Improve documentation quality

### Future Enhancements

- **Additional LLM providers**: Support for more LLM APIs
- **Web interface**: Web-based user interface
- **Plugin system**: Extensible plugin architecture
- **Distributed execution**: Support for distributed computing
- **Advanced analytics**: Enhanced performance analytics

## License

By contributing to ML Master, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Thank you to all contributors who have helped make ML Master what it is today. Your contributions are greatly appreciated! 