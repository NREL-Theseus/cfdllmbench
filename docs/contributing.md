# Contributing to CFDLLMBench

We welcome contributions from the community! This guide will help you get started.

## Ways to Contribute

- **Bug Reports**: Report issues you encounter
- **Feature Requests**: Suggest new evaluation tasks or metrics
- **Code Contributions**: Implement improvements or new features
- **Documentation**: Improve documentation and examples
- **Datasets**: Contribute additional CFD problems or test cases

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/cfdllmbench.git
cd cfdllmbench
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv cfd_env
source cfd_env/bin/activate  # On Windows: cfd_env\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .
```

### 3. Run Tests

```bash
# Run the test suite
pytest tests/

# Run with coverage
pytest --cov=cfdllmbench tests/
```

## Development Guidelines

### Code Style

We follow PEP 8 with some modifications:

```python
# Use black for formatting
black cfdllmbench/

# Use flake8 for linting  
flake8 cfdllmbench/

# Use mypy for type checking
mypy cfdllmbench/
```

### Documentation

- Document all public functions and classes
- Use Google-style docstrings
- Include examples in docstrings
- Update API documentation when adding features

```python
def evaluate_model(self, model: LLMInterface) -> Dict[str, float]:
    """Evaluate language model on CFD questions.
    
    Args:
        model: The language model interface to evaluate.
        
    Returns:
        Dictionary containing accuracy and detailed metrics.
        
    Example:
        >>> evaluator = CFDQueryEvaluator("questions.json")
        >>> results = evaluator.evaluate_model(my_model)
        >>> print(f"Accuracy: {results['accuracy']:.2%}")
    """
```

### Testing

- Write tests for all new functionality
- Aim for >90% test coverage
- Use pytest fixtures for setup
- Test both success and failure cases

```python
def test_cfdquery_evaluation():
    """Test CFDQuery evaluation with mock model."""
    evaluator = CFDQueryEvaluator("test_data/questions.json")
    mock_model = MockLLM(responses=["A", "B", "C", "D"])
    
    results = evaluator.evaluate_model(mock_model)
    
    assert "accuracy" in results
    assert 0 <= results["accuracy"] <= 1
```

## Adding New Components

### 1. New Evaluation Task

To add a new evaluation component:

```python
# 1. Create evaluator class
class NewTaskEvaluator:
    def __init__(self, config_file: str):
        self.config = load_config(config_file)
        
    def evaluate_model(self, model: LLMInterface) -> Dict:
        # Implementation here
        pass

# 2. Add to main benchmark
class CFDLLMBench:
    def add_evaluator(self, name: str, evaluator: Evaluator):
        self.evaluators[name] = evaluator
```

### 2. New Model Interface

To support a new LLM provider:

```python
class NewModelInterface(LLMInterface):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        # Initialize model connection
        
    def generate(self, prompt: str) -> str:
        # Implement model inference
        pass
        
    def batch_generate(self, prompts: List[str]) -> List[str]:
        # Implement batch inference
        pass
```

## Submitting Changes

### 1. Create Feature Branch

```bash
git checkout -b feature/new-evaluation-task
```

### 2. Make Changes

- Follow coding guidelines
- Add tests for new functionality
- Update documentation
- Ensure all tests pass

### 3. Commit Changes

```bash
git add .
git commit -m "Add new CFD evaluation task

- Implement heat transfer problem set
- Add corresponding test cases  
- Update documentation"
```

### 4. Submit Pull Request

1. Push branch to your fork
2. Create pull request on GitHub
3. Describe changes and motivation
4. Link to relevant issues

## Pull Request Guidelines

### Checklist

- [ ] Code follows style guidelines
- [ ] Tests added for new functionality
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Changelog entry added
- [ ] No merge conflicts

### Review Process

1. **Automated checks**: CI runs tests and style checks
2. **Code review**: Maintainers review changes
3. **Discussion**: Address feedback and questions
4. **Approval**: Changes approved by maintainer
5. **Merge**: Changes merged to main branch

## Community

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Email**: Direct contact for sensitive issues

### Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. Please read our [Code of Conduct](CODE_OF_CONDUCT.md).

## Recognition

Contributors are recognized in:

- Release notes
- Contributors file
- Project documentation
- Conference papers (for significant contributions)

Thank you for contributing to CFDLLMBench!
