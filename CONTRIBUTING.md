# Contributing to A-Share Quantitative Trading Platform

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. Fork the repository
2. Clone your fork: `git clone git@github.com:YOUR_USERNAME/baostock_experiments.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Commit and push: `git push origin feature/your-feature-name`
7. Create a Pull Request

## 📋 Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🎯 Areas for Contribution

### 1. Strategy Development
- Create new strategy classes inheriting from `BaseStrategy`
- Add strategy templates to the Strategy Builder
- Improve existing strategies

Example:
```python
# strategies/your_strategy.py
from strategies.base import BaseStrategy, Signal, SignalType

class YourStrategy(BaseStrategy):
    def generate_signals(self, data, stock_code):
        # Your strategy logic
        pass
    
    def calculate_position_size(self, signal, portfolio_value, current_positions):
        # Position sizing logic
        pass
```

### 2. Technical Indicators
- Add new indicators to `core/indicators.py`
- Optimize existing indicator calculations
- Add indicator documentation

### 3. Data Sources
- Extend `DataManager` with new data types
- Improve caching mechanisms
- Add data validation

### 4. UI/UX Improvements
- Create new Streamlit pages
- Improve existing visualizations
- Add interactive features

## 📝 Code Style Guidelines

### Python Style
- Follow PEP 8
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Keep functions focused and small

### Naming Conventions
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Documentation
- Use Google-style docstrings
- Include examples in docstrings
- Update README.md if adding features
- Add inline comments for complex logic

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_strategies.py

# Run with coverage
pytest --cov=.
```

### Writing Tests
- Add tests for new features
- Maintain test coverage above 80%
- Test edge cases
- Use meaningful test names

## 📊 Performance Guidelines

- Cache expensive calculations
- Use vectorized operations with pandas/numpy
- Profile code for bottlenecks
- Consider memory usage for large datasets

## 🐛 Bug Reports

### Before Submitting
1. Check existing issues
2. Verify the bug in latest version
3. Isolate the problem

### Bug Report Template
```markdown
**Description**
A clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**
What should happen

**Screenshots**
If applicable

**Environment**
- OS: [e.g., Windows 10]
- Python version: [e.g., 3.9]
- Package versions: [from pip freeze]
```

## 🌟 Feature Requests

### Feature Request Template
```markdown
**Problem Statement**
What problem does this solve?

**Proposed Solution**
Your suggested implementation

**Alternatives Considered**
Other approaches you've thought about

**Additional Context**
Any other relevant information
```

## 📄 Pull Request Process

1. Update documentation
2. Add/update tests
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Request review from maintainers

### PR Title Format
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Maintenance tasks

## 🤝 Code of Conduct

### Our Standards
- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the community

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or insulting comments
- Public or private harassment
- Publishing others' private information

## 📞 Contact

- GitHub Issues: Bug reports and feature requests
- Email: [project-email@example.com]
- Discussion Forum: [link-to-forum]

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to make this platform better!

---

*Happy coding! 🚀*