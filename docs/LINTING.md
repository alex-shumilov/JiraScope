# Code Quality and Type Checking

This project uses type checking and code formatting for Python 3.13 with GitHub Actions integration.

## Tools Used

- **MyPy**: Static type checker
- **Black**: Code formatter

## Quick Start

### Install Dependencies
```bash
# Install development dependencies
make install-dev

# Or manually
pip install mypy black
```

### Run Type Checking Locally

```bash
# Run type checking
make type-check

# Using the Python script
python scripts/lint.py           # Run type checking

# Format code
black src/
black --check src/               # Check formatting without changes
```

## GitHub Actions Integration

Type checking runs automatically on:
- Push to `main` branch
- Pull requests to `main` branch

### Features:
- ✅ **GitHub Annotations**: Type errors are highlighted directly in PR diffs
- 📊 **Summary Reports**: Clear results in GitHub Action summaries
- 🔍 **Type Safety**: MyPy catches type-related bugs before runtime
- 🐍 **Python 3.13 Only**: Enforces latest Python standards

## MyPy Configuration

Our mypy setup includes:
- **Gradual typing**: Start lenient, gradually make stricter
- **Type checking**: Focus on core modules with strict checking
- **Error reporting**: Show error codes and pretty formatting
- **Missing imports**: Currently ignored for external libraries

### Configuration Highlights
- `warn_return_any`: Disabled initially for gradual adoption
- `disallow_untyped_defs`: Enabled for core modules only
- `ignore_missing_imports`: True for external libraries
- `show_error_codes`: Detailed error information

## IDE Integration

Most modern IDEs support mypy and black:

### VS Code
Install the "Python" extension which includes mypy support and black formatting.

### PyCharm/IntelliJ
Built-in support for mypy and black in the Python plugin.

### Vim/Neovim
Use ALE or nvim-lspconfig with mypy and black support.

## Configuration Files

- `pyproject.toml`: Main configuration for mypy and black
- `.github/workflows/pylint.yml`: GitHub Actions workflow

## Troubleshooting

### Common Issues

1. **Module not found errors**: Install with `pip install -e .`
2. **MyPy not found**: Install with `pip install mypy`
3. **MyPy cache issues**: Run `make clean` to clear caches
4. **Type errors**: Review mypy configuration in `pyproject.toml`

### Performance Tips

- MyPy can be slow on first run but caches results
- Black is very fast for code formatting
- Use incremental type checking for better performance

## Philosophy

This setup focuses on:
- ✅ **Type Safety**: Catch errors before runtime with MyPy
- ✅ **Simplicity**: Fewer tools, clearer purpose
- ✅ **Reliability**: Well-established tools with broad community support
- ✅ **Modern**: Uses latest Python 3.13 features
