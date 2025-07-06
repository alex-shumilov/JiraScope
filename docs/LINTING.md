# Linting and Code Quality

This project uses a comprehensive linting setup for Python 3.13 with GitHub Actions integration.

## Tools Used

- **Ruff**: Fast Python linter and formatter (replaces pylint, flake8, isort, and more)
- **MyPy**: Static type checker
- **Black**: Code formatter (backup to Ruff)

## Quick Start

### Install Dependencies
```bash
# Install development dependencies
make install-dev

# Or manually
pip install ruff mypy black
```

### Run Linting Locally

```bash
# Check all issues
make check-all

# Individual tools
make lint           # Run ruff linter (check only)
make lint-fix       # Run ruff linter with auto-fix
make format         # Format code with ruff
make format-check   # Check formatting without changes
make type-check     # Run mypy type checker

# Using the Python script
python scripts/lint.py           # Check all
python scripts/lint.py --fix     # Fix auto-fixable issues
```

## GitHub Actions Integration

The linting runs automatically on:
- Push to `main` branch
- Pull requests to `main` branch

### Features:
- ✅ **GitHub Annotations**: Issues are highlighted directly in PR diffs
- 🔧 **Auto-fix Suggestions**: PR checks show what can be auto-fixed
- 📊 **Summary Reports**: Clear results in GitHub Action summaries
- ⚡ **Fast Execution**: Ruff is ~100x faster than pylint
- 🐍 **Python 3.13 Only**: Enforces latest Python standards

## Ruff Configuration

Our ruff setup includes these rule sets:
- **E/W**: pycodestyle errors and warnings
- **F**: Pyflakes (undefined names, imports)
- **UP**: pyupgrade (modernize syntax)
- **B**: flake8-bugbear (common bugs)
- **SIM**: flake8-simplify (code simplification)
- **I**: isort (import sorting)
- **N**: pep8-naming (naming conventions)
- **PL**: Pylint rules
- **ARG**: unused arguments
- **PTH**: prefer pathlib
- **PERF**: performance lints
- **RUF**: Ruff-specific rules

### Ignored Rules
- `E501`: Line too long (handled by formatter)
- `PLR0913`: Too many function arguments (reasonable flexibility)
- `TRY003`: Long exception messages (project preference)

### Per-File Exceptions
- **Tests**: Allow magic values, print statements, assertions
- **Scripts**: Allow print statements and magic values

## IDE Integration

Most modern IDEs support ruff automatically:

### VS Code
Install the "Ruff" extension for real-time linting and formatting.

### PyCharm/IntelliJ
Enable ruff in Settings → Tools → External Tools or use the ruff plugin.

### Vim/Neovim
Use ALE, coc-ruff, or nvim-lspconfig with ruff-lsp.

## Configuration Files

- `pyproject.toml`: Main configuration (preferred)
- `ruff.toml`: Standalone config for IDE integration
- `.github/workflows/pylint.yml`: GitHub Actions workflow

## Troubleshooting

### Common Issues

1. **Module not found errors**: Install with `pip install -e .`
2. **Ruff not found**: Install with `pip install ruff`
3. **MyPy cache issues**: Run `make clean` to clear caches

### Performance Tips

- Ruff is extremely fast (~50ms for most projects)
- Use `--fix` flag to auto-fix most issues
- MyPy can be slow on first run but caches results

## Migration Notes

This setup replaces the old pylint configuration:
- ✅ **Faster**: Ruff is ~100x faster than pylint
- ✅ **More Comprehensive**: Includes formatter, import sorter
- ✅ **Better GHA Integration**: Native GitHub annotations
- ✅ **Modern**: Uses latest Python 3.13 features

Old commands are no longer supported:
- `pylint` → `ruff check`
- `flake8` → `ruff check`  
- `isort` → `ruff format` or `ruff check --select I`
- `black` → `ruff format` 