# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Features
- (qdrant): Update filter usage to new client model (741046a)
- (web): enhance web interface and API endpoints (d4e474d)
- (utils): enhance utility modules and package structure (9fc5620)
- (pipeline): enhance processing pipeline and data extraction (2e6d6f1)
- (analysis): enhance analysis engines and algorithms (b07cbd0)
- (clients): enhance API client implementations (478f036)
- (models): update core data models (c0c3f92)
- (models): enhance core models and configuration (8f7c667)
- (cli): enhance CLI with comprehensive command structure (78c12b5)
- (quality): Add new quality testing and cost optimization modules to __all__ exports (153e8e3)
- (web): Add FastAPI backend and web dashboard for JiraScope analysis (20390e8)
- (web): Add web dashboard command and update Docker configurations (c51693a)

### Fixes
- (typing): improve type annotations for mypy compliance (183d213)
- (linter): address unused variable warnings (F841) (e975a2f)
- (imports): replace star imports with explicit imports (d766a2b)
- (security): resolve security issues identified by bandit (bc0412a)
- (pydantic): Fix Pydantic deprecation warnings in work_item.py (ed962bd)
- (tests): Fix warnings in tests, Pydantic deprecation, and TestQuery class name collision (c7b97e7)
- (tests): Fix failing tests by updating config and fixing imports (42542ff)
- (tests): Fix failing tests in multiple clients modules (3b4af29)
- (tests): Fix failing tests in temporal_analyzer module (6fc0acc)
- (tests): Fix failing tests in template_inference and temporal_analyzer modules (887732d)
- (tests): Fix failing tests in multiple analyzers (12598a1)
- (tests): Fix failing tests in structural and similarity analyzers (f97e354)
- (tests): Fix similarity analyzer test for no duplicates scenario (6a4ebc3)
- (tests): Fix similarity analyzer medium confidence level action text (f034900)
- (tests): Fix context manager tests in cross_epic_analyzer (a5fd4a7)
- (tests): Fix test_analyze_misplacement_with_claude in CrossEpicAnalyzer (09c8eca)
- (tests): claude fixing tests (9629439)

### Refactors
- (logging): introduce custom logger for cost tracking (4f0b349)
- (cli): remove old CLI structure (1bba70a)
- (analysis): Refactor similarity analysis methods and enhance tech debt identification logic (e075199)

### Chores
- (dev): add development tooling configuration (ae84754)
- (ide): update IDE configuration and development settings (27c25f6)
- (env): Remove EMBEDDING_MODEL_URL from .env.dist and update README.md (e85580a)
- (env): Add environment configuration template and update .gitignore (4cea039)
- (ide): Add IDE config files and setup gitignore (0bdc5b0)
- (ide): Add IDE config files and setup gitignore (123f38a)
- (git): initial commit (4f138f3)

### Build
- (pre-commit): Temporarily disable mypy (ade9065)
- (deps): update project configuration and dependencies (c981c37)

### Docs
- (readme): enhance project documentation (31fb274)
- (worklog): add comprehensive pre-commit fix worklog (d0d0092)

### Style
- (format): add missing newlines at end of files (8fb3e0d)

### Docker
- (setup): optimize containerization setup (085239a)

### Tests
- (coverage): enhance test suite coverage and reliability (3a907e3) 