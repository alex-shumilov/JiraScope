"""Tests for configuration management."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from src.jirascope.core.config import Config


def test_config_from_env(monkeypatch):
    """Test loading configuration from environment variables."""
    monkeypatch.setenv("JIRA_MCP_ENDPOINT", "http://test:8080/mcp")
    monkeypatch.setenv("CLAUDE_API_KEY", "test-key-123")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "16")
    monkeypatch.setenv("JIRA_DRY_RUN", "false")
    monkeypatch.setenv("CLAUDE_MODEL", "test-model")

    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = False
        config = Config.from_env()

    assert config.jira_mcp_endpoint == "http://test:8080/mcp"
    assert config.claude_api_key == "test-key-123"
    assert config.embedding_batch_size == 16
    assert config.jira_dry_run is False
    assert config.claude_model == "test-model"


def test_config_from_file():
    """Test loading configuration from YAML file."""
    config_data = {
        "jira_mcp_endpoint": "http://file:8080/mcp",
        "claude_api_key": "file-key-123",
        "embedding_batch_size": 64,
        "jira_dry_run": False,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = Path(f.name)

    try:
        config = Config.from_file(config_path)

        assert config.jira_mcp_endpoint == "http://file:8080/mcp"
        assert config.claude_api_key == "file-key-123"
        assert config.embedding_batch_size == 64
        assert config.jira_dry_run is False
    finally:
        config_path.unlink()


def test_config_load_priority(monkeypatch):
    """Test configuration loading priority: file > env > defaults."""
    # Set environment variables
    monkeypatch.setenv("JIRA_MCP_ENDPOINT", "http://env:8080/mcp")
    monkeypatch.setenv("CLAUDE_API_KEY", "env-key")

    # Create temporary config file
    config_data = {"jira_mcp_endpoint": "http://file:8080/mcp", "claude_api_key": "file-key"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = Path(f.name)

    try:
        # Test file takes priority
        config = Config.load(config_path)
        assert config.jira_mcp_endpoint == "http://file:8080/mcp"
        assert config.claude_api_key == "file-key"

        # Test env fallback when no file
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False
            config = Config.load(Path("nonexistent.yaml"))
            assert config.jira_mcp_endpoint == "http://env:8080/mcp"
            assert config.claude_api_key == "env-key"
    finally:
        config_path.unlink()


def test_config_defaults(monkeypatch):
    """Test default configuration values."""
    # Ensure no env vars or files are loaded
    for var in [
        "JIRA_MCP_ENDPOINT",
        "LMSTUDIO_ENDPOINT",
        "QDRANT_URL",
        "CLAUDE_MODEL",
        "EMBEDDING_BATCH_SIZE",
    ]:
        monkeypatch.delenv(var, raising=False)

    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = False
        config = Config(jira_mcp_endpoint="http://default-mcp")

    assert config.lmstudio_endpoint == "http://localhost:1234/v1"
    assert config.qdrant_url == "http://localhost:6333"
    assert config.claude_model == "claude-3-5-sonnet-latest"
    assert config.embedding_batch_size == 32
    assert config.jira_batch_size == 100
    assert config.similarity_threshold == 0.8
    assert config.report_retention_days == 30
    assert config.cost_tracking is True
    assert config.jira_dry_run is True
