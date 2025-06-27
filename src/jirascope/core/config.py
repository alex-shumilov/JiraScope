"""Configuration management for JiraScope."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv


@dataclass
class Config:
    """Core configuration for JiraScope."""

    jira_mcp_endpoint: str
    jira_sse_client_id: str = "jirascope-cli"
    jira_sse_client_secret: str = ""
    jira_sse_redirect_port: int = 0
    jira_sse_scope: str = "read:jira-work read:jira-user write:jira-work"
    lmstudio_endpoint: str = "http://localhost:1234/v1"
    qdrant_url: str = "http://localhost:6333"
    claude_api_key: str = ""
    claude_model: str = "claude-3-5-sonnet-latest"

    embedding_batch_size: int = 32
    jira_batch_size: int = 100

    rag_test_queries: List[str] = field(default_factory=list)
    similarity_threshold: float = 0.8

    report_retention_days: int = 30
    cost_tracking: bool = True

    daily_budget: float = 50.0
    monthly_budget: float = 1000.0

    jira_dry_run: bool = True
    sentry_dsn: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        # Load .env file if it exists
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)

        # Check for required environment variables
        jira_mcp_endpoint = os.getenv("JIRA_MCP_ENDPOINT")
        if not jira_mcp_endpoint:
            raise ValueError(
                "JIRA_MCP_ENDPOINT environment variable is required. "
                "Please set it in your .env file or environment."
            )

        return cls(
            jira_mcp_endpoint=jira_mcp_endpoint,
            jira_sse_client_id=os.getenv("JIRA_SSE_CLIENT_ID", "jirascope-cli"),
            jira_sse_client_secret=os.getenv("JIRA_SSE_CLIENT_SECRET", ""),
            jira_sse_redirect_port=int(os.getenv("JIRA_SSE_REDIRECT_PORT", "0")),
            jira_sse_scope=os.getenv(
                "JIRA_SSE_SCOPE", "read:jira-work read:jira-user write:jira-work"
            ),
            lmstudio_endpoint=os.getenv("LMSTUDIO_ENDPOINT", "http://localhost:1234/v1"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            claude_api_key=os.getenv("CLAUDE_API_KEY", ""),
            claude_model=os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            jira_batch_size=int(os.getenv("JIRA_BATCH_SIZE", "100")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.8")),
            report_retention_days=int(os.getenv("REPORT_RETENTION_DAYS", "30")),
            cost_tracking=os.getenv("COST_TRACKING", "true").lower() == "true",
            daily_budget=float(os.getenv("DAILY_BUDGET", "50.0")),
            monthly_budget=float(os.getenv("MONTHLY_BUDGET", "1000.0")),
            jira_dry_run=os.getenv("JIRA_DRY_RUN", "true").lower() == "true",
            sentry_dsn=os.getenv("SENTRY_DSN"),
        )

    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """Load configuration with fallback priority: file -> env -> defaults."""
        if config_path and config_path.exists():
            return cls.from_file(config_path)

        default_config_path = Path.home() / ".jirascope" / "config.yaml"
        if default_config_path.exists():
            return cls.from_file(default_config_path)

        return cls.from_env()


EMBEDDING_CONFIG = {
    "model": "text-embedding-bge-large-en-v1.5",
    "api_base": "http://localhost:1234/v1",
    "dimensions": 1024,
    "batch_size": 32,
    "instruction_prefix": "Represent this Jira work item for semantic search: ",
    "max_tokens": 512,
    "timeout": 30.0,
}

CLAUDE_CONFIG = {
    "model": "claude-3-5-sonnet-latest",
    "max_tokens": 4096,
    "temperature": 0.1,
    "cost_per_token": {"input": 0.000003, "output": 0.000015},
    "daily_budget": 50.0,
    "session_budget": 10.0,
}

SAFETY_CONFIG = {
    "jira_dry_run": True,
    "require_confirmation": True,
    "backup_before_changes": True,
    "max_bulk_operations": 50,
    "rate_limit_requests": True,
}
