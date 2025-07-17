"""Configuration management for JiraScope."""

import os
from dataclasses import dataclass, field
from pathlib import Path

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

    rag_test_queries: list[str] = field(default_factory=list)
    similarity_threshold: float = 0.8

    report_retention_days: int = 30
    cost_tracking: bool = True

    daily_budget: float = 50.0
    monthly_budget: float = 1000.0

    jira_dry_run: bool = True
    sentry_dsn: str | None = None

    # Embedding configuration (consolidated from EMBEDDING_CONFIG)
    embedding_model: str = "text-embedding-bge-large-en-v1.5"
    embedding_dimensions: int = 1024
    embedding_instruction_prefix: str = "Represent this Jira work item for semantic search: "
    embedding_max_tokens: int = 512
    embedding_timeout: float = 30.0

    # Claude configuration (consolidated from CLAUDE_CONFIG)
    claude_max_tokens: int = 4096
    claude_temperature: float = 0.1
    claude_input_cost_per_token: float = 0.000003
    claude_output_cost_per_token: float = 0.000015
    claude_session_budget: float = 10.0

    # Safety configuration (consolidated from SAFETY_CONFIG)
    require_confirmation: bool = True
    backup_before_changes: bool = True
    max_bulk_operations: int = 50
    rate_limit_requests: bool = True

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
            claude_model=os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest"),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            jira_batch_size=int(os.getenv("JIRA_BATCH_SIZE", "100")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.8")),
            report_retention_days=int(os.getenv("REPORT_RETENTION_DAYS", "30")),
            cost_tracking=os.getenv("COST_TRACKING", "true").lower() == "true",
            daily_budget=float(os.getenv("DAILY_BUDGET", "50.0")),
            monthly_budget=float(os.getenv("MONTHLY_BUDGET", "1000.0")),
            jira_dry_run=os.getenv("JIRA_DRY_RUN", "true").lower() == "true",
            sentry_dsn=os.getenv("SENTRY_DSN"),
            # Extended configuration from environment
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-bge-large-en-v1.5"),
            embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1024")),
            embedding_max_tokens=int(os.getenv("EMBEDDING_MAX_TOKENS", "512")),
            embedding_timeout=float(os.getenv("EMBEDDING_TIMEOUT", "30.0")),
            claude_max_tokens=int(os.getenv("CLAUDE_MAX_TOKENS", "4096")),
            claude_temperature=float(os.getenv("CLAUDE_TEMPERATURE", "0.1")),
            claude_input_cost_per_token=float(os.getenv("CLAUDE_INPUT_COST_PER_TOKEN", "0.000003")),
            claude_output_cost_per_token=float(
                os.getenv("CLAUDE_OUTPUT_COST_PER_TOKEN", "0.000015")
            ),
            claude_session_budget=float(os.getenv("CLAUDE_SESSION_BUDGET", "10.0")),
            require_confirmation=os.getenv("REQUIRE_CONFIRMATION", "true").lower() == "true",
            backup_before_changes=os.getenv("BACKUP_BEFORE_CHANGES", "true").lower() == "true",
            max_bulk_operations=int(os.getenv("MAX_BULK_OPERATIONS", "50")),
            rate_limit_requests=os.getenv("RATE_LIMIT_REQUESTS", "true").lower() == "true",
        )

    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path) as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def load(cls, config_path: Path | None = None) -> "Config":
        """Load configuration with fallback priority: file -> env -> defaults."""
        if config_path and config_path.exists():
            return cls.from_file(config_path)

        default_config_path = Path.home() / ".jirascope" / "config.yaml"
        if default_config_path.exists():
            return cls.from_file(default_config_path)

        return cls.from_env()


# Legacy support - will be removed in next version
# These constants are now part of the Config class for better configurability


def get_embedding_config(config: "Config") -> dict:
    """Get embedding configuration from Config object for backward compatibility."""
    return {
        "model": config.embedding_model,
        "api_base": config.lmstudio_endpoint,
        "dimensions": config.embedding_dimensions,
        "batch_size": config.embedding_batch_size,
        "instruction_prefix": config.embedding_instruction_prefix,
        "max_tokens": config.embedding_max_tokens,
        "timeout": config.embedding_timeout,
    }


def get_claude_config(config: "Config") -> dict:
    """Get Claude configuration from Config object for backward compatibility."""
    return {
        "model": config.claude_model,
        "max_tokens": config.claude_max_tokens,
        "temperature": config.claude_temperature,
        "cost_per_token": {
            "input": config.claude_input_cost_per_token,
            "output": config.claude_output_cost_per_token,
        },
        "daily_budget": config.daily_budget,
        "session_budget": config.claude_session_budget,
    }


def get_safety_config(config: "Config") -> dict:
    """Get safety configuration from Config object for backward compatibility."""
    return {
        "jira_dry_run": config.jira_dry_run,
        "require_confirmation": config.require_confirmation,
        "backup_before_changes": config.backup_before_changes,
        "max_bulk_operations": config.max_bulk_operations,
        "rate_limit_requests": config.rate_limit_requests,
    }
