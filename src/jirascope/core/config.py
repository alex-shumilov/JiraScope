"""Configuration management for JiraScope."""

import os
from dataclasses import dataclass, field
from typing import List, Optional
import yaml
from pathlib import Path


@dataclass
class Config:
    """Core configuration for JiraScope."""
    
    jira_mcp_endpoint: str
    lmstudio_endpoint: str = "http://localhost:1234/v1"
    qdrant_url: str = "http://localhost:6333"
    claude_api_key: str = ""
    claude_model: str = "claude-3-5-sonnet-20241022"
    
    embedding_batch_size: int = 32
    jira_batch_size: int = 100
    
    rag_test_queries: List[str] = field(default_factory=list)
    similarity_threshold: float = 0.8
    
    report_retention_days: int = 30
    cost_tracking: bool = True
    
    jira_dry_run: bool = True
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            jira_mcp_endpoint=os.getenv("JIRA_MCP_ENDPOINT", ""),
            lmstudio_endpoint=os.getenv("LMSTUDIO_ENDPOINT", "http://localhost:1234/v1"),
            qdrant_url=os.getenv("QDRANT_URL", "http://10.16.10.53:6333"),
            claude_api_key=os.getenv("CLAUDE_API_KEY", ""),
            claude_model=os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            jira_batch_size=int(os.getenv("JIRA_BATCH_SIZE", "100")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.8")),
            report_retention_days=int(os.getenv("REPORT_RETENTION_DAYS", "30")),
            cost_tracking=os.getenv("COST_TRACKING", "true").lower() == "true",
            jira_dry_run=os.getenv("JIRA_DRY_RUN", "true").lower() == "true",
        )
    
    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
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
    "model": "BAAI/bge-large-en-v1.5",
    "api_base": "http://localhost:1234/v1",
    "dimensions": 1024,
    "batch_size": 32,
    "instruction_prefix": "Represent this Jira work item for semantic search: ",
    "max_tokens": 512,
    "timeout": 30.0
}

CLAUDE_CONFIG = {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 4096,
    "temperature": 0.1,
    "cost_per_token": {
        "input": 0.000003,
        "output": 0.000015
    },
    "daily_budget": 50.0,
    "session_budget": 10.0
}

SAFETY_CONFIG = {
    "jira_dry_run": True,
    "require_confirmation": True,
    "backup_before_changes": True,
    "max_bulk_operations": 50,
    "rate_limit_requests": True
}