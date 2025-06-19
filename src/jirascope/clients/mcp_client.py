"""MCP client for Jira integration."""

import logging
from typing import Any, Dict, List, Optional
import httpx
from ..models import WorkItem
from ..core.config import Config


logger = logging.getLogger(__name__)


class MCPClient:
    """Client for communicating with Jira via MCP protocol."""
    
    def __init__(self, config: Config):
        self.config = config
        self.endpoint = config.jira_mcp_endpoint
        self.session: Optional[httpx.AsyncClient] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_connections=10)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.aclose()
    
    async def get_work_items(
        self, 
        jql: str = "",
        batch_size: Optional[int] = None
    ) -> List[WorkItem]:
        """Fetch work items from Jira using JQL query."""
        if not self.session:
            raise RuntimeError("MCP client not initialized. Use async context manager.")
        
        batch_size = batch_size or self.config.jira_batch_size
        
        try:
            response = await self.session.post(
                f"{self.endpoint}/search",
                json={
                    "jql": jql,
                    "maxResults": batch_size,
                    "expand": ["changelog", "renderedFields"]
                }
            )
            response.raise_for_status()
            
            data = response.json()
            work_items = []
            
            for issue_data in data.get("issues", []):
                work_item = self._parse_work_item(issue_data)
                work_items.append(work_item)
            
            logger.info(f"Retrieved {len(work_items)} work items")
            return work_items
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch work items: {e}")
            raise
    
    async def get_work_item(self, key: str) -> Optional[WorkItem]:
        """Fetch a single work item by key."""
        if not self.session:
            raise RuntimeError("MCP client not initialized. Use async context manager.")
        
        try:
            response = await self.session.get(
                f"{self.endpoint}/issue/{key}",
                params={"expand": "changelog,renderedFields"}
            )
            
            if response.status_code == 404:
                return None
                
            response.raise_for_status()
            issue_data = response.json()
            
            return self._parse_work_item(issue_data)
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch work item {key}: {e}")
            raise
    
    async def update_work_item(
        self, 
        key: str, 
        fields: Dict[str, Any],
        dry_run: Optional[bool] = None
    ) -> bool:
        """Update a work item. Returns True if successful."""
        if not self.session:
            raise RuntimeError("MCP client not initialized. Use async context manager.")
        
        dry_run = dry_run if dry_run is not None else self.config.jira_dry_run
        
        if dry_run:
            logger.info(f"DRY RUN: Would update {key} with fields: {fields}")
            return True
        
        try:
            response = await self.session.put(
                f"{self.endpoint}/issue/{key}",
                json={"fields": fields}
            )
            response.raise_for_status()
            
            logger.info(f"Successfully updated work item {key}")
            return True
            
        except httpx.HTTPError as e:
            logger.error(f"Failed to update work item {key}: {e}")
            return False
    
    def _parse_work_item(self, issue_data: Dict[str, Any]) -> WorkItem:
        """Parse Jira issue data into WorkItem model."""
        fields = issue_data["fields"]
        
        from datetime import datetime as dt
        
        def parse_datetime(date_str):
            if isinstance(date_str, str):
                return dt.fromisoformat(date_str.replace('Z', '+00:00'))
            return date_str or dt.now()
        
        return WorkItem(
            key=issue_data["key"],
            summary=fields.get("summary", ""),
            description=fields.get("description"),
            issue_type=fields.get("issuetype", {}).get("name", ""),
            status=fields.get("status", {}).get("name", ""),
            parent_key=fields.get("parent", {}).get("key") if fields.get("parent") else None,
            epic_key=fields.get("customfield_10014") if fields.get("customfield_10014") else None,
            created=parse_datetime(fields.get("created")),
            updated=parse_datetime(fields.get("updated")),
            assignee=fields.get("assignee", {}).get("displayName") if fields.get("assignee") else None,
            reporter=fields.get("reporter", {}).get("displayName", ""),
            components=[c.get("name", "") for c in fields.get("components", [])],
            labels=fields.get("labels", [])
        )