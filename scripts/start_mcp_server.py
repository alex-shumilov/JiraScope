#!/usr/bin/env python3
"""
JiraScope MCP Server Startup Script

This script provides automated startup for the JiraScope MCP server with:
- Environment validation
- Health checks for dependencies
- Graceful shutdown handling
- Configuration validation
"""

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Optional

import httpx
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from jirascope.core.config import Config
from jirascope.utils.logging import StructuredLogger

console = Console()
logger = StructuredLogger(__name__)


class MCPServerManager:
    """Manages the lifecycle of the JiraScope MCP server."""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config: Optional[Config] = None
        self.server_process = None
        self.shutdown_requested = False

    async def validate_environment(self) -> bool:
        """Validate that all required dependencies are available."""
        console.print("[bold blue]🔍 Validating environment...[/bold blue]")

        checks = [
            ("Python version", self._check_python_version),
            ("Required packages", self._check_packages),
            ("Configuration", self._check_configuration),
            ("LMStudio", self._check_lmstudio),
            ("Qdrant", self._check_qdrant),
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            for check_name, check_func in checks:
                task = progress.add_task(f"Checking {check_name}...", total=None)
                success = await check_func()
                if success:
                    console.print(f"✅ {check_name}")
                else:
                    console.print(f"❌ {check_name} - Failed")
                    return False
                progress.remove_task(task)

        return True

    async def _check_python_version(self) -> bool:
        """Check Python version compatibility."""
        try:
            version = sys.version_info
            return version >= (3, 11)
        except Exception:
            return False

    async def _check_packages(self) -> bool:
        """Check that required packages are installed."""
        required_packages = [
            "qdrant_client",
            "httpx",
            "pydantic",
            "mcp",
            "fastapi",
            "rich",
        ]

        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                console.print(f"❌ Missing package: {package}")
                return False
        return True

    async def _check_configuration(self) -> bool:
        """Validate configuration file and environment variables."""
        try:
            if self.config_file and Path(self.config_file).exists():
                # Load from file
                with open(self.config_file, "r") as f:
                    yaml.safe_load(f)
                console.print(f"📁 Using config file: {self.config_file}")
            else:
                # Check environment variables
                if not os.getenv("JIRA_MCP_ENDPOINT"):
                    console.print("❌ JIRA_MCP_ENDPOINT environment variable required")
                    return False

            # Try to load config
            self.config = Config.from_env()
            return True
        except Exception as e:
            console.print(f"❌ Configuration error: {e}")
            return False

    async def _check_lmstudio(self) -> bool:
        """Check LMStudio availability."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                endpoint = (
                    self.config.lmstudio_endpoint if self.config else "http://localhost:1234/v1"
                )
                response = await client.get(f"{endpoint}/models")
                return response.status_code == 200
        except Exception:
            console.print("⚠️  LMStudio not running (optional for MCP server)")
            return True  # Non-critical for MCP server operation

    async def _check_qdrant(self) -> bool:
        """Check Qdrant availability."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                qdrant_url = self.config.qdrant_url if self.config else "http://localhost:6333"
                response = await client.get(f"{qdrant_url}/collections")
                return response.status_code == 200
        except Exception:
            console.print("⚠️  Qdrant not running (required for vector search)")
            return False

    async def start_server(self) -> bool:
        """Start the JiraScope MCP server."""
        try:
            console.print("[bold green]🚀 Starting JiraScope MCP Server...[/bold green]")

            # Set environment variables
            if self.config_file:
                os.environ["JIRASCOPE_CONFIG_FILE"] = str(self.config_file)

            # Import and start the server
            from jirascope.mcp_server.server import main

            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            console.print(
                Panel(
                    "[bold green]JiraScope MCP Server is running![/bold green]\n\n"
                    "📊 Available tools:\n"
                    "  • search_jira_issues - Natural language search\n"
                    "  • analyze_technical_debt - Find tech debt patterns\n"
                    "  • detect_scope_drift - Analyze Epic scope changes\n"
                    "  • map_dependencies - Dependency mapping\n\n"
                    "🔗 Resources:\n"
                    "  • jira://config - Server configuration\n\n"
                    "💡 Prompts:\n"
                    "  • jira_analysis_prompt - Analysis templates\n"
                    "  • sprint_planning_prompt - Sprint planning\n\n"
                    "[yellow]Press Ctrl+C to stop the server[/yellow]",
                    title="🎯 JiraScope MCP Server",
                    border_style="green",
                )
            )

            # Run the MCP server
            await main()
            return True

        except KeyboardInterrupt:
            console.print("\n[yellow]🛑 Shutdown requested by user[/yellow]")
            return True
        except Exception as e:
            console.print(f"[red]❌ Failed to start server: {e}[/red]")
            logger.error(f"Server startup failed: {e}")
            return False

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        console.print(f"\n[yellow]🛑 Received signal {signum}, shutting down...[/yellow]")
        self.shutdown_requested = True


async def main():
    """Main entry point for the MCP server startup script."""
    import argparse

    parser = argparse.ArgumentParser(description="Start JiraScope MCP Server")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration file (default: use environment variables)",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip environment validation checks",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    # Create server manager
    manager = MCPServerManager(config_file=args.config)

    # Welcome message
    console.print(
        Panel(
            "[bold blue]JiraScope MCP Server[/bold blue]\n"
            "AI-powered Jira analysis through Model Context Protocol\n\n"
            "🔧 Validating environment and starting server...",
            title="🎯 JiraScope",
            border_style="blue",
        )
    )

    try:
        # Validate environment
        if not args.skip_checks:
            if not await manager.validate_environment():
                console.print("\n[red]❌ Environment validation failed[/red]")
                console.print("\n💡 Tips:")
                console.print("  • Ensure all dependencies are installed: pip install -e .")
                console.print("  • Set JIRA_MCP_ENDPOINT environment variable")
                console.print("  • Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
                console.print("  • Start LMStudio (optional)")
                sys.exit(1)

        # Start server
        success = await manager.start_server()
        if not success:
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/yellow]")
    except Exception as e:
        console.print(f"\n[red]💥 Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
