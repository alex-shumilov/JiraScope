#!/usr/bin/env python3
"""
Test LMStudio Integration

This script validates that the JiraScope MCP server can be started
and is properly configured for LMStudio integration.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

console = Console()


class IntegrationTester:
    """Tests the LMStudio integration setup."""

    def __init__(self):
        self.project_root = project_root
        self.config_dir = self.project_root / "config"
        self.results = {}

    async def run_tests(self) -> bool:
        """Run all integration tests."""
        console.print(
            Panel(
                "[bold blue]JiraScope LMStudio Integration Test[/bold blue]\n\n"
                "This script validates that your setup is ready for LMStudio integration.",
                title="🧪 Integration Test",
                border_style="blue",
            )
        )

        tests = [
            ("Configuration Files", self._test_config_files),
            ("Environment Variables", self._test_environment),
            ("Python Dependencies", self._test_dependencies),
            ("MCP Server Import", self._test_mcp_import),
            ("Server Startup", self._test_server_startup),
        ]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            for test_name, test_func in tests:
                task = progress.add_task(f"Testing {test_name}...", total=None)

                try:
                    result = await test_func()
                    self.results[test_name] = result
                    status = "✅ Pass" if result["success"] else "❌ Fail"
                    console.print(f"{status} {test_name}: {result['message']}")
                except Exception as e:
                    self.results[test_name] = {"success": False, "message": str(e)}
                    console.print(f"❌ Fail {test_name}: {e}")

                progress.remove_task(task)

        # Show summary
        self._show_summary()

        # Return overall success
        return all(r["success"] for r in self.results.values())

    async def _test_config_files(self) -> dict[str, Any]:
        """Test that configuration files exist and are valid."""
        try:
            # Check for LMStudio config
            lmstudio_config = self.config_dir / "lmstudio_mcp_config.json"
            if not lmstudio_config.exists():
                return {
                    "success": False,
                    "message": "lmstudio_mcp_config.json not found. Run setup script.",
                }

            # Validate JSON
            with lmstudio_config.open() as f:
                config = json.load(f)

            if "mcpServers" not in config:
                return {"success": False, "message": "Invalid LMStudio config format"}

            if "jirascope" not in config["mcpServers"]:
                return {"success": False, "message": "JiraScope server not configured"}

            # Check for YAML config
            yaml_config = self.config_dir / "jirascope.yaml"
            if yaml_config.exists():
                return {"success": True, "message": "All config files present and valid"}

            return {"success": True, "message": "LMStudio config valid, YAML optional"}

        except Exception as e:
            return {"success": False, "message": f"Config validation failed: {e}"}

    async def _test_environment(self) -> dict[str, Any]:
        """Test environment variable configuration."""
        try:
            # Check for required variables
            jira_endpoint = os.getenv("JIRA_MCP_ENDPOINT")
            if not jira_endpoint:
                return {
                    "success": False,
                    "message": "JIRA_MCP_ENDPOINT not set. Check your .env file.",
                }

            # Check optional variables with defaults
            os.getenv("QDRANT_URL", "http://localhost:6333")
            os.getenv("LMSTUDIO_ENDPOINT", "http://localhost:1234/v1")

            return {
                "success": True,
                "message": f"Environment configured (Jira: {len(jira_endpoint)} chars)",
            }

        except Exception as e:
            return {"success": False, "message": f"Environment check failed: {e}"}

    async def _test_dependencies(self) -> dict[str, Any]:
        """Test that required Python packages are available."""
        try:
            required_packages = [
                "mcp",
                "httpx",
                "qdrant_client",
                "pydantic",
                "rich",
                "fastapi",
                "yaml",
            ]

            missing = []
            for package in required_packages:
                try:
                    __import__(package.replace("-", "_"))
                except ImportError:
                    missing.append(package)

            if missing:
                return {"success": False, "message": f"Missing packages: {', '.join(missing)}"}

            return {"success": True, "message": f"All {len(required_packages)} packages available"}

        except Exception as e:
            return {"success": False, "message": f"Dependency check failed: {e}"}

    async def _test_mcp_import(self) -> dict[str, Any]:
        """Test that the MCP server can be imported."""
        try:
            # Try importing the MCP server module
            from jirascope.core.config import Config  # noqa: F401
            from jirascope.mcp_server.server import mcp  # noqa: F401
        except ImportError as e:
            return {"success": False, "message": f"Import failed: {e}"}
        except Exception as e:
            return {"success": False, "message": f"MCP server test failed: {e}"}
        else:
            # Test that we can import the server without errors
            # Note: We can't easily check tool count without inspecting FastMCP internals
            return {"success": True, "message": "MCP server imported successfully"}

    async def _test_server_startup(self) -> dict[str, Any]:
        """Test that the server can start (briefly)."""
        try:
            # Import server components

            # Try initializing components (this tests config loading)
            start_time = time.time()

            # Note: We don't actually start the server to avoid hanging
            # Just test that we can load the configuration
            from jirascope.core.config import Config

            Config.from_env()

            elapsed = time.time() - start_time
        except Exception as e:
            return {"success": False, "message": f"Server startup test failed: {e}"}
        else:
            return {"success": True, "message": f"Server components ready ({elapsed:.2f}s)"}

    def _show_summary(self):
        """Show test results summary."""
        console.print("\n[bold blue]📊 Test Results Summary[/bold blue]")

        table = Table(title="Integration Test Results")
        table.add_column("Test", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Details", style="green")

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r["success"])

        for test_name, result in self.results.items():
            status = "✅ Pass" if result["success"] else "❌ Fail"
            table.add_row(test_name, status, result["message"])

        console.print(table)

        # Overall result
        if passed_tests == total_tests:
            console.print(f"\n[bold green]🎉 All {total_tests} tests passed![/bold green]")
            console.print("\n💡 Next steps:")
            console.print("1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
            console.print("2. Open LMStudio and add the MCP server configuration")
            console.print("3. Load a function-calling model in LMStudio")
            console.print("4. Try: 'Find high priority bugs in frontend components'")
        else:
            console.print(
                f"\n[bold red]❌ {total_tests - passed_tests} of {total_tests} tests failed[/bold red]"
            )
            console.print("\n🔧 To fix issues:")
            console.print("• Run: python scripts/lmstudio_integration_setup.py")
            console.print("• Check your .env file configuration")
            console.print("• Ensure all dependencies are installed: pip install -e .")

        # Show configuration info
        lmstudio_config = self.config_dir / "lmstudio_mcp_config.json"
        if lmstudio_config.exists():
            console.print(f"\n📁 LMStudio config: {lmstudio_config}")
            console.print("📖 Usage guide: docs/examples/lmstudio_prompts.md")


async def main():
    """Main entry point."""
    tester = IntegrationTester()

    try:
        success = await tester.run_tests()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Test cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Test failed with error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
