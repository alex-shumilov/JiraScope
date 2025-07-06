#!/usr/bin/env python3
"""
LMStudio Integration Setup Script

This script helps users set up JiraScope MCP server integration with LMStudio.
It provides guided configuration and validation.
"""

import json
import os
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()


class LMStudioIntegrator:
    """Handles LMStudio integration setup for JiraScope."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_dir = self.project_root / "config"
        self.lmstudio_config_file = self.config_dir / "lmstudio_mcp_config.json"

    def run_setup(self):
        """Run the complete setup process."""
        console.print(
            Panel(
                "[bold blue]JiraScope × LMStudio Integration Setup[/bold blue]\n\n"
                "This script will help you configure LMStudio to work with JiraScope's MCP server.\n"
                "You'll be able to analyze Jira data using natural language through LMStudio.",
                title="🎯 Setup Wizard",
                border_style="blue",
            )
        )

        try:
            # Step 1: Check prerequisites
            if not self._check_prerequisites():
                return False

            # Step 2: Configure environment
            if not self._configure_environment():
                return False

            # Step 3: Generate LMStudio configuration
            if not self._generate_lmstudio_config():
                return False

            # Step 4: Provide integration instructions
            self._show_integration_instructions()

        except KeyboardInterrupt:
            console.print("\n[yellow]Setup cancelled by user[/yellow]")
            return False
        except Exception as e:
            console.print(f"\n[red]Setup failed: {e}[/red]")
            return False
        else:
            console.print("\n[bold green]✅ Setup completed successfully![/bold green]")
            return True

    def _check_prerequisites(self) -> bool:
        """Check that prerequisites are met."""
        console.print("\n[bold blue]🔍 Checking Prerequisites[/bold blue]")

        checks = [
            ("Python 3.11+", self._check_python),
            ("JiraScope installed", self._check_jirascope),
            ("Configuration directory", self._check_config_dir),
        ]

        table = Table(title="Prerequisites Check")
        table.add_column("Requirement", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Details", style="green")

        all_passed = True

        for name, check_func in checks:
            status, details = check_func()
            if status:
                table.add_row(name, "✅ Pass", details)
            else:
                table.add_row(name, "❌ Fail", details)
                all_passed = False

        console.print(table)

        if not all_passed:
            console.print("\n[red]Please resolve the failed prerequisites before continuing.[/red]")

        return all_passed

    def _check_python(self) -> tuple[bool, str]:
        """Check Python version."""
        version = sys.version_info
        if version >= (3, 11):
            return True, f"Python {version.major}.{version.minor}.{version.micro}"
        return False, f"Python {version.major}.{version.minor} (need 3.11+)"

    def _check_jirascope(self) -> tuple[bool, str]:
        """Check if JiraScope is installed."""
        try:
            import jirascope  # noqa: F401
        except ImportError:
            return False, "Run: pip install -e ."
        else:
            return True, "Package found"

    def _check_config_dir(self) -> tuple[bool, str]:
        """Check if config directory exists."""
        if self.config_dir.exists():
            return True, str(self.config_dir)
        return False, "Config directory missing"

    def _configure_environment(self) -> bool:
        """Configure environment variables."""
        console.print("\n[bold blue]⚙️  Environment Configuration[/bold blue]")

        # Check for existing .env file
        env_file = self.project_root / ".env"
        env_vars = {}

        if env_file.exists():
            console.print(f"📁 Found existing .env file: {env_file}")
            with env_file.open() as f:
                for line in f:
                    if "=" in line and not line.startswith("#"):
                        key, value = line.strip().split("=", 1)
                        env_vars[key] = value

        # Required environment variables
        required_vars = {
            "JIRA_MCP_ENDPOINT": "Your Jira MCP endpoint URL",
            "QDRANT_URL": "Qdrant vector database URL (default: http://localhost:6333)",
            "LMSTUDIO_ENDPOINT": "LMStudio API endpoint (default: http://localhost:1234/v1)",
        }

        updated = False
        for var_name, description in required_vars.items():
            current_value = env_vars.get(var_name, "")

            if not current_value and var_name == "JIRA_MCP_ENDPOINT":
                console.print(f"\n[yellow]Required: {var_name}[/yellow]")
                console.print(f"Description: {description}")
                new_value = Prompt.ask(f"Enter {var_name}")
                env_vars[var_name] = new_value
                updated = True
            elif not current_value:
                # Set defaults for optional vars
                defaults = {
                    "QDRANT_URL": "http://localhost:6333",
                    "LMSTUDIO_ENDPOINT": "http://localhost:1234/v1",
                }
                if var_name in defaults:
                    use_default = Confirm.ask(
                        f"Use default for {var_name} ({defaults[var_name]})?", default=True
                    )
                    if use_default:
                        env_vars[var_name] = defaults[var_name]
                        updated = True

        # Write updated .env file
        if updated:
            with env_file.open("w") as f:
                f.write("# JiraScope Configuration\n")
                for key, value in env_vars.items():
                    f.write(f"{key}={value}\n")
            console.print(f"✅ Updated {env_file}")

        return True

    def _generate_lmstudio_config(self) -> bool:
        """Generate LMStudio MCP configuration."""
        console.print("\n[bold blue]📝 Generating LMStudio Configuration[/bold blue]")

        # Get project root as absolute path
        project_root_abs = self.project_root.resolve()

        # Create the configuration
        config = {
            "mcpServers": {
                "jirascope": {
                    "command": "python",
                    "args": ["-m", "jirascope.mcp_server"],
                    "cwd": str(project_root_abs),
                    "env": {
                        "PYTHONPATH": str(project_root_abs / "src"),
                        "JIRASCOPE_CONFIG": str(project_root_abs / "config" / "jirascope.yaml"),
                    },
                }
            }
        }

        # Write configuration file
        with self.lmstudio_config_file.open("w") as f:
            json.dump(config, f, indent=2)

        console.print(f"✅ Created: {self.lmstudio_config_file}")
        return True

    def _show_integration_instructions(self):
        """Show step-by-step integration instructions."""
        console.print("\n[bold blue]🔗 LMStudio Integration Instructions[/bold blue]")

        instructions = Panel(
            "[bold]Step 1: Open LMStudio[/bold]\n"
            "• Launch LMStudio application\n"
            "• Make sure you have version 0.3.17 or later\n\n"
            "[bold]Step 2: Configure MCP Server[/bold]\n"
            "• Switch to the 'Program' tab in the right sidebar\n"
            "• Click 'Install > Edit mcp.json'\n"
            "• Copy the configuration below and paste it into the file\n\n"
            "[bold]Step 3: Save and Connect[/bold]\n"
            "• Save the mcp.json file (Ctrl+S / Cmd+S)\n"
            "• LMStudio will automatically detect the server\n"
            "• Load a function-calling capable model (e.g., Llama 3, Mixtral)\n\n"
            "[bold]Step 4: Test the Integration[/bold]\n"
            "• Start a new chat\n"
            "• Try: 'Find high priority bugs in frontend components'\n"
            "• LMStudio will ask to confirm tool usage\n\n"
            f"[bold]Configuration file location:[/bold]\n{self.lmstudio_config_file}",
            title="📋 Integration Steps",
            border_style="green",
        )

        console.print(instructions)

        # Show the configuration to copy
        with self.lmstudio_config_file.open() as f:
            config_content = f.read()

        console.print("\n[bold]Configuration to copy into LMStudio's mcp.json:[/bold]")
        console.print(Panel(config_content, title="📋 MCP Configuration", border_style="cyan"))

        # Show usage examples
        console.print("\n[bold blue]💡 Try These Commands in LMStudio[/bold blue]")
        examples = [
            "Find technical debt in authentication components",
            "Analyze scope drift for Epic MOBILE-123",
            "Map dependencies for the frontend team",
            "Show high priority bugs from last week",
        ]

        for i, example in enumerate(examples, 1):
            console.print(f"{i}. [cyan]'{example}'[/cyan]")

    def copy_to_lmstudio(self) -> bool:
        """Attempt to copy configuration directly to LMStudio's config directory."""
        console.print("\n[bold blue]🔄 Attempting automatic LMStudio configuration[/bold blue]")

        # Common LMStudio config locations
        possible_locations = []

        if sys.platform == "darwin":  # macOS
            home = Path.home()
            possible_locations.extend(
                [
                    home / "Library" / "Application Support" / "LMStudio",
                    home / ".lmstudio",
                ]
            )
        elif sys.platform == "win32":  # Windows
            possible_locations.extend(
                [
                    Path(os.environ.get("APPDATA", "")) / "LMStudio",
                    Path(os.environ.get("LOCALAPPDATA", "")) / "LMStudio",
                ]
            )
        else:  # Linux
            home = Path.home()
            possible_locations.extend(
                [
                    home / ".config" / "lmstudio",
                    home / ".lmstudio",
                ]
            )

        # Look for existing LMStudio config
        lmstudio_config_dir = None
        for location in possible_locations:
            if location.exists():
                lmstudio_config_dir = location
                break

        if not lmstudio_config_dir:
            console.print("[yellow]⚠️  Could not find LMStudio config directory[/yellow]")
            console.print("You'll need to manually copy the configuration.")
            return False

        # Check if mcp.json exists
        mcp_json_path = lmstudio_config_dir / "mcp.json"

        if mcp_json_path.exists():
            backup = Confirm.ask(
                f"mcp.json already exists at {mcp_json_path}. Create backup?", default=True
            )
            if backup:
                shutil.copy2(mcp_json_path, f"{mcp_json_path}.backup")
                console.print("✅ Created backup of existing mcp.json")

        # Copy our configuration
        try:
            shutil.copy2(self.lmstudio_config_file, mcp_json_path)
        except Exception as e:
            console.print(f"❌ Failed to copy configuration: {e}")
            return False
        else:
            console.print(f"✅ Configuration copied to: {mcp_json_path}")
            return True


def main():
    """Main entry point."""
    integrator = LMStudioIntegrator()
    success = integrator.run_setup()

    if success:
        console.print(
            "\n[bold green]🎉 Setup complete! You're ready to use JiraScope with LMStudio.[/bold green]"
        )
        console.print("\n💡 Next steps:")
        console.print("1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
        console.print("2. Open LMStudio and follow the integration instructions above")
        console.print("3. Load a function-calling model in LMStudio")
        console.print("4. Try the example commands!")
    else:
        console.print("\n[red]Setup incomplete. Please resolve any issues and try again.[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
