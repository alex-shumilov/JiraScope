#!/usr/bin/env python3
"""
Local type checking script for JiraScope project.
Runs mypy type checker with proper output formatting.
"""

import subprocess  # nosec B404 - subprocess needed for running type checking tools
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n🔍 {description}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # nosec B603

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode == 0:
            print(f"✅ {description} passed!")
            return True
        print(f"❌ {description} failed!")
        return False

    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Please install the required tools: pip install mypy")
        return False


def main():
    """Run type checking tool."""
    # Change to project root
    project_root = Path(__file__).parent.parent
    subprocess.run(  # nosec B603 B607
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, cwd=project_root, check=False
    )

    print("🐍 JiraScope Type Checking Suite (Python 3.13)")
    print("=" * 60)

    success = True

    # Run mypy type checker
    success &= run_command(["mypy", "src/", "--show-error-codes", "--pretty"], "MyPy Type Checking")

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 Type checking passed!")
        sys.exit(0)

    print("💥 Type checking failed!")
    sys.exit(1)


if __name__ == "__main__":
    main()
