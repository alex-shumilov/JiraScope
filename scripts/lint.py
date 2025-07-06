#!/usr/bin/env python3
"""
Local linting script for JiraScope project.
Runs ruff linter, formatter, and mypy type checker with proper output formatting.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str, fix_mode: bool = False) -> bool:
    """Run a command and return True if successful."""
    print(f"\n🔍 {description}")
    print("=" * 60)
    
    try:
        if fix_mode and "ruff check" in " ".join(cmd):
            cmd.append("--fix")
            
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
            
        if result.returncode == 0:
            print(f"✅ {description} passed!")
            return True
        else:
            print(f"❌ {description} failed!")
            return False
            
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        print("Please install the required tools: pip install ruff mypy")
        return False


def main():
    """Run all linting tools."""
    # Change to project root
    project_root = Path(__file__).parent.parent
    subprocess.run(["git", "rev-parse", "--show-toplevel"], 
                  capture_output=True, cwd=project_root)
    
    print("🐍 JiraScope Linting Suite (Python 3.13)")
    print("=" * 60)
    
    # Check if we're in fix mode
    fix_mode = "--fix" in sys.argv
    if fix_mode:
        print("🔧 Running in FIX mode - auto-fixing issues where possible")
    
    success = True
    
    # Run ruff linter
    success &= run_command(
        ["ruff", "check", ".", "--show-fixes"],
        "Ruff Linting",
        fix_mode
    )
    
    # Run ruff formatter
    format_cmd = ["ruff", "format"]
    if not fix_mode:
        format_cmd.append("--check")
    format_cmd.append(".")
    
    success &= run_command(
        format_cmd,
        "Ruff Formatting" if fix_mode else "Ruff Format Check"
    )
    
    # Run mypy type checker
    success &= run_command(
        ["mypy", "src/", "--show-error-codes", "--pretty"],
        "MyPy Type Checking"
    )
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 All linting checks passed!")
        sys.exit(0)
    else:
        print("💥 Some linting checks failed!")
        if not fix_mode:
            print("💡 Try running with --fix to auto-fix issues: python scripts/lint.py --fix")
        sys.exit(1)


if __name__ == "__main__":
    main() 