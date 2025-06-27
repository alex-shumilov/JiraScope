#!/bin/bash
# JiraScope Environment Activation Script

echo "🚀 Activating JiraScope environment..."

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed. Please install Poetry first:"
    echo "   curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating from template..."
    cp .env.dist .env
    echo "📝 Please edit .env file with your configuration before using JiraScope."
    echo "   Especially set JIRA_MCP_ENDPOINT to your Jira MCP server URL."
fi

# Install dependencies
echo "📦 Installing dependencies..."
poetry install

# Activate the environment
echo "🔓 Activating environment..."
echo "Run the following command to activate the JiraScope environment:"
echo ""
echo "   source \$(poetry env info --path)/bin/activate"
echo ""
echo "After activation, you can use 'jirascope' command directly."
echo "Test with: jirascope --help" 