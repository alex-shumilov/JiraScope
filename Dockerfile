# Multi-stage Dockerfile for JiraScope
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt* pyproject.toml* README.md ./
RUN pip install --no-cache-dir --upgrade pip

# Development stage
FROM base as dev

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio black mypy

# Copy source code
COPY . .

# Install dependencies and package in development mode
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
RUN pip install -e .

# Expose port for development server
EXPOSE 8000

# Default command for development
CMD ["jirascope", "health-check"]

# Production stage
FROM base as prod

# Copy only necessary files
COPY src/ ./src/
COPY pyproject.toml ./

# Install in production mode
RUN pip install --no-cache-dir .

# Create non-root user
RUN groupadd -r jirascope && useradd -r -g jirascope jirascope
RUN chown -R jirascope:jirascope /app
USER jirascope

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD jirascope health-check || exit 1

# Default command
CMD ["jirascope", "health-check"]
