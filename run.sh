#!/bin/bash
# Run script for the Data Science AI Agent

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if using uv (recommended)
if command -v uv &> /dev/null; then
    echo "Running with uv..."
    uv run python -m src.data_scientist_ai_agent "$@"
else
    # Fallback to regular python
    echo "Running with python..."
    python -m src.data_scientist_ai_agent "$@"
fi

