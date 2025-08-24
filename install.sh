#!/bin/bash
# Installation script for SCHF Auto Colinear project

set -e  # Exit on any error

echo "=== SCHF Auto Colinear Installation Script ==="

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project root: $PROJECT_ROOT"

# Check if Python 3.11+ is available
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.11 or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$PROJECT_ROOT/.venv"
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$PROJECT_ROOT/.venv/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install the package in development mode
echo "Installing SCHF Auto Colinear in development mode..."
pip install -e .

# Install additional dependencies if needed
echo "Installing additional dependencies..."
pip install -e ".[jupyter]"

echo ""
echo "✅ Installation completed successfully!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source $PROJECT_ROOT/.venv/bin/activate"
echo ""
echo "To run the SCHF calculation:"
echo "  python parallel_scripts/jax_schf_fixed_filling_u_v1_v2_random_basis.py"
echo ""
echo "To install development dependencies:"
echo "  pip install -e .[dev]"

