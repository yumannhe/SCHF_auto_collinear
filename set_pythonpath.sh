#!/bin/bash
# Add this project to PYTHONPATH permanently
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "Added $(pwd) to PYTHONPATH"
echo "Add this line to your ~/.bashrc or ~/.zshrc for permanent setup:"
echo "export PYTHONPATH=\"\${PYTHONPATH}:$(pwd)\""
