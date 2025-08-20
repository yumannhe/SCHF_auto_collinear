"""
Path setup for interactive environments.
Import this module at the beginning of any script or interactive session
to automatically set up the project path for importing from functions_parameters.
"""
import sys
import os

# Get the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Add project root to Python path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
else:
    # Move project_root to the front of sys.path to ensure it's found first
    sys.path.remove(project_root)
    sys.path.insert(0, project_root)

print(f"✓ Project path set up: {project_root}")
