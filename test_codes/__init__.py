# This file makes test_codes a Python package
import sys
import os

# Automatically set up the project path when this package is imported
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
else:
    # Move project_root to the front of sys.path
    sys.path.remove(project_root)
    sys.path.insert(0, project_root)