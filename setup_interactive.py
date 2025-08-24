#!/usr/bin/env python3
"""
Setup script for interactive use (Jupyter, IPython, etc.)
Run this in your interactive session to set up the environment.
"""

import sys
import os

# Get the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Add the project root to Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"✅ Added {project_root} to Python path")

# Change to the project root directory
os.chdir(project_root)
print(f"✅ Changed working directory to: {os.getcwd()}")

# Test imports
try:
    from functions_parameters.jax_schf_helpers import *
    print("✅ functions_parameters.jax_schf_helpers imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")

try:
    from functions_parameters.jax_schf_kernel import schf_fixed_filling_prallel_u_v, schf_single_job
    print("✅ functions_parameters.jax_schf_kernel imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")

try:
    from functions_parameters.universal_parameters import a, b
    print("✅ functions_parameters.universal_parameters imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")

try:
    from functions_parameters.bond_table_generator import build_buckets_per_shell
    print("✅ functions_parameters.bond_table_generator imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")

# Test file access
try:
    import numpy as np
    data = np.load('functions_parameters/random_basis_arr.npy')
    print(f"✅ random_basis_arr.npy loaded successfully, shape: {data.shape}")
except Exception as e:
    print(f"❌ Failed to load random_basis_arr.npy: {e}")

print("\n🎉 Environment setup complete! You can now run your SCHF calculations.")
print("Available functions:")
print("- schf_fixed_filling_prallel_u_v")
print("- schf_single_job")
print("- precompute_k_phase_tables")
print("- hk_all_k_from_phases")
print("- prepare_reference_state")
print("- build_buckets_per_shell")

