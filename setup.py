#!/usr/bin/env python3
"""
Setup script for SCHF Auto Colinear project
"""

from setuptools import setup, find_packages
import os

# Read the README file if it exists
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "SCHF Auto Colinear - Self-Consistent Hartree-Fock calculations for collinear systems"

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Core dependencies for the project
CORE_DEPENDENCIES = [
    'numpy>=2.0.0',
    'jax>=0.7.0',
    'jaxlib>=0.7.0',
    'matplotlib>=3.0.0',
    'scipy>=1.0.0',
]

# Development dependencies
DEV_DEPENDENCIES = [
    'pytest>=7.0.0',
    'pytest-cov>=4.0.0',
    'black>=22.0.0',
    'flake8>=5.0.0',
    'mypy>=1.0.0',
]

# Jupyter dependencies (optional)
JUPYTER_DEPENDENCIES = [
    'jupyter>=1.0.0',
    'ipykernel>=6.0.0',
    'ipywidgets>=8.0.0',
    'notebook>=7.0.0',
]

setup(
    name="schf-auto-colinear",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Self-Consistent Hartree-Fock calculations for collinear systems using JAX",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/SCHF_auto_colinear",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/SCHF_auto_colinear/issues",
        "Source": "https://github.com/yourusername/SCHF_auto_colinear",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'functions_parameters': ['*.npy', '*.py'],
    },
    install_requires=CORE_DEPENDENCIES,
    extras_require={
        'dev': DEV_DEPENDENCIES,
        'jupyter': JUPYTER_DEPENDENCIES,
        'all': CORE_DEPENDENCIES + DEV_DEPENDENCIES + JUPYTER_DEPENDENCIES,
    },
    entry_points={
        'console_scripts': [
            'schf-run=parallel_scripts.jax_schf_fixed_filling_u_v1_v2_random_basis:main',
        ],
    },
    keywords=[
        "physics", "chemistry", "quantum", "hartree-fock", "self-consistent", 
        "jax", "numpy", "scientific-computing", "condensed-matter"
    ],
    zip_safe=False,
)

