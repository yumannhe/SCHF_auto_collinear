from setuptools import setup, find_packages

setup(
    name="schf_auto_colinear",
    version="0.1.0",
    description="Self-Consistent Hartree-Fock for auto-colinear systems",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "scipy",
    ],
    # This makes the package editable/developable
    zip_safe=False,
)
