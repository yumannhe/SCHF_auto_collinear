# Import Setup Solutions

This document explains how to handle imports for different execution environments.

## ✅ Recommended Solution: Interactive-Compatible Scripts

**What it is**: Scripts that work in all environments (interactive, command-line, module execution).

**How to use**: Add this at the top of your scripts:
```python
# Set up path for interactive environments
try:
    import setup_path
except ImportError:
    # If setup_path not found, we're probably running as module
    pass
```

**Benefits**:
- Works in PyCharm interactive window
- Works in IPython/Jupyter notebooks
- Works with direct script execution
- Works with module execution
- Minimal code overhead

**Example**: 
- `test_codes/interactive_schf_test.py` - Full SCHF test
- `test_codes/interactive_template.py` - Template for new scripts

## Alternative Solutions

### Option 1: Environment Variable (Permanent)
Add to your `~/.bashrc` or `~/.zshrc`:
```bash
export PYTHONPATH="${PYTHONPATH}:/Users/yuzimeimei/PycharmProjects/SCHF_auto_colinear"
```

Or run the provided script:
```bash
./set_pythonpath.sh
```

### Option 2: Package Installation (Already Done)
Your project is installed as a package with:
```bash
pip install -e .
```

This works great with the module approach:
```bash
python -m test_codes.script_name
```

### Option 3: sitecustomize.py (Advanced)
**WARNING**: Affects ALL Python sessions system-wide.
```bash
python create_sitecustomize.py --execute
```

## For New Scripts

1. Copy `test_codes/interactive_template.py`
2. Rename it to your script name
3. Modify the content as needed
4. Works in all environments!

## Execution Methods That Work

**All these methods work with interactive-compatible scripts**:
```bash
# Direct execution (works perfectly)
python test_codes/your_script.py

# Module execution (works perfectly)
python -m test_codes.your_script

# From test_codes directory (works perfectly)
cd test_codes && python your_script.py

# In PyCharm interactive window (works perfectly)
# Just run the script normally
```

## Summary

The **recommended approach** is using interactive-compatible scripts because they:
- Work in all environments (interactive, command-line, module)
- Require minimal setup code
- Are reliable and predictable
- Support your preferred workflow
- Keep your code portable
