import importlib
from pathlib import Path

import pytest

# Find project src directory
SRC_DIR = Path(__file__).resolve().parents[1] / "src"


def get_all_python_modules():
    """Discover all importable python modules inside src/."""
    modules = []
    for py_file in SRC_DIR.rglob("*.py"):
        # Skip __pycache__, temporary/hidden files, or local exploratory files
        if (
            "__pycache__" in py_file.parts
            or py_file.name.startswith(".")
            or py_file.name == "exploring.py"
        ):
            continue
        rel_path = py_file.relative_to(SRC_DIR)
        parts = list(rel_path.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            continue
        module_name = ".".join(parts)
        modules.append(module_name)
    return sorted(set(modules))


ALL_MODULES = get_all_python_modules()


@pytest.mark.parametrize("module_name", ALL_MODULES)
def test_module_import_smoke(module_name):
    """
    Smoke test to guarantee that every Python module across the codebase
    imports successfully without syntax errors, missing dependencies,
    or broken relative/package imports.
    """
    # Attempt import
    mod = importlib.import_module(module_name)
    assert mod is not None, f"Module '{module_name}' imported as None"
