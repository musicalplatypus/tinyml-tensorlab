"""Tests for logging configuration (improvement 3.12).

Verifies that all modules use the ``logging`` module instead of ``print()``
and that logger names follow the ``"root.*"`` convention.
"""

import ast
import logging
import os
import pathlib

import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "tinyml_modelmaker"


# ---------------------------------------------------------------------------
# Regression guard: no print() calls left in source
# ---------------------------------------------------------------------------
class _PrintCallVisitor(ast.NodeVisitor):
    """AST visitor that records ``print(...)`` call locations."""

    def __init__(self):
        self.print_calls: list[tuple[str, int]] = []
        self._current_file: str = ""

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            self.print_calls.append((self._current_file, node.lineno))
        self.generic_visit(node)


def _find_print_calls(root: pathlib.Path) -> list[tuple[str, int]]:
    """Parse every ``.py`` file under *root* and return ``(file, line)`` for
    each ``print()`` call found.

    Skips files that cannot be parsed (syntax errors in stubs, etc.).
    """
    visitor = _PrintCallVisitor()
    for py_file in sorted(root.rglob("*.py")):
        # Skip test files and __pycache__
        rel = py_file.relative_to(root)
        if "__pycache__" in str(rel):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError:
            continue
        visitor._current_file = str(rel)
        visitor.visit(tree)
    return visitor.print_calls


def test_no_print_statements_in_source():
    """No ``print()`` calls should remain in the tinyml_modelmaker package.

    This is a regression guard to ensure new code uses the ``logging``
    module instead of ``print()``.
    """
    print_calls = _find_print_calls(_PACKAGE_ROOT)
    if print_calls:
        details = "\n".join(f"  {f}:{line}" for f, line in print_calls)
        pytest.fail(
            f"Found {len(print_calls)} print() call(s) in source code. "
            f"Use logging instead:\n{details}"
        )


# ---------------------------------------------------------------------------
# Logger naming convention
# ---------------------------------------------------------------------------
def test_loggers_use_root_prefix():
    """Every logger created by tinyml_modelmaker modules should use a name
    that starts with ``'root.'``."""

    # Import the modules that register loggers at import time.
    # conftest.py already mocks heavy deps so these imports should succeed.
    from tinyml_modelmaker import run_tinyml_modelmaker  # noqa: F401
    from tinyml_modelmaker.utils import misc_utils, download_utils  # noqa: F401

    # Collect loggers whose name starts with "root."
    root_loggers = [
        name for name in logging.Logger.manager.loggerDict
        if name.startswith("root.")
    ]
    # We expect at least the loggers we explicitly created
    expected_prefixes = [
        "root.run",
        "root.utils.misc",
        "root.utils.download",
    ]
    for prefix in expected_prefixes:
        assert any(
            name == prefix or name.startswith(prefix + ".")
            for name in root_loggers
        ), f"Expected a logger with name/prefix '{prefix}' but found none"


# ---------------------------------------------------------------------------
# basicConfig placement
# ---------------------------------------------------------------------------
def test_basicconfig_in_entry_point():
    """``run_tinyml_modelmaker.py`` should contain a ``logging.basicConfig``
    call so that logging output is visible when run as ``__main__``."""
    entry_point = _PACKAGE_ROOT / "run_tinyml_modelmaker.py"
    source = entry_point.read_text(encoding="utf-8")
    assert "logging.basicConfig" in source, (
        "run_tinyml_modelmaker.py should call logging.basicConfig() "
        "to configure the root logger"
    )


# ---------------------------------------------------------------------------
# Log level mapping sanity
# ---------------------------------------------------------------------------
def test_error_messages_use_error_level():
    """Spot-check that error-like messages use ``logger.error`` rather than
    ``logger.info`` by inspecting the source of a few key files."""
    files_with_errors = [
        _PACKAGE_ROOT / "run_tinyml_modelmaker.py",
        _PACKAGE_ROOT / "utils" / "download_utils.py",
        _PACKAGE_ROOT / "utils" / "misc_utils.py",
    ]
    for filepath in files_with_errors:
        source = filepath.read_text(encoding="utf-8")
        # These files should have at least one logger.error call
        assert "logger.error" in source, (
            f"{filepath.name} should use logger.error() for error messages"
        )
