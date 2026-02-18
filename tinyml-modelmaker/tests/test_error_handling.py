"""Regression guard tests for consistent error handling (improvement 3.9).

These tests use AST analysis to ensure that no assert statements, sys.exit()
calls, or bare string raises creep back into the source code.
"""

import ast
import pathlib

import pytest

from tinyml_modelmaker.utils.config_dict import ConfigDict

SOURCE_ROOT = pathlib.Path(__file__).resolve().parent.parent / 'tinyml_modelmaker'


def _iter_source_files():
    """Yield all .py source files under tinyml_modelmaker/, skipping caches."""
    for f in sorted(SOURCE_ROOT.rglob('*.py')):
        if '__pycache__' in str(f):
            continue
        yield f


class TestNoAssertStatements:
    """Verify that production source code contains no assert statements."""

    def test_no_assert_statements_in_source(self):
        violations = []
        for filepath in _iter_source_files():
            try:
                tree = ast.parse(filepath.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Assert):
                    rel = filepath.relative_to(SOURCE_ROOT.parent)
                    violations.append(f'  {rel}:{node.lineno}')
        assert violations == [], (
            f'Found {len(violations)} assert statement(s) in source code:\n'
            + '\n'.join(violations)
        )


class TestNoSysExit:
    """Verify that production source code does not use sys.exit() for error handling."""

    def test_no_sys_exit_in_source(self):
        violations = []
        for filepath in _iter_source_files():
            try:
                tree = ast.parse(filepath.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    # Match sys.exit(...)
                    if (isinstance(func, ast.Attribute)
                            and func.attr == 'exit'
                            and isinstance(func.value, ast.Name)
                            and func.value.id == 'sys'):
                        rel = filepath.relative_to(SOURCE_ROOT.parent)
                        violations.append(f'  {rel}:{node.lineno}')
        assert violations == [], (
            f'Found {len(violations)} sys.exit() call(s) in source code:\n'
            + '\n'.join(violations)
        )


class TestNoStringRaise:
    """Verify that no 'raise "string"' patterns exist (Python syntax bug)."""

    def test_no_string_raise_in_source(self):
        violations = []
        for filepath in _iter_source_files():
            try:
                tree = ast.parse(filepath.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Raise) and node.exc is not None:
                    # In Python 3, `raise "string"` parses as Raise(exc=Constant(value="string"))
                    if isinstance(node.exc, ast.Constant) and isinstance(node.exc.value, str):
                        rel = filepath.relative_to(SOURCE_ROOT.parent)
                        violations.append(f'  {rel}:{node.lineno}')
        assert violations == [], (
            f'Found {len(violations)} bare string raise(s) in source code:\n'
            + '\n'.join(violations)
        )


class TestConfigDictExceptions:
    """Verify ConfigDict raises proper exception types."""

    def test_invalid_type_raises_typeerror(self):
        with pytest.raises(TypeError, match='expected str, dict, or None'):
            ConfigDict(42)

    def test_non_yaml_file_raises_valueerror(self, tmp_path):
        txt_file = tmp_path / 'config.txt'
        txt_file.write_text('key: value')
        with pytest.raises(ValueError, match='unrecognized file type'):
            ConfigDict(str(txt_file))
