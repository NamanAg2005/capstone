"""
Static code analysis (Section 4.5 step 3 / 10. Tech stack).

AST-based rule checker that rejects disallowed constructs before generated
code goes anywhere near execution. This is the first of three independent
layers bounding what a single generated attempt can do (the others are the
sandbox's runtime policy and its resource limits -- see sandbox.py).
"""
from __future__ import annotations

import ast

DISALLOWED_IMPORTS = {
    "os", "sys", "subprocess", "socket", "shutil", "requests", "urllib",
    "urllib2", "urllib3", "http", "ftplib", "telnetlib", "ctypes",
    "multiprocessing", "threading", "importlib", "pickle", "marshal",
    "pty", "pdb", "code", "inspect", "resource", "signal",
}

DISALLOWED_CALLS = {"eval", "exec", "compile", "__import__", "open", "input", "globals", "vars"}

DISALLOWED_ATTR_NAMES = {
    "__subclasses__", "__globals__", "__builtins__", "__base__", "__bases__",
    "__mro__", "__class__", "__loader__", "__import__", "__getattribute__",
}


class StaticCheckViolation(Exception):
    pass


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in DISALLOWED_IMPORTS:
                self.violations.append(f"disallowed import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        top = (node.module or "").split(".")[0]
        if top in DISALLOWED_IMPORTS:
            self.violations.append(f"disallowed import: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name in DISALLOWED_CALLS:
            self.violations.append(f"disallowed call: {name}()")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in DISALLOWED_ATTR_NAMES:
            self.violations.append(f"disallowed attribute access: {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in ("__builtins__", "__loader__"):
            self.violations.append(f"disallowed name reference: {node.id}")
        self.generic_visit(node)


def check(code: str) -> list[str]:
    """Return a list of violation descriptions; empty list means code passed."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    visitor = _Visitor()
    visitor.visit(tree)
    return visitor.violations


def is_safe(code: str) -> bool:
    return len(check(code)) == 0
