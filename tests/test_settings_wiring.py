"""Every setting the UI can change must be handled by the app.

A switch wired to a key nothing handles looks like it worked -- it flips, it
stays flipped until you reopen the window, and nothing happens. That is a
miserable bug to notice, so it is pinned here by reading both sides.

Both sides are read with ast rather than regex: the keys reach the callback
through three different shapes (a direct call, a positional argument to the
_switch helper, a lambda closing over a loop variable), and a regex that
understood all three would be less trustworthy than the thing it is checking.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
WINDOW = ROOT / "openflow" / "ui" / "main_window.py"
APP = ROOT / "openflow" / "app.py"

# Handled in _apply_setting but sent from somewhere other than the settings
# page, so an "unreachable handler" check must not flag them.
DRIVEN_ELSEWHERE = {"launch_at_login"}


def _is_setting_callback(node: ast.AST) -> bool:
    """True for `self.cb["setting"]`."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "cb"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "setting"
    )


def _is_self_method(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == name
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def emitted_keys() -> set[str]:
    tree = ast.parse(WINDOW.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # self.cb["setting"]("some.key", value)
        if _is_setting_callback(node.func) and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                keys.add(first.value)
        # self._switch(layout, title, subtitle, "some_key", initial)
        elif _is_self_method(node.func, "_switch") and len(node.args) >= 4:
            key = node.args[3]
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
    return keys


def handled_keys() -> set[str]:
    """Keys compared against inside _apply_setting."""
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_apply_setting"):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Compare):
                continue
            if not (isinstance(inner.left, ast.Name) and inner.left.id == "key"):
                continue
            for comparator in inner.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    keys.add(comparator.value)
    return keys


class SettingsWiring(unittest.TestCase):
    def test_the_scrape_finds_both_sides(self):
        # Guards every other test in this file from passing vacuously.
        self.assertGreater(len(emitted_keys()), 5, "found almost no emitted keys")
        self.assertGreater(len(handled_keys()), 5, "found almost no handled keys")

    def test_every_control_reaches_a_handler(self):
        missing = emitted_keys() - handled_keys()
        self.assertFalse(
            missing,
            f"the settings UI emits {sorted(missing)} but app._apply_setting "
            f"handles none of them -- those controls would silently do nothing",
        )

    def test_no_handler_is_unreachable(self):
        orphans = handled_keys() - emitted_keys() - DRIVEN_ELSEWHERE
        self.assertFalse(
            orphans,
            f"_apply_setting handles {sorted(orphans)}, which no control sends "
            f"-- either dead code or a rename that left the UI behind",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
