"""Tests for the GitHub Action entrypoint."""

import textwrap

from action.run_review import _filter_diff, _matches_any

SAMPLE_DIFF = textwrap.dedent("""\
diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,4 @@
 import os
+import sys

 def main():
diff --git a/package-lock.json b/package-lock.json
--- a/package-lock.json
+++ b/package-lock.json
@@ -1,5 +1,5 @@
 {
-  "version": "1.0.0",
+  "version": "1.0.1",
   "lockfileVersion": 3
 }
diff --git a/docs/README.md b/docs/README.md
--- a/docs/README.md
+++ b/docs/README.md
@@ -1 +1,2 @@
 # Docs
+New content
""")


def test_filter_diff_removes_excluded_files():
    result = _filter_diff(SAMPLE_DIFF, ["*.json"])
    assert "package-lock.json" not in result
    assert "src/main.py" in result
    assert "docs/README.md" in result


def test_filter_diff_supports_directory_glob():
    result = _filter_diff(SAMPLE_DIFF, ["docs/**"])
    assert "docs/README.md" not in result
    assert "src/main.py" in result
    assert "package-lock.json" in result


def test_filter_diff_multiple_patterns():
    result = _filter_diff(SAMPLE_DIFF, ["*.json", "docs/**"])
    assert "package-lock.json" not in result
    assert "docs/README.md" not in result
    assert "src/main.py" in result


def test_filter_diff_no_patterns_keeps_all():
    result = _filter_diff(SAMPLE_DIFF, [])
    assert "src/main.py" in result
    assert "package-lock.json" in result
    assert "docs/README.md" in result


def test_matches_any_basic():
    assert _matches_any("package-lock.json", ["*.json"]) is True
    assert _matches_any("src/main.py", ["*.json"]) is False
    assert _matches_any("docs/guide/setup.md", ["docs/**"]) is True
