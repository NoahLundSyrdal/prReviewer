from __future__ import annotations

import json

import pytest

from pr_reviewer.models import (
    Category,
    DiffStats,
    ReviewFinding,
    ReviewResult,
    Severity,
    Verdict,
)

SAMPLE_DIFF = """diff --git a/app/main.py b/app/main.py
index 1111111..2222222 100644
--- a/app/main.py
+++ b/app/main.py
@@ -1,1 +1,1 @@
-print("old")
+print("new")
"""

MULTI_FILE_DIFF = """diff --git a/app/a.py b/app/a.py
index 1111111..2222222 100644
--- a/app/a.py
+++ b/app/a.py
@@ -1,2 +1,4 @@
-def reciprocal(x):
-    return 1 / x
+def reciprocal(x):
+    if x is None:
+        return 0
+    return 1 / x
diff --git a/app/b.py b/app/b.py
index 3333333..4444444 100644
--- a/app/b.py
+++ b/app/b.py
@@ -10,2 +10,4 @@
-def serialize(user):
-    return user["id"]
+def serialize(user):
+    token = user["token"]
+    print(token)
+    return user["id"]
"""


@pytest.fixture()
def sample_diff():
    return SAMPLE_DIFF


@pytest.fixture()
def multi_file_diff():
    return MULTI_FILE_DIFF


@pytest.fixture()
def sample_finding():
    return ReviewFinding(
        severity=Severity.high,
        category=Category.bug,
        title="Potential None access",
        explanation="The updated path dereferences `user` without guarding after a branch.",
        file="app/main.py",
        line=1,
        confidence=0.90,
        suggested_fix="Add a guard clause before dereferencing user fields.",
    )


@pytest.fixture()
def sample_result(sample_finding):
    return ReviewResult(
        summary="Looks mostly fine.",
        verdict=Verdict.needs_attention,
        findings=[sample_finding],
        model="fake-model",
        diff=DiffStats(files=["app/main.py"], files_changed=1, additions=1, deletions=1, line_count=6),
    )


@pytest.fixture()
def empty_result():
    return ReviewResult(
        summary="No issues found.",
        verdict=Verdict.looks_good,
        findings=[],
        model="fake-model",
        diff=DiffStats(files=["app/main.py"], files_changed=1, additions=1, deletions=1, line_count=6),
    )


class FakeProvider:
    """Configurable fake LLM provider for tests."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0
        self.prompts: list[str] = []

    def complete_json(
        self, *, model: str, system_prompt: str, user_prompt: str, json_schema: dict | None = None
    ) -> str:
        self.prompts.append(user_prompt)
        response = self._responses[self._idx]
        self._idx += 1
        return response


def make_llm_response(*, summary: str, verdict: str, findings: list[dict] | None = None) -> str:
    return json.dumps({
        "summary": summary,
        "verdict": verdict,
        "findings": findings or [],
    })
