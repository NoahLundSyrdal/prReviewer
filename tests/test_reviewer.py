import json

from pr_reviewer.llm import LLMError
from pr_reviewer.reviewer import PRReviewer

SAMPLE_DIFF = """diff --git a/app/main.py b/app/main.py
index 1111111..2222222 100644
--- a/app/main.py
+++ b/app/main.py
@@ -1,2 +1,4 @@
-def reciprocal(x):
-    return 1 / x
+def reciprocal(x):
+    if x == 0:
+        return 0
+    return 1 / x
"""

BIG_DIFF = """diff --git a/app/a.py b/app/a.py
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


class FakeProvider:
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


class FailingProvider:
    """Provider that raises LLMError to test timeout/failure handling."""

    def __init__(self, error_message: str = "Request timed out") -> None:
        self._error_message = error_message
        self.call_count = 0

    def complete_json(
        self, *, model: str, system_prompt: str, user_prompt: str, json_schema: dict | None = None
    ) -> str:
        self.call_count += 1
        raise LLMError(self._error_message)


def test_multi_pass_review_dedupes_and_annotates_findings() -> None:
    responses = [
        json.dumps(
            {
                "summary": "Correctness review found an edge-case issue.",
                "verdict": "needs attention",
                "findings": [
                    {
                        "severity": "high",
                        "category": "bug",
                        "title": "Potential divide-by-zero semantics regression",
                        "explanation": "Returning 0 for x==0 may mask data quality issues.",
                        "file": "app/main.py",
                        "line": 2,
                        "confidence": 0.73,
                    }
                ],
            }
        ),
        json.dumps(
            {
                "summary": "Security review found no material concerns.",
                "verdict": "looks good",
                "findings": [],
            }
        ),
        json.dumps(
            {
                "summary": "Performance pass repeated a concern and added one extra note.",
                "verdict": "needs attention",
                "findings": [
                    {
                        "severity": "high",
                        "category": "bug",
                        "title": "Potential divide-by-zero semantics regression",
                        "explanation": "This duplicate should be deduped in merged output.",
                        "file": "app/main.py",
                        "line": 2,
                        "confidence": 0.91,
                    },
                    {
                        "severity": "low",
                        "category": "performance",
                        "title": "Minor branching overhead",
                        "explanation": "An extra branch was added in the hot path.",
                        "file": "app/main.py",
                        "line": 3,
                        "confidence": 0.55,
                    },
                ],
            }
        ),
    ]

    reviewer = PRReviewer(FakeProvider(responses))
    result = reviewer.review(diff_text=SAMPLE_DIFF, model="fake", review_mode="multi")

    assert result.review_mode == "multi"
    assert result.passes_run == ["correctness", "security", "performance"]
    assert len(result.findings) == 2

    duplicate_merged = next(f for f in result.findings if "divide-by-zero" in f.title)
    assert duplicate_merged.confidence == 0.91
    assert duplicate_merged.code_frame is not None
    assert duplicate_merged.on_changed_line is True


def test_single_pass_review_splits_large_diffs_into_chunks() -> None:
    responses = [
        json.dumps(
            {
                "summary": "Chunk one found a correctness issue.",
                "verdict": "needs attention",
                "findings": [
                    {
                        "severity": "high",
                        "category": "bug",
                        "title": "None case changes reciprocal behavior",
                        "explanation": "Returning 0 for None hides invalid input.",
                        "file": "app/a.py",
                        "line": 3,
                        "confidence": 0.88,
                    }
                ],
            }
        ),
        json.dumps(
            {
                "summary": "Chunk two found a security issue.",
                "verdict": "needs attention",
                "findings": [
                    {
                        "severity": "high",
                        "category": "security",
                        "title": "Sensitive token is printed",
                        "explanation": "Printing the token leaks secrets into logs.",
                        "file": "app/b.py",
                        "line": 12,
                        "confidence": 0.94,
                    }
                ],
            }
        ),
        json.dumps(
            {
                "summary": "Cross-chunk synthesis found two serious risks spanning correctness and secret handling.",
                "verdict": "high risk",
            }
        ),
    ]

    provider = FakeProvider(responses)
    reviewer = PRReviewer(provider)
    result = reviewer.review(diff_text=BIG_DIFF, model="fake", review_mode="single", max_lines=11)

    assert len(provider.prompts) == 3
    assert "Chunk: 1 of 2" in provider.prompts[0]
    assert "Chunk: 2 of 2" in provider.prompts[1]
    assert "Synthesize the final review outcome for a chunked diff review." in provider.prompts[2]
    assert result.summary == (
        "Cross-chunk synthesis found two serious risks spanning correctness and secret handling."
    )
    assert result.verdict.value == "high risk"
    assert len(result.findings) == 2
    assert any("split the diff into 2 chunk(s)" in warning for warning in result.warnings)


def test_multi_pass_review_handles_chunked_diffs_and_dedupes_results() -> None:
    responses = [
        json.dumps(
            {
                "summary": "Correctness chunk one found an issue.",
                "verdict": "needs attention",
                "findings": [
                    {
                        "severity": "high",
                        "category": "bug",
                        "title": "None case changes reciprocal behavior",
                        "explanation": "Returning 0 for None hides invalid input.",
                        "file": "app/a.py",
                        "line": 3,
                        "confidence": 0.83,
                    }
                ],
            }
        ),
        json.dumps(
            {
                "summary": "Correctness chunk two found no issue.",
                "verdict": "looks good",
                "findings": [],
            }
        ),
        json.dumps(
            {
                "summary": "Security chunk one found no issue.",
                "verdict": "looks good",
                "findings": [],
            }
        ),
        json.dumps(
            {
                "summary": "Security chunk two found a leak.",
                "verdict": "high risk",
                "findings": [
                    {
                        "severity": "high",
                        "category": "security",
                        "title": "Sensitive token is printed",
                        "explanation": "Printing the token leaks secrets into logs.",
                        "file": "app/b.py",
                        "line": 12,
                        "confidence": 0.95,
                    }
                ],
            }
        ),
        json.dumps(
            {
                "summary": "Performance chunk one repeated the concern.",
                "verdict": "needs attention",
                "findings": [
                    {
                        "severity": "high",
                        "category": "bug",
                        "title": "None case changes reciprocal behavior",
                        "explanation": "This duplicate should be deduped across chunks and passes.",
                        "file": "app/a.py",
                        "line": 3,
                        "confidence": 0.91,
                    }
                ],
            }
        ),
        json.dumps(
            {
                "summary": "Performance chunk two found no issue.",
                "verdict": "looks good",
                "findings": [],
            }
        ),
        json.dumps(
            {
                "summary": "Cross-chunk synthesis confirms the patch is high risk because it combines input masking with token leakage.",
                "verdict": "high risk",
            }
        ),
    ]

    provider = FakeProvider(responses)
    reviewer = PRReviewer(provider)
    result = reviewer.review(diff_text=BIG_DIFF, model="fake", review_mode="multi", max_lines=11)

    assert len(provider.prompts) == 7
    assert result.passes_run == ["correctness", "security", "performance"]
    assert "Synthesize the final review outcome for a chunked diff review." in provider.prompts[-1]
    assert result.summary == (
        "Cross-chunk synthesis confirms the patch is high risk because it combines input masking "
        "with token leakage."
    )
    assert len(result.findings) == 2
    assert any(warning == "Deduped 1 overlapping finding(s) across review passes and diff chunks." for warning in result.warnings)
    duplicate_merged = next(f for f in result.findings if "reciprocal behavior" in f.title)
    assert duplicate_merged.confidence == 0.91


def test_chunk_synthesis_falls_back_to_heuristics_when_response_is_invalid() -> None:
    responses = [
        json.dumps(
            {
                "summary": "Chunk one found a correctness issue.",
                "verdict": "needs attention",
                "findings": [
                    {
                        "severity": "high",
                        "category": "bug",
                        "title": "None case changes reciprocal behavior",
                        "explanation": "Returning 0 for None hides invalid input.",
                        "file": "app/a.py",
                        "line": 3,
                        "confidence": 0.88,
                    }
                ],
            }
        ),
        json.dumps(
            {
                "summary": "Chunk two found a security issue.",
                "verdict": "needs attention",
                "findings": [
                    {
                        "severity": "high",
                        "category": "security",
                        "title": "Sensitive token is printed",
                        "explanation": "Printing the token leaks secrets into logs.",
                        "file": "app/b.py",
                        "line": 12,
                        "confidence": 0.94,
                    }
                ],
            }
        ),
        "not json",
    ]

    reviewer = PRReviewer(FakeProvider(responses))
    result = reviewer.review(diff_text=BIG_DIFF, model="fake", review_mode="single", max_lines=11)

    assert result.summary.startswith("Chunked review across 2 diff chunks surfaced 2 unique findings")
    assert result.verdict.value == "high risk"
    assert any("[synthesis] Chunk synthesis returned non-JSON output" in warning for warning in result.warnings)


def test_single_pass_with_empty_diff() -> None:
    """Review with empty diff should produce a result with no findings."""
    responses = [
        json.dumps({
            "summary": "Nothing to review.",
            "verdict": "looks good",
            "findings": [],
        })
    ]

    reviewer = PRReviewer(FakeProvider(responses))
    result = reviewer.review(diff_text="", model="fake", review_mode="single")

    assert result.verdict.value == "looks good"
    assert len(result.findings) == 0


def test_single_pass_multi_chunk_exercises_fixed_indentation_bug_path() -> None:
    """Exercises the multi-chunk single-pass path that was broken by the indentation bug."""
    responses = [
        json.dumps({
            "summary": "Chunk one looks good.",
            "verdict": "looks good",
            "findings": [],
        }),
        json.dumps({
            "summary": "Chunk two has a note.",
            "verdict": "needs attention",
            "findings": [
                {
                    "severity": "low",
                    "category": "maintainability",
                    "title": "Consider adding a docstring",
                    "explanation": "The function serialize lacks documentation.",
                    "file": "app/b.py",
                    "line": 10,
                    "confidence": 0.60,
                }
            ],
        }),
        json.dumps({
            "summary": "Overall the PR has minor documentation gaps.",
            "verdict": "needs attention",
        }),
    ]

    provider = FakeProvider(responses)
    reviewer = PRReviewer(provider)
    result = reviewer.review(diff_text=BIG_DIFF, model="fake", review_mode="single", max_lines=11)

    # Verify the synthesis path was used (3 prompts: 2 chunks + 1 synthesis)
    assert len(provider.prompts) == 3
    assert result.verdict.value == "needs attention"
    assert result.summary == "Overall the PR has minor documentation gaps."


def test_llm_failure_surfaces_warning_instead_of_crash() -> None:
    """When the LLM fails (e.g., timeout), _run_pass catches it and returns a warning."""
    provider = FailingProvider("Request timed out")
    reviewer = PRReviewer(provider)
    result = reviewer.review(diff_text=SAMPLE_DIFF, model="fake", review_mode="single")

    # Should not crash — fallback finding should be generated
    assert result.verdict.value == "needs attention"
    assert any("Could not parse model response" in f.title for f in result.findings)
    assert provider.call_count == 1


def test_multi_pass_partial_failure() -> None:
    """If one pass fails but others succeed, results from successful passes are used."""

    class MixedProvider:
        def __init__(self):
            self.call_count = 0

        def complete_json(self, *, model, system_prompt, user_prompt, json_schema=None):
            self.call_count += 1
            if self.call_count == 1:
                # First pass succeeds
                return json.dumps({
                    "summary": "Correctness found an issue.",
                    "verdict": "needs attention",
                    "findings": [{
                        "severity": "medium",
                        "category": "bug",
                        "title": "Possible null dereference",
                        "explanation": "x could be None when x==0 returns 0.",
                        "file": "app/main.py",
                        "line": 2,
                        "confidence": 0.80,
                    }],
                })
            if self.call_count == 2:
                # Second pass fails
                raise LLMError("Rate limited")
            # Third pass succeeds
            return json.dumps({
                "summary": "No performance issues.",
                "verdict": "looks good",
                "findings": [],
            })

    reviewer = PRReviewer(MixedProvider())
    result = reviewer.review(diff_text=SAMPLE_DIFF, model="fake", review_mode="multi")

    # Should still return results from successful passes
    assert len(result.findings) >= 1
    assert "correctness" in result.passes_run
    assert "performance" in result.passes_run


def test_review_with_file_context_included_in_prompt(sample_diff: str) -> None:
    """File context should appear in the user_prompt passed to the LLM."""
    captured_prompts: list[str] = []

    class CapturingProvider:
        def complete_json(self, *, model: str, system_prompt: str, user_prompt: str, json_schema=None) -> str:
            captured_prompts.append(user_prompt)
            return '{"summary": "ok", "verdict": "looks good", "findings": []}'

    reviewer = PRReviewer(CapturingProvider())
    # sample_diff touches app/main.py — use that as the context key so it matches stats.files
    file_context = {"app/main.py": "def foo():\n    return 42\n"}
    reviewer.review(
        diff_text=sample_diff,
        model="gpt-4.1-mini",
        file_context=file_context,
    )

    assert captured_prompts, "No LLM calls were made"
    assert "app/main.py" in captured_prompts[0], "File context not injected into prompt"
    assert "def foo():" in captured_prompts[0], "File content not in prompt"
    assert "File context" in captured_prompts[0], "Context label not in prompt"
