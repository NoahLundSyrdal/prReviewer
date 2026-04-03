from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher

from pydantic import ValidationError

from .llm import LLMError, LLMProvider
from .models import (
    Category,
    ChunkSynthesisPayload,
    DiffStats,
    LLMReviewPayload,
    ReviewFinding,
    ReviewResult,
    Severity,
    Verdict,
)
from .parsing import build_finding_annotation, chunk_diff, normalize_diff_path, parse_unified_diff

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert senior software engineer performing PR review on a unified diff.

Rules:
- Base every claim only on the provided diff text.
- Do NOT reference files, symbols, or behavior not visible in the diff.
- Prioritize correctness, security, performance, and maintainability.
- Be direct and specific. Avoid fluff.
- If there are no meaningful issues, return findings as an empty list.
- Return strict JSON only. No markdown, no code fences, no extra keys.

JSON schema:
{
  "summary": "short executive summary",
  "verdict": "looks good" | "needs attention" | "high risk",
  "findings": [
    {
      "severity": "low" | "medium" | "high",
      "category": "bug" | "security" | "performance" | "maintainability",
      "title": "brief title",
      "explanation": "why this matters",
      "file": "optional file path",
      "line": 123,
      "confidence": 0.0,
      "suggested_fix": "optional practical fix"
    }
  ]
}

Confidence guidance:
- 0.90-1.00: very likely issue based on explicit evidence in the diff
- 0.70-0.89: likely issue with minor uncertainty
- 0.40-0.69: plausible risk, less certain
"""

SYNTHESIS_SYSTEM_PROMPT = """You are an expert senior software engineer consolidating chunked PR review results.

Rules:
- Base every claim only on the provided chunk summaries and findings.
- Do not invent new file paths, lines, or issues that are not already represented.
- Focus on producing the strongest overall executive summary and verdict.
- Return strict JSON only. No markdown, no code fences, no extra keys.

JSON schema:
{
  "summary": "short executive summary",
  "verdict": "looks good" | "needs attention" | "high risk"
}
"""

REVIEW_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "verdict": {"type": "string", "enum": ["looks good", "needs attention", "high risk"]},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "category": {"type": "string", "enum": ["bug", "security", "performance", "maintainability"]},
                    "title": {"type": "string"},
                    "explanation": {"type": "string"},
                    "file": {"type": ["string", "null"]},
                    "line": {"type": ["integer", "null"]},
                    "confidence": {"type": "number"},
                    "suggested_fix": {"type": ["string", "null"]},
                },
                "required": ["severity", "category", "title", "explanation", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "verdict", "findings"],
    "additionalProperties": False,
}

SYNTHESIS_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "verdict": {"type": "string", "enum": ["looks good", "needs attention", "high risk"]},
    },
    "required": ["summary", "verdict"],
    "additionalProperties": False,
}

MULTI_PASS_FOCI: list[tuple[str, str]] = [
    (
        "correctness",
        "Focus on correctness and reliability: bugs, edge cases, broken invariants, and API contract changes.",
    ),
    (
        "security",
        "Focus on security risk: injection, auth/session mistakes, secret leakage, crypto misuse, and permission issues.",
    ),
    (
        "performance",
        "Focus on performance and scalability: N+1 queries, expensive loops, memory pressure, blocking operations.",
    ),
]

_SEVERITY_RANK = {
    Severity.low: 1,
    Severity.medium: 2,
    Severity.high: 3,
}
_VERDICT_RANK = {
    Verdict.looks_good: 1,
    Verdict.needs_attention: 2,
    Verdict.high_risk: 3,
}


class PRReviewer:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def review(
        self,
        *,
        diff_text: str,
        model: str,
        max_lines: int = 1200,
        review_mode: str = "single",
    ) -> ReviewResult:
        if review_mode not in {"single", "multi"}:
            raise ValueError("review_mode must be 'single' or 'multi'")

        parsed_diff = parse_unified_diff(diff_text)
        stats = parsed_diff.stats

        diff_chunks, was_chunked, original_line_count = chunk_diff(diff_text, max_lines=max_lines)
        if was_chunked:
            stats.original_line_count = original_line_count

        warnings: list[str] = []
        if was_chunked:
            warnings.append(
                f"Diff exceeded max lines ({max_lines}); review split the diff into "
                f"{len(diff_chunks)} chunk(s) to preserve more context."
            )
        if not stats.patch_like:
            warnings.append("Input does not appear to be a standard unified diff; confidence may be lower.")

        if review_mode == "multi":
            return self._review_multi(
                diff_chunks=diff_chunks,
                full_parsed_diff=parsed_diff,
                stats=stats,
                model=model,
                warnings=warnings,
            )

        return self._review_single(
            diff_chunks=diff_chunks,
            full_parsed_diff=parsed_diff,
            stats=stats,
            model=model,
            warnings=warnings,
        )

    def _review_single(
        self,
        *,
        diff_chunks,
        full_parsed_diff,
        stats: DiffStats,
        model: str,
        warnings: list[str],
    ) -> ReviewResult:
        payloads: list[LLMReviewPayload] = []
        chunk_reviews: list[dict[str, object]] = []
        raw_failures: list[str] = []
        chunk_count = len(diff_chunks)
        focus = (
            "Use a balanced review across correctness, security, performance, and maintainability. "
            "Prefer high-signal findings."
        )

        for chunk_index, diff_chunk in enumerate(diff_chunks, start=1):
            payload, raw_response, parse_warning = self._run_pass(
                model=model,
                pass_name="general",
                focus=focus,
                diff_text=diff_chunk.diff_text,
                stats=diff_chunk.stats,
                full_stats=stats,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
            )

            if parse_warning:
                warnings.append(
                    _format_chunk_warning(
                        parse_warning,
                        pass_name="general",
                        chunk_index=chunk_index,
                        chunk_count=chunk_count,
                    )
                )

            if payload is None:
                raw_failures.append(
                    _tag_raw_response(
                        raw_response,
                        pass_name="general",
                        chunk_index=chunk_index,
                        chunk_count=chunk_count,
                    )
                )
                continue

            payloads.append(payload)
            chunk_reviews.append(
                {
                    "pass_name": "general",
                    "chunk_index": chunk_index,
                    "payload": payload,
                }
            )

        if not payloads:
            fallback_finding = ReviewFinding(
                severity=Severity.low,
                category=Category.maintainability,
                title="Could not parse model response",
                explanation=(
                    "The LLM response was not valid JSON matching the expected schema. "
                    "Re-run the review or try a different model/provider."
                ),
                confidence=0.95,
                suggested_fix="Re-run with --format json to inspect output, then retry with a more reliable model.",
            )
            return ReviewResult(
                summary="Review generation failed to return structured output.",
                verdict=Verdict.needs_attention,
                findings=[fallback_finding],
                model=model,
                diff=stats,
                review_mode="single",
                passes_run=["general"],
                warnings=warnings,
                raw_response="\n\n".join(raw_failures) if raw_failures else None,
            )

        all_findings: list[ReviewFinding] = []
        for payload in payloads:
            all_findings.extend(payload.findings)

        merged_findings = _dedupe_findings(all_findings)
        deduped_count = len(all_findings) - len(merged_findings)
        if chunk_count > 1 and deduped_count > 0:
            warnings.append(f"Deduped {deduped_count} overlapping finding(s) across diff chunks.")

        findings = self._annotate_findings(merged_findings, full_parsed_diff)
        missing_context = _count_unmapped_findings(findings)
        if missing_context:
            warnings.append(
                f"Could not attach hunk context for {missing_context} finding(s); file/line may not map to visible hunks."
            )

        sorted_findings = _sort_findings(findings)
        if chunk_count == 1 and len(payloads) == 1:
            summary = payloads[0].summary
            verdict = payloads[0].verdict
        else:
            summary = _build_single_summary(sorted_findings, chunk_count=chunk_count)
            verdict = _infer_verdict(sorted_findings, [payload.verdict for payload in payloads])
            if chunk_count > 1:
                summary, verdict, synthesis_warning = self._synthesize_chunked_review(
                    model=model,
                    review_mode="single",
                    stats=stats,
                    findings=sorted_findings,
                    summary=summary,
                    verdict=verdict,
                    chunk_reviews=chunk_reviews,
                )
                if synthesis_warning:
                    warnings.append(synthesis_warning)

        return ReviewResult(
            summary=summary,
            verdict=verdict,
            findings=sorted_findings,
            model=model,
            diff=stats,
            review_mode="single",
            passes_run=["general"],
            warnings=warnings,
        )

    def _review_multi(
        self,
        *,
        diff_chunks,
        full_parsed_diff,
        stats: DiffStats,
        model: str,
        warnings: list[str],
    ) -> ReviewResult:
        payloads: list[tuple[str, LLMReviewPayload]] = []
        chunk_reviews: list[dict[str, object]] = []
        raw_failures: list[str] = []
        chunk_count = len(diff_chunks)
        successful_passes: list[str] = []

        for pass_name, focus in MULTI_PASS_FOCI:
            pass_had_success = False

            for chunk_index, diff_chunk in enumerate(diff_chunks, start=1):
                payload, raw_response, parse_warning = self._run_pass(
                    model=model,
                    pass_name=pass_name,
                    focus=focus,
                    diff_text=diff_chunk.diff_text,
                    stats=diff_chunk.stats,
                    full_stats=stats,
                    chunk_index=chunk_index,
                    chunk_count=chunk_count,
                )

                if parse_warning:
                    warnings.append(
                        _format_chunk_warning(
                            parse_warning,
                            pass_name=pass_name,
                            chunk_index=chunk_index,
                            chunk_count=chunk_count,
                        )
                    )

                if payload is None:
                    raw_failures.append(
                        _tag_raw_response(
                            raw_response,
                            pass_name=pass_name,
                            chunk_index=chunk_index,
                            chunk_count=chunk_count,
                        )
                    )
                    continue

                payloads.append((pass_name, payload))
                chunk_reviews.append(
                    {
                        "pass_name": pass_name,
                        "chunk_index": chunk_index,
                        "payload": payload,
                    }
                )
                pass_had_success = True

            if pass_had_success:
                successful_passes.append(pass_name)

        if not payloads:
            fallback_finding = ReviewFinding(
                severity=Severity.low,
                category=Category.maintainability,
                title="Could not parse model responses",
                explanation=(
                    "All review passes returned malformed output. "
                    "Re-run with a more reliable model/provider."
                ),
                confidence=0.95,
                suggested_fix="Try a different model and inspect raw provider output with --format json.",
            )
            return ReviewResult(
                summary="Multi-pass review failed to produce structured output.",
                verdict=Verdict.needs_attention,
                findings=[fallback_finding],
                model=model,
                diff=stats,
                review_mode="multi",
                passes_run=[name for name, _ in MULTI_PASS_FOCI],
                warnings=warnings,
                raw_response="\n\n".join(raw_failures) if raw_failures else None,
            )

        all_findings: list[ReviewFinding] = []
        for _, payload in payloads:
            all_findings.extend(payload.findings)

        merged_findings = _dedupe_findings(all_findings)
        deduped_count = len(all_findings) - len(merged_findings)
        if deduped_count > 0:
            scope = "review passes and diff chunks" if chunk_count > 1 else "review passes"
            warnings.append(f"Deduped {deduped_count} overlapping finding(s) across {scope}.")

        annotated_findings = self._annotate_findings(merged_findings, full_parsed_diff)
        missing_context = _count_unmapped_findings(annotated_findings)
        if missing_context:
            warnings.append(
                f"Could not attach hunk context for {missing_context} finding(s); file/line may not map to visible hunks."
            )

        pass_verdicts = [payload.verdict for _, payload in payloads]
        sorted_findings = _sort_findings(annotated_findings)
        verdict = _infer_verdict(sorted_findings, pass_verdicts)
        summary = _build_multi_summary(payloads, sorted_findings, chunk_count=chunk_count)
        if chunk_count > 1:
            summary, verdict, synthesis_warning = self._synthesize_chunked_review(
                model=model,
                review_mode="multi",
                stats=stats,
                findings=sorted_findings,
                summary=summary,
                verdict=verdict,
                chunk_reviews=chunk_reviews,
            )
            if synthesis_warning:
                warnings.append(synthesis_warning)

        return ReviewResult(
            summary=summary,
            verdict=verdict,
            findings=sorted_findings,
            model=model,
            diff=stats,
            review_mode="multi",
            passes_run=successful_passes,
            warnings=warnings,
        )

    def _run_pass(
        self,
        *,
        model: str,
        pass_name: str,
        focus: str,
        diff_text: str,
        stats: DiffStats,
        full_stats: DiffStats,
        chunk_index: int,
        chunk_count: int,
    ) -> tuple[LLMReviewPayload | None, str, str | None]:
        logger.debug(
            "Running pass=%s model=%s chunk=%d/%d lines=%d",
            pass_name,
            model,
            chunk_index,
            chunk_count,
            stats.line_count,
        )
        user_prompt = self._build_user_prompt(
            diff_text=diff_text,
            stats=stats,
            full_stats=full_stats,
            pass_name=pass_name,
            focus=focus,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
        )

        try:
            raw_response = self.provider.complete_json(
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                json_schema=REVIEW_JSON_SCHEMA,
            )
        except LLMError as exc:
            logger.warning("LLM call failed for pass=%s chunk=%d/%d: %s", pass_name, chunk_index, chunk_count, exc)
            return None, str(exc), f"LLM call failed: {exc}"

        payload, parse_warning = _parse_llm_payload(raw_response)
        return payload, raw_response, parse_warning

    @staticmethod
    def _build_user_prompt(
        *,
        diff_text: str,
        stats: DiffStats,
        full_stats: DiffStats,
        pass_name: str,
        focus: str,
        chunk_index: int,
        chunk_count: int,
    ) -> str:
        files_block = "\n".join(f"- {name}" for name in stats.files[:50])
        if not files_block:
            files_block = "- (none detected)"

        scope_block = ""
        if chunk_count > 1:
            scope_block = (
                "Chunk metadata:\n"
                f"- Chunk: {chunk_index} of {chunk_count}\n"
                f"- Full diff files changed: {full_stats.files_changed}\n"
                f"- Full diff additions: {full_stats.additions}\n"
                f"- Full diff deletions: {full_stats.deletions}\n"
                f"- Full diff visible lines: {full_stats.line_count}\n"
                f"- Chunk files changed: {stats.files_changed}\n"
                f"- Chunk visible diff lines: {stats.line_count}\n\n"
            )

        return (
            "Review this unified diff and return JSON using the required schema.\n\n"
            f"Review pass: {pass_name}\n"
            f"Focus: {focus}\n\n"
            f"{scope_block}"
            "Diff metadata:\n"
            f"- Files changed: {stats.files_changed}\n"
            f"- Additions: {stats.additions}\n"
            f"- Deletions: {stats.deletions}\n"
            f"- Visible diff lines: {stats.line_count}\n"
            "- Changed files:\n"
            f"{files_block}\n\n"
            "Constraints:\n"
            "- Only comment on visible code in the diff.\n"
            "- Keep findings actionable and concise.\n"
            "- Avoid duplicate findings; include only meaningful issues for this pass.\n"
            "- If this is one chunk of a larger diff, do not speculate about code outside this chunk.\n"
            "- If uncertain, lower confidence instead of overstating.\n\n"
            "DIFF_START\n"
            f"{diff_text}\n"
            "DIFF_END"
        )

    @staticmethod
    def _annotate_findings(findings: list[ReviewFinding], parsed_diff) -> list[ReviewFinding]:
        for finding in findings:
            if finding.file:
                finding.file = normalize_diff_path(finding.file)

            if not finding.file or not finding.line:
                continue

            hunk_header, code_frame, on_changed_line = build_finding_annotation(
                parsed_diff,
                file_path=finding.file,
                line=finding.line,
            )
            finding.hunk_header = hunk_header
            finding.code_frame = code_frame
            finding.on_changed_line = on_changed_line

        return findings

    def _synthesize_chunked_review(
        self,
        *,
        model: str,
        review_mode: str,
        stats: DiffStats,
        findings: list[ReviewFinding],
        summary: str,
        verdict: Verdict,
        chunk_reviews: list[dict[str, object]],
    ) -> tuple[str, Verdict, str | None]:
        user_prompt = _build_synthesis_prompt(
            review_mode=review_mode,
            stats=stats,
            findings=findings,
            summary=summary,
            verdict=verdict,
            chunk_reviews=chunk_reviews,
        )

        raw_response = self.provider.complete_json(
            model=model,
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_schema=SYNTHESIS_JSON_SCHEMA,
        )

        payload, parse_warning = _parse_synthesis_payload(raw_response)
        if payload is None:
            return summary, verdict, f"[synthesis] {parse_warning}"

        return payload.summary, payload.verdict, None


def _parse_llm_payload(raw_text: str) -> tuple[LLMReviewPayload | None, str | None]:
    direct_warning: str | None = None

    text = raw_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1).strip()

    candidate_json = _extract_json_blob(text)
    if candidate_json is None:
        return None, "Model returned non-JSON output; showing fallback result."

    try:
        parsed = json.loads(candidate_json)
    except json.JSONDecodeError:
        return None, "Model returned malformed JSON; showing fallback result."

    try:
        payload = LLMReviewPayload.model_validate(parsed)
    except ValidationError as exc:
        direct_warning = (
            "Model JSON did not match expected schema; showing fallback result. "
            f"Validation error: {exc.errors()[0].get('msg', 'unknown error')}."
        )
        return None, direct_warning

    return payload, direct_warning


def _parse_synthesis_payload(raw_text: str) -> tuple[ChunkSynthesisPayload | None, str]:
    text = raw_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1).strip()

    candidate_json = _extract_json_blob(text)
    if candidate_json is None:
        return None, "Chunk synthesis returned non-JSON output; using heuristic summary and verdict."

    try:
        parsed = json.loads(candidate_json)
    except json.JSONDecodeError:
        return None, "Chunk synthesis returned malformed JSON; using heuristic summary and verdict."

    try:
        payload = ChunkSynthesisPayload.model_validate(parsed)
    except ValidationError as exc:
        return (
            None,
            "Chunk synthesis JSON did not match the expected schema; using heuristic summary and verdict. "
            f"Validation error: {exc.errors()[0].get('msg', 'unknown error')}.",
        )

    return payload, ""


def _extract_json_blob(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    decoder = json.JSONDecoder()
    for idx, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed_obj, end = decoder.raw_decode(stripped[idx:])
            if isinstance(parsed_obj, dict):
                return stripped[idx : idx + end]
        except json.JSONDecodeError:
            continue
    return None


def _build_synthesis_prompt(
    *,
    review_mode: str,
    stats: DiffStats,
    findings: list[ReviewFinding],
    summary: str,
    verdict: Verdict,
    chunk_reviews: list[dict[str, object]],
) -> str:
    files_block = "\n".join(f"- {name}" for name in stats.files[:50]) or "- (none detected)"

    chunk_blocks: list[str] = []
    for review in chunk_reviews:
        payload = review["payload"]
        if not isinstance(payload, LLMReviewPayload):
            continue

        pass_name = str(review["pass_name"])
        chunk_index = int(review["chunk_index"])
        titles = "; ".join(finding.title for finding in payload.findings[:3]) or "(none)"
        chunk_blocks.append(
            f"- Pass: {pass_name}\n"
            f"  Chunk: {chunk_index}\n"
            f"  Verdict: {payload.verdict.value}\n"
            f"  Summary: {payload.summary}\n"
            f"  Top finding titles: {titles}"
        )

    if not chunk_blocks:
        chunk_blocks.append("- (none)")

    finding_blocks: list[str] = []
    for finding in findings[:25]:
        location = ""
        if finding.file and finding.line:
            location = f" ({finding.file}:{finding.line})"
        elif finding.file:
            location = f" ({finding.file})"
        finding_blocks.append(
            f"- [{finding.severity.value.upper()}][{finding.category.value}] {finding.title}{location} "
            f"(confidence: {finding.confidence:.2f})"
        )

    if not finding_blocks:
        finding_blocks.append("- (none)")

    return (
        "Synthesize the final review outcome for a chunked diff review.\n\n"
        f"Review mode: {review_mode}\n"
        f"Files changed: {stats.files_changed}\n"
        f"Additions: {stats.additions}\n"
        f"Deletions: {stats.deletions}\n"
        f"Visible diff lines: {stats.line_count}\n"
        f"Original diff lines: {stats.original_line_count or stats.line_count}\n"
        "Changed files:\n"
        f"{files_block}\n\n"
        "Current heuristic final result:\n"
        f"- Verdict: {verdict.value}\n"
        f"- Summary: {summary}\n\n"
        "Chunk review outputs:\n"
        f"{chr(10).join(chunk_blocks)}\n\n"
        "Merged findings:\n"
        f"{chr(10).join(finding_blocks)}\n\n"
        "Return the best final summary and verdict only. Do not add new findings."
    )


def _dedupe_findings(findings: list[ReviewFinding]) -> list[ReviewFinding]:
    ordered = sorted(
        findings,
        key=lambda finding: (
            _SEVERITY_RANK[finding.severity],
            finding.confidence,
            finding.file or "",
            finding.line or 0,
        ),
        reverse=True,
    )

    merged: list[ReviewFinding] = []
    for candidate in ordered:
        duplicate = next((existing for existing in merged if _is_duplicate(existing, candidate)), None)
        if duplicate is None:
            merged.append(candidate.model_copy(deep=True))
            continue
        _merge_finding(duplicate, candidate)

    return merged


def _is_duplicate(left: ReviewFinding, right: ReviewFinding) -> bool:
    if left.category != right.category:
        return False

    left_file = normalize_diff_path(left.file) if left.file else ""
    right_file = normalize_diff_path(right.file) if right.file else ""
    if left_file and right_file and left_file != right_file:
        return False

    if left.line and right.line and abs(left.line - right.line) > 1:
        return False

    title_ratio = _similarity(_normalize_text(left.title), _normalize_text(right.title))
    explanation_ratio = _similarity(
        _normalize_text(left.explanation),
        _normalize_text(right.explanation),
    )

    if left_file and right_file and left.line and right.line and title_ratio >= 0.55:
        return True

    return title_ratio >= 0.84 or (title_ratio >= 0.70 and explanation_ratio >= 0.72)


def _merge_finding(target: ReviewFinding, candidate: ReviewFinding) -> None:
    if _SEVERITY_RANK[candidate.severity] > _SEVERITY_RANK[target.severity]:
        target.severity = candidate.severity

    target.confidence = max(target.confidence, candidate.confidence)

    if not target.file and candidate.file:
        target.file = normalize_diff_path(candidate.file)
    if not target.line and candidate.line:
        target.line = candidate.line

    if (not target.suggested_fix) and candidate.suggested_fix:
        target.suggested_fix = candidate.suggested_fix

    if len(candidate.explanation) > len(target.explanation):
        target.explanation = candidate.explanation


def _sort_findings(findings: list[ReviewFinding]) -> list[ReviewFinding]:
    return sorted(
        findings,
        key=lambda finding: (
            _SEVERITY_RANK[finding.severity],
            finding.confidence,
            finding.file or "",
            finding.line or 0,
            finding.title.lower(),
        ),
        reverse=True,
    )


def _build_multi_summary(
    payloads: list[tuple[str, LLMReviewPayload]],
    findings: list[ReviewFinding],
    *,
    chunk_count: int = 1,
) -> str:
    passes = ", ".join(_unique_pass_names(payloads))
    if not findings:
        if chunk_count > 1:
            return (
                f"Chunked multi-pass review across {chunk_count} diff chunks ({passes}) "
                "found no material issues in the visible diff."
            )
        return f"Multi-pass review ({passes}) found no material issues in the visible diff."

    high = sum(1 for finding in findings if finding.severity == Severity.high)
    medium = sum(1 for finding in findings if finding.severity == Severity.medium)
    low = sum(1 for finding in findings if finding.severity == Severity.low)

    top_titles = "; ".join(finding.title for finding in _sort_findings(findings)[:3])
    if chunk_count > 1:
        return (
            f"Chunked multi-pass review across {chunk_count} diff chunks ({passes}) surfaced "
            f"{len(findings)} unique findings (high: {high}, medium: {medium}, low: {low}). "
            f"Top risks: {top_titles}."
        )
    return (
        f"Multi-pass review ({passes}) surfaced {len(findings)} unique findings "
        f"(high: {high}, medium: {medium}, low: {low}). "
        f"Top risks: {top_titles}."
    )


def _build_single_summary(
    findings: list[ReviewFinding],
    *,
    chunk_count: int,
) -> str:
    if not findings:
        return (
            f"Chunked review across {chunk_count} diff chunks found no material issues "
            "in the visible diff."
        )

    high = sum(1 for finding in findings if finding.severity == Severity.high)
    medium = sum(1 for finding in findings if finding.severity == Severity.medium)
    low = sum(1 for finding in findings if finding.severity == Severity.low)

    top_titles = "; ".join(finding.title for finding in findings[:3])
    return (
        f"Chunked review across {chunk_count} diff chunks surfaced {len(findings)} unique findings "
        f"(high: {high}, medium: {medium}, low: {low}). "
        f"Top risks: {top_titles}."
    )


def _infer_verdict(findings: list[ReviewFinding], pass_verdicts: list[Verdict]) -> Verdict:
    if not findings:
        return max(pass_verdicts, key=lambda verdict: _VERDICT_RANK[verdict], default=Verdict.looks_good)

    high = sum(1 for finding in findings if finding.severity == Severity.high)
    medium = sum(1 for finding in findings if finding.severity == Severity.medium)

    if high > 0:
        return Verdict.high_risk
    if medium > 0 or findings:
        return Verdict.needs_attention
    return Verdict.looks_good


def _count_unmapped_findings(findings: list[ReviewFinding]) -> int:
    return sum(
        1
        for finding in findings
        if finding.file and finding.line and not finding.code_frame
    )


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _format_chunk_warning(
    warning: str,
    *,
    pass_name: str,
    chunk_index: int,
    chunk_count: int,
) -> str:
    labels: list[str] = []
    if pass_name != "general":
        labels.append(pass_name)
    if chunk_count > 1:
        labels.append(f"chunk {chunk_index}/{chunk_count}")

    if not labels:
        return warning
    return f"[{', '.join(labels)}] {warning}"


def _tag_raw_response(
    raw_response: str,
    *,
    pass_name: str,
    chunk_index: int,
    chunk_count: int,
) -> str:
    labels: list[str] = []
    if pass_name != "general":
        labels.append(pass_name)
    if chunk_count > 1:
        labels.append(f"chunk {chunk_index}/{chunk_count}")
    if not labels:
        return raw_response
    return f"[{', '.join(labels)}]\n{raw_response}"


def _unique_pass_names(payloads: list[tuple[str, LLMReviewPayload]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for pass_name, _ in payloads:
        if pass_name not in seen:
            ordered.append(pass_name)
            seen.add(pass_name)
    return ordered
