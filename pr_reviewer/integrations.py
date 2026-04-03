from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import requests

from . import __version__
from .parsing import normalize_diff_path, parse_unified_diff, resolve_diff_line

if TYPE_CHECKING:
    from .models import ReviewFinding, ReviewResult

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 30
_REQUEST_TIMEOUT = (_CONNECT_TIMEOUT, _READ_TIMEOUT)


class IntegrationError(RuntimeError):
    pass


@dataclass
class PostingReport:
    platform: str
    attempted: int = 0
    posted: int = 0
    skipped: int = 0
    fallback_posted: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class CommentTarget:
    file_path: str
    old_path: str
    new_path: str
    old_line: int | None
    new_line: int | None
    line_type: str


def post_findings(
    *,
    platform: str,
    result: ReviewResult,
    diff_text: str,
    repo: str | None,
    pr_number: int | None,
    mr_iid: int | None,
    token: str | None,
    base_url: str | None,
    dry_run: bool = False,
) -> PostingReport:
    parsed_diff = parse_unified_diff(diff_text)

    if platform == "github":
        return _post_to_github(
            result=result,
            parsed_diff=parsed_diff,
            repo=repo,
            pr_number=pr_number,
            token=token or os.getenv("GITHUB_TOKEN"),
            base_url=base_url or "https://api.github.com",
            dry_run=dry_run,
        )

    if platform == "gitlab":
        return _post_to_gitlab(
            result=result,
            parsed_diff=parsed_diff,
            repo=repo,
            mr_iid=mr_iid,
            token=token or os.getenv("GITLAB_TOKEN"),
            base_url=base_url or "https://gitlab.com/api/v4",
            dry_run=dry_run,
        )

    raise IntegrationError(f"Unsupported platform: {platform}")


def _post_to_github(
    *,
    result: ReviewResult,
    parsed_diff,
    repo: str | None,
    pr_number: int | None,
    token: str | None,
    base_url: str,
    dry_run: bool,
) -> PostingReport:
    if not repo:
        raise IntegrationError("GitHub posting requires --repo owner/name")
    if not pr_number:
        raise IntegrationError("GitHub posting requires --pr <number>")

    report = PostingReport(platform="github")
    postable_findings, skipped_before_post = _collect_postable_findings(result, parsed_diff)
    report.skipped += skipped_before_post

    if dry_run:
        for _ in postable_findings:
            report.attempted += 1
            report.posted += 1
        logger.info("GitHub dry-run: %d comment(s) would be posted", report.posted)
        return report

    if not token:
        raise IntegrationError("Missing GitHub token. Set --integration-token or GITHUB_TOKEN.")

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": f"pr-reviewer/{__version__}",
        }
    )

    pr_url = f"{base_url.rstrip('/')}/repos/{repo}/pulls/{pr_number}"
    pr_response = session.get(pr_url, timeout=_REQUEST_TIMEOUT)
    if pr_response.status_code >= 400:
        raise IntegrationError(f"GitHub PR lookup failed ({pr_response.status_code}): {pr_response.text}")

    pr_payload = pr_response.json()
    head_sha = pr_payload.get("head", {}).get("sha")
    if not head_sha:
        raise IntegrationError("Could not resolve GitHub PR head SHA.")

    postable_with_status: list[tuple[ReviewFinding, CommentTarget, bool]] = []
    for finding, target in postable_findings:
        report.attempted += 1

        payload = _build_github_comment_payload(
            finding=finding,
            target=target,
            head_sha=head_sha,
        )
        comment_url = f"{base_url.rstrip('/')}/repos/{repo}/pulls/{pr_number}/comments"
        response = session.post(comment_url, json=payload, timeout=_REQUEST_TIMEOUT)

        if response.status_code in {200, 201}:
            report.posted += 1
            postable_with_status.append((finding, target, False))
            continue

        if response.status_code == 422:
            report.skipped += 1
            report.errors.append(
                f"Skipped {finding.file}:{finding.line} ({finding.title}): "
                "inline position rejected, will try fallback."
            )
            postable_with_status.append((finding, target, True))
            continue

        report.errors.append(
            f"GitHub comment failed for {finding.file}:{finding.line} ({finding.title}) "
            f"[{response.status_code}]."
        )
        postable_with_status.append((finding, target, False))

    # Post fallback summary for 422-rejected findings
    fallback_findings = [(f, t) for f, t, had_422 in postable_with_status if had_422]
    if fallback_findings:
        body = _build_fallback_summary_body(fallback_findings)
        issues_url = f"{base_url.rstrip('/')}/repos/{repo}/issues/{pr_number}/comments"
        fb_response = session.post(issues_url, json={"body": body}, timeout=_REQUEST_TIMEOUT)
        if fb_response.status_code in {200, 201}:
            report.fallback_posted += len(fallback_findings)
            report.posted += len(fallback_findings)
            report.skipped -= len(fallback_findings)
            # Clean up the "will try fallback" error messages
            report.errors = [e for e in report.errors if "will try fallback" not in e]
        else:
            report.errors.append(f"Fallback summary comment failed [{fb_response.status_code}].")

    logger.info("GitHub posting complete: %d/%d posted, %d skipped", report.posted, report.attempted, report.skipped)
    return report


def _post_to_gitlab(
    *,
    result: ReviewResult,
    parsed_diff,
    repo: str | None,
    mr_iid: int | None,
    token: str | None,
    base_url: str,
    dry_run: bool,
) -> PostingReport:
    if not repo:
        raise IntegrationError("GitLab posting requires --repo <project-id-or-path>")
    if not mr_iid:
        raise IntegrationError("GitLab posting requires --mr <iid>")

    report = PostingReport(platform="gitlab")
    postable_findings, skipped_before_post = _collect_postable_findings(result, parsed_diff)
    report.skipped += skipped_before_post

    if dry_run:
        for _ in postable_findings:
            report.attempted += 1
            report.posted += 1
        logger.info("GitLab dry-run: %d comment(s) would be posted", report.posted)
        return report

    if not token:
        raise IntegrationError("Missing GitLab token. Set --integration-token or GITLAB_TOKEN.")

    project_ref = requests.utils.quote(repo, safe="")
    root = base_url.rstrip("/")

    session = requests.Session()
    session.headers.update(
        {
            "PRIVATE-TOKEN": token,
            "Content-Type": "application/json",
            "User-Agent": f"pr-reviewer/{__version__}",
        }
    )

    versions_url = f"{root}/projects/{project_ref}/merge_requests/{mr_iid}/versions"
    versions_response = session.get(versions_url, timeout=_REQUEST_TIMEOUT)
    if versions_response.status_code >= 400:
        raise IntegrationError(
            f"GitLab MR versions lookup failed ({versions_response.status_code}): {versions_response.text}"
        )

    versions = versions_response.json()
    if not versions:
        raise IntegrationError("GitLab MR has no available diff versions for inline comments.")

    latest = versions[0]
    base_sha = latest.get("base_commit_sha")
    start_sha = latest.get("start_commit_sha")
    head_sha = latest.get("head_commit_sha")
    if not (base_sha and start_sha and head_sha):
        raise IntegrationError("Could not resolve GitLab MR diff version SHAs.")

    discussions_url = f"{root}/projects/{project_ref}/merge_requests/{mr_iid}/discussions"

    for finding, target in postable_findings:
        report.attempted += 1

        payload = _build_gitlab_comment_payload(
            finding=finding,
            target=target,
            base_sha=base_sha,
            start_sha=start_sha,
            head_sha=head_sha,
        )
        response = session.post(discussions_url, json=payload, timeout=_REQUEST_TIMEOUT)

        if response.status_code in {200, 201}:
            report.posted += 1
            continue

        if response.status_code in {400, 422}:
            report.skipped += 1
            report.errors.append(
                f"Skipped {finding.file}:{finding.line} ({finding.title}): GitLab rejected comment position."
            )
            continue

        report.errors.append(
            f"GitLab comment failed for {finding.file}:{finding.line} ({finding.title}) "
            f"[{response.status_code}]."
        )

    logger.info("GitLab posting complete: %d/%d posted, %d skipped", report.posted, report.attempted, report.skipped)
    return report


def _collect_postable_findings(
    result: ReviewResult,
    parsed_diff,
) -> tuple[list[tuple[ReviewFinding, CommentTarget]], int]:
    postable: list[tuple[ReviewFinding, CommentTarget]] = []
    skipped = 0

    for finding in result.findings:
        if not finding.file or not finding.line:
            skipped += 1
            continue

        target = _resolve_comment_target(finding, parsed_diff)
        if target is None:
            skipped += 1
            continue

        postable.append((finding, target))

    return postable, skipped


def _resolve_comment_target(
    finding: ReviewFinding,
    parsed_diff,
) -> CommentTarget | None:
    if not finding.file or not finding.line:
        return None

    resolved = resolve_diff_line(
        parsed_diff,
        file_path=finding.file,
        line=finding.line,
    )
    if resolved is None:
        return None

    diff_file = resolved.diff_file
    primary_path = diff_file.new_path or diff_file.old_path or diff_file.display_path
    old_path = diff_file.old_path or primary_path
    new_path = diff_file.new_path or primary_path
    return CommentTarget(
        file_path=primary_path,
        old_path=old_path,
        new_path=new_path,
        old_line=resolved.hunk_line.old_line,
        new_line=resolved.hunk_line.new_line,
        line_type=resolved.hunk_line.line_type,
    )


def _build_github_comment_payload(
    *,
    finding: ReviewFinding,
    target: CommentTarget,
    head_sha: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "body": _build_comment_body(finding),
        "path": normalize_diff_path(target.file_path),
        "commit_id": head_sha,
    }

    if target.line_type == "del":
        payload["line"] = target.old_line
        payload["side"] = "LEFT"
        return payload

    payload["line"] = target.new_line or target.old_line
    payload["side"] = "RIGHT"
    return payload


def _build_gitlab_comment_payload(
    *,
    finding: ReviewFinding,
    target: CommentTarget,
    base_sha: str,
    start_sha: str,
    head_sha: str,
) -> dict[str, object]:
    position: dict[str, object] = {
        "position_type": "text",
        "base_sha": base_sha,
        "start_sha": start_sha,
        "head_sha": head_sha,
        "old_path": normalize_diff_path(target.old_path),
        "new_path": normalize_diff_path(target.new_path),
    }

    if target.line_type == "del":
        position["old_line"] = target.old_line
    elif target.line_type == "add":
        position["new_line"] = target.new_line
    else:
        if target.old_line is not None:
            position["old_line"] = target.old_line
        if target.new_line is not None:
            position["new_line"] = target.new_line

    return {
        "body": _build_comment_body(finding),
        "position": position,
    }


def _build_fallback_summary_body(
    findings_with_targets: list[tuple[ReviewFinding, CommentTarget]],
) -> str:
    lines = ["## PR Review — Findings (could not post as inline comments)\n"]
    lines.append(
        "The following findings could not be posted as inline comments"
        " (the diff position may have changed). They are summarized here instead.\n"
    )
    for finding, target in findings_with_targets:
        location = (
            f"`{target.file_path}:{finding.line}`"
            if finding.line
            else f"`{target.file_path}`"
        )
        lines.append(
            f"### [{finding.severity.value.upper()}][{finding.category.value}] {finding.title}\n"
            f"**Location**: {location}  \n"
            f"**Confidence**: {finding.confidence:.2f}  \n"
            f"**Why it matters**: {finding.explanation}\n"
        )
        if finding.suggested_fix:
            lines.append(f"**Suggested fix**: {finding.suggested_fix}\n")
        lines.append("")
    return "\n".join(lines)


def _build_comment_body(finding: ReviewFinding) -> str:
    lines = [
        f"**[{finding.severity.value.upper()}][{finding.category.value}] {finding.title}**",
        f"Confidence: {finding.confidence:.2f}",
        "",
        f"Why it matters: {finding.explanation}",
    ]

    if finding.suggested_fix:
        lines.append(f"Suggested fix: {finding.suggested_fix}")

    if finding.code_frame:
        lines.extend(["", "Code frame:", "```text", finding.code_frame, "```"])

    return "\n".join(lines)
