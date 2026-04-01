#!/usr/bin/env python3
"""GitHub Action entrypoint for pr-reviewer."""

from __future__ import annotations

import contextlib
import fnmatch
import os
import re
import subprocess
import sys
import tempfile


def main() -> int:
    pr_number = os.environ.get("PR_NUMBER")
    repo = os.environ.get("REPO")
    token = os.environ.get("GITHUB_TOKEN")
    model = os.environ.get("INPUT_MODEL", "gpt-4.1-mini")
    mode = os.environ.get("INPUT_MODE", "multi")
    max_lines = os.environ.get("INPUT_MAX_LINES", "1200")
    exclude = os.environ.get("INPUT_EXCLUDE", "")
    post_comments = os.environ.get("INPUT_POST_COMMENTS", "true").lower() == "true"

    if not pr_number or not repo:
        print("::error::This action must run on a pull_request event.", file=sys.stderr)
        return 1

    if not os.environ.get("PR_REVIEWER_API_KEY"):
        print("::error::Missing api_key input. Set it to your OpenAI (or compatible) API key.", file=sys.stderr)
        return 1

    # Fetch PR diff via GitHub API
    print(f"Fetching diff for PR #{pr_number} in {repo}...")
    diff_text = _fetch_pr_diff(repo, pr_number, token)
    if not diff_text or not diff_text.strip():
        print("PR diff is empty — nothing to review.")
        return 0

    # Apply exclude patterns
    if exclude:
        patterns = [p.strip() for p in exclude.split(",") if p.strip()]
        diff_text = _filter_diff(diff_text, patterns)
        if not diff_text.strip():
            print("All files matched exclude patterns — nothing to review.")
            return 0

    # Write diff to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(diff_text)
        diff_path = f.name

    # Build review command
    cmd = [
        sys.executable,
        "-m",
        "pr_reviewer",
        "review",
        diff_path,
        "--mode",
        mode,
        "--model",
        model,
        "--max-lines",
        max_lines,
        "--format",
        "text",
        "--color",
        "never",
    ]

    if post_comments:
        cmd.extend([
            "--post",
            "github",
            "--repo",
            repo,
            "--pr",
            str(pr_number),
        ])

    print(f"Running review (mode={mode}, model={model})...")
    result = subprocess.run(cmd, capture_output=False, text=True)

    # Clean up
    with contextlib.suppress(OSError):
        os.unlink(diff_path)

    if result.returncode != 0:
        print(f"::warning::Review exited with code {result.returncode}")

    return result.returncode


def _fetch_pr_diff(repo: str, pr_number: str, token: str | None) -> str:
    """Fetch the PR diff using curl (avoids adding requests as a dependency for the action)."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    cmd = [
        "curl",
        "-sS",
        "-f",
        "-H",
        "Accept: application/vnd.github.v3.diff",
        "-H",
        f"Authorization: Bearer {token}" if token else "",
        "-H",
        "X-GitHub-Api-Version: 2022-11-28",
        url,
    ]
    # Remove empty header if no token
    cmd = [arg for arg in cmd if arg]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"::error::Failed to fetch PR diff: {proc.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return proc.stdout


def _filter_diff(diff_text: str, exclude_patterns: list[str]) -> str:
    """Remove diff sections for files matching exclude patterns."""
    sections: list[str] = []
    current_section: list[str] = []
    current_file: str | None = None

    diff_header_re = re.compile(r"^diff --git a/(.+?) b/(.+)$")

    for line in diff_text.splitlines(keepends=True):
        header_match = diff_header_re.match(line)
        if header_match:
            # Flush previous section
            if current_section and current_file is not None and not _matches_any(current_file, exclude_patterns):
                sections.extend(current_section)
            current_section = [line]
            # Use the new path (b/ side)
            current_file = header_match.group(2)
        else:
            current_section.append(line)

    # Flush last section
    if current_section and current_file is not None and not _matches_any(current_file, exclude_patterns):
        sections.extend(current_section)

    return "".join(sections)


def _matches_any(file_path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(file_path, pattern) for pattern in patterns)


if __name__ == "__main__":
    raise SystemExit(main())
