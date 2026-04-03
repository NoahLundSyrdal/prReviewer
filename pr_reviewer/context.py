from __future__ import annotations

import base64
import logging
import os

import requests

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 10
_READ_TIMEOUT = 30
_REQUEST_TIMEOUT = (_CONNECT_TIMEOUT, _READ_TIMEOUT)
_MAX_FILE_LINES = 300

# Project-level files fetched once per review to give the model broader context.
# Ordered by priority — first match wins within each group.
_PROJECT_CONTEXT_CANDIDATES: list[tuple[str, int]] = [
    # Project description / purpose (most valuable — tells the model what it's reviewing)
    ("README.md", 150),
    ("README.rst", 150),
    ("README", 100),
    # Conventions / contribution guidelines
    ("CONTRIBUTING.md", 80),
    ("AGENTS.md", 80),
    ("DEVELOPMENT.md", 80),
    # Architecture / design docs (first found wins)
    ("docs/architecture.md", 100),
    ("docs/ARCHITECTURE.md", 100),
    ("docs/design.md", 80),
    # Primary package/config manifest (reveals language, deps, tooling)
    ("pyproject.toml", 60),
    ("package.json", 60),
    ("go.mod", 40),
    ("Cargo.toml", 40),
    ("pom.xml", 40),
    ("build.gradle", 40),
]


def fetch_github_file_context(
    *,
    repo: str,
    file_paths: list[str],
    ref: str,
    token: str | None,
    base_url: str = "https://api.github.com",
) -> dict[str, str]:
    """Fetch full file content from GitHub API. Returns {path: content}. Silently skips failures."""
    token = token or os.getenv("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    context: dict[str, str] = {}
    session = requests.Session()
    session.headers.update(headers)

    for path in file_paths:
        url = f"{base_url.rstrip('/')}/repos/{repo}/contents/{path}"
        try:
            resp = session.get(url, params={"ref": ref}, timeout=_REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            logger.debug("Failed to fetch context for %s: %s", path, exc)
            continue

        if resp.status_code == 404:
            # File may be deleted in this PR — skip silently
            continue

        if resp.status_code >= 400:
            logger.debug("GitHub returned %d for %s", resp.status_code, path)
            continue

        try:
            data = resp.json()
            content_b64 = data.get("content", "")
            if not content_b64:
                continue
            content = base64.b64decode(content_b64.replace("\n", "")).decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to decode context for %s: %s", path, exc)
            continue

        lines = content.splitlines()
        if len(lines) > _MAX_FILE_LINES:
            content = "\n".join(lines[:_MAX_FILE_LINES]) + f"\n# ... truncated at {_MAX_FILE_LINES} lines ..."

        context[path] = content

    logger.debug("Fetched file context for %d/%d files", len(context), len(file_paths))
    return context


def fetch_project_context(
    *,
    repo: str,
    ref: str,
    token: str | None,
    base_url: str = "https://api.github.com",
    candidates: list[tuple[str, int]] | None = None,
) -> dict[str, str]:
    """Fetch project-level context files (README, conventions, config manifest).

    Returns a dict of {path: truncated_content} for whichever candidates exist.
    Files are fetched in candidate order; each has its own line cap.
    Failures are silently skipped.
    """
    token = token or os.getenv("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    if candidates is None:
        candidates = _PROJECT_CONTEXT_CANDIDATES

    context: dict[str, str] = {}
    session = requests.Session()
    session.headers.update(headers)

    # Group candidates by their "role" so we stop after the first hit per group.
    # Groups are defined by sequential runs of the same comment block above each entry.
    # Simpler heuristic: stop after first README hit, first CONTRIBUTING hit, etc.
    # We implement this by tracking which prefix groups have been satisfied.
    _readme_done = False
    _contributing_done = False
    _arch_done = False
    _manifest_done = False

    for path, max_lines in candidates:
        # Skip if this group already has a hit
        lower = path.lower()
        if lower.startswith("readme") and _readme_done:
            continue
        if any(k in lower for k in ("contributing", "agents", "development")) and _contributing_done:
            continue
        if "docs/" in lower and _arch_done:
            continue
        if any(lower.endswith(ext) for ext in (".toml", ".json", ".mod", ".gradle", ".xml")) and _manifest_done:
            continue

        url = f"{base_url.rstrip('/')}/repos/{repo}/contents/{path}"
        try:
            resp = session.get(url, params={"ref": ref}, timeout=_REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            logger.debug("Failed to fetch project context for %s: %s", path, exc)
            continue

        if resp.status_code == 404:
            continue
        if resp.status_code >= 400:
            logger.debug("GitHub returned %d fetching project context %s", resp.status_code, path)
            continue

        try:
            data = resp.json()
            content_b64 = data.get("content", "")
            if not content_b64:
                continue
            content = base64.b64decode(content_b64.replace("\n", "")).decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to decode project context for %s: %s", path, exc)
            continue

        lines = content.splitlines()
        if len(lines) > max_lines:
            content = "\n".join(lines[:max_lines]) + f"\n# ... truncated at {max_lines} lines ..."

        context[path] = content

        # Mark this group as satisfied
        if lower.startswith("readme"):
            _readme_done = True
        elif any(k in lower for k in ("contributing", "agents", "development")):
            _contributing_done = True
        elif "docs/" in lower:
            _arch_done = True
        else:
            _manifest_done = True

    logger.debug("Fetched project context: %s", list(context.keys()))
    return context
