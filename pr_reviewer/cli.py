from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from textwrap import dedent

import requests

from . import __version__
from .formatters import format_review
from .llm import LLMError, OpenAICompatibleProvider, ProviderConfigError
from .parsing import read_patch_file
from .reviewer import PRReviewer

logger = logging.getLogger(__name__)

_CONFIGURABLE_REVIEW_OPTIONS = {
    "model",
    "mode",
    "max_lines",
    "format",
    "compact",
    "base_url",
    "color",
    "post",
    "repo",
    "pr",
    "mr",
    "integration_token",
    "integration_base_url",
    "dry_run_post",
}
_CHOICE_VALIDATORS = {
    "mode": {"single", "multi"},
    "format": {"text", "json", "markdown"},
    "color": {"auto", "always", "never"},
    "post": {"github", "gitlab"},
}
_INT_VALIDATORS = {"max_lines", "pr", "mr"}
_BOOL_VALIDATORS = {"compact", "dry_run_post"}


def _setup_logging(*, verbose: bool = False, debug: bool = False) -> None:
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    root = logging.getLogger("pr_reviewer")
    root.setLevel(level)
    root.addHandler(handler)


class ConfigError(RuntimeError):
    pass


def _default_review_options() -> dict[str, object]:
    return {
        "model": os.getenv("PR_REVIEWER_MODEL", "gpt-4.1-mini"),
        "mode": "single",
        "max_lines": 1200,
        "format": "text",
        "compact": False,
        "base_url": os.getenv("PR_REVIEWER_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
        "color": "auto",
        "post": None,
        "repo": None,
        "pr": None,
        "mr": None,
        "integration_token": None,
        "integration_base_url": None,
        "dry_run_post": False,
    }


def build_parser(*, review_defaults: dict[str, object] | None = None) -> argparse.ArgumentParser:
    defaults = _default_review_options()
    if review_defaults:
        defaults.update(review_defaults)

    parser = argparse.ArgumentParser(
        prog="pr-reviewer",
        description=(
            "Review a unified diff with an LLM and emit structured, actionable PR feedback."
        ),
        epilog=dedent(
            """\
            Quick examples:
              python -m pr_reviewer review path/to/changes.diff
              git diff | python -m pr_reviewer review --stdin
              python -m pr_reviewer review --cached --mode multi --compact --color always
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--config",
        help="Path to .pr-reviewer.toml or pyproject.toml to use for default review settings",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose (INFO level) logging on stderr",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug (DEBUG level) logging on stderr",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    review_parser = subparsers.add_parser(
        "review",
        help="Analyze a diff from file, stdin, or staged git changes",
        description=(
            "Review code changes in unified diff format and return categorized findings "
            "(bug, security, performance, maintainability) plus a final verdict."
        ),
        epilog=dedent(
            """\
            Inputs are mutually exclusive:
              1) Provide [patch] file path
              2) Use --stdin
              3) Use --cached (runs: git diff --cached)
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    review_parser.add_argument(
        "patch",
        nargs="?",
        help="Path to a unified .diff/.patch file",
    )
    review_parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read diff text from stdin (for example: git diff | ...)",
    )
    review_parser.add_argument(
        "--cached",
        action="store_true",
        help="Read staged changes using `git diff --cached`",
    )
    review_parser.add_argument(
        "--model",
        default=defaults["model"],
        help="LLM model identifier sent to the provider",
    )
    review_parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default=defaults["mode"],
        help="Review strategy: single-pass or multi-pass (correctness/security/performance)",
    )
    review_parser.add_argument(
        "--max-lines",
        type=int,
        default=defaults["max_lines"],
        help="Approximate per-request diff line budget before the review splits into chunks",
    )
    review_parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default=defaults["format"],
        help="Render format for review output",
    )
    review_parser.add_argument(
        "--save",
        help="Write rendered output to a file (in addition to stdout)",
    )
    review_parser.add_argument(
        "--compact",
        action="store_true",
        default=defaults["compact"],
        help="Compact list view (one line per finding)",
    )
    review_parser.add_argument(
        "--base-url",
        default=defaults["base_url"],
        help="OpenAI-compatible API base URL (default: OpenAI v1 endpoint)",
    )
    review_parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default=defaults["color"],
        help="Color mode for text output",
    )
    review_parser.add_argument(
        "--post",
        choices=["github", "gitlab"],
        default=defaults["post"],
        help="Post findings as inline review comments to a PR/MR",
    )
    review_parser.add_argument(
        "--repo",
        default=defaults["repo"],
        help=(
            "Repository reference for posting: GitHub owner/name or GitLab project id/path"
        ),
    )
    review_parser.add_argument(
        "--pr",
        type=int,
        default=defaults["pr"],
        help="GitHub pull request number (required with --post github)",
    )
    review_parser.add_argument(
        "--mr",
        type=int,
        default=defaults["mr"],
        help="GitLab merge request IID (required with --post gitlab)",
    )
    review_parser.add_argument(
        "--integration-token",
        default=defaults["integration_token"],
        help="Token override for posting (otherwise uses GITHUB_TOKEN or GITLAB_TOKEN)",
    )
    review_parser.add_argument(
        "--integration-base-url",
        default=defaults["integration_base_url"],
        help="API base URL override for posting (GitHub Enterprise / self-hosted GitLab)",
    )
    review_parser.add_argument(
        "--dry-run-post",
        action="store_true",
        default=defaults["dry_run_post"],
        help="Simulate posting without making network calls",
    )
    review_parser.add_argument(
        "--no-context",
        action="store_true",
        default=False,
        help="Skip automatic file context fetching (fetched by default when --post github is used)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    try:
        review_defaults = _load_review_config(_extract_config_arg(argv))
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser = build_parser(review_defaults=review_defaults)
    args = parser.parse_args(argv)

    _setup_logging(verbose=getattr(args, "verbose", False), debug=getattr(args, "debug", False))

    if args.command == "review":
        return run_review(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


def run_review(args: argparse.Namespace) -> int:
    validation_error = _validate_post_args(args)
    if validation_error:
        logger.error("%s", validation_error)
        return 2

    try:
        diff_text = _resolve_diff_input(args)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 2

    if not diff_text.strip():
        logger.error("empty diff input")
        return 2

    # Auto-fetch file + project context when posting to GitHub (we have repo + PR number)
    file_context: dict[str, str] | None = None
    project_context: dict[str, str] | None = None
    if args.post == "github" and args.repo and args.pr and not getattr(args, "no_context", False):
        from .context import fetch_github_file_context, fetch_project_context
        from .parsing import parse_diff_stats

        # Resolve PR head SHA for accurate file fetching
        ref = _resolve_github_head_sha(
            repo=args.repo,
            pr_number=args.pr,
            token=args.integration_token,
        )

        # Per-file context: full content of each changed file
        file_paths = parse_diff_stats(diff_text).files
        if file_paths:
            file_context = fetch_github_file_context(
                repo=args.repo,
                file_paths=file_paths,
                ref=ref,
                token=args.integration_token,
            )
            if file_context:
                logger.info("Fetched file context for %d file(s)", len(file_context))

        # Project context: README, conventions, config manifest
        project_context = fetch_project_context(
            repo=args.repo,
            ref=ref,
            token=args.integration_token,
        )
        if project_context:
            logger.info("Fetched project context: %s", ", ".join(project_context.keys()))

    try:
        provider = OpenAICompatibleProvider(base_url=args.base_url)
        reviewer = PRReviewer(provider)
        result = reviewer.review(
            diff_text=diff_text,
            model=args.model,
            max_lines=args.max_lines,
            review_mode=args.mode,
            file_context=file_context,
            project_context=project_context,
        )
    except (ProviderConfigError, LLMError) as exc:
        logger.error("%s", exc)
        return 1

    color = _use_color(args.color, args.format)
    rendered = format_review(
        result,
        output_format=args.format,
        compact=args.compact,
        color=color,
    )
    print(rendered)

    if args.save:
        output_path = Path(args.save).expanduser()
        try:
            if output_path.exists() and output_path.is_dir():
                raise IsADirectoryError(f"{output_path} is a directory")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            logger.error("could not save output to %s: %s", output_path, exc)
            return 1

        print(f"Saved output to {output_path}", file=sys.stderr)

    if args.post:
        from .integrations import IntegrationError, post_findings

        try:
            report = post_findings(
                platform=args.post,
                result=result,
                diff_text=diff_text,
                repo=args.repo,
                pr_number=args.pr,
                mr_iid=args.mr,
                token=args.integration_token,
                base_url=args.integration_base_url,
                dry_run=args.dry_run_post,
            )
        except IntegrationError as exc:
            logger.error("%s", exc)
            return 1

        _print_posting_report(report=report, dry_run=args.dry_run_post)

    return 0


def _resolve_diff_input(args: argparse.Namespace) -> str:
    selected_modes = sum(bool(mode) for mode in [args.stdin, args.cached, bool(args.patch)])
    if selected_modes == 0:
        raise ValueError("provide one input source: patch path, --stdin, or --cached")
    if selected_modes > 1:
        raise ValueError("choose only one input source")

    if args.stdin:
        if sys.stdin.isatty():
            raise ValueError(
                "stdin is a terminal — did you forget to pipe a diff? "
                "Example: git diff | pr-reviewer review --stdin"
            )
        return sys.stdin.read()

    if args.cached:
        proc = subprocess.run(
            ["git", "diff", "--cached"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "failed to run git diff --cached")
        return proc.stdout

    if not args.patch:
        raise ValueError("missing patch file path")

    return read_patch_file(args.patch)


def _use_color(color_mode: str, output_format: str) -> bool:
    if output_format != "text":
        return False
    if color_mode == "always":
        return True
    if color_mode == "never":
        return False
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _validate_post_args(args: argparse.Namespace) -> str | None:
    if args.dry_run_post and not args.post:
        return "--dry-run-post requires --post"

    if args.repo and not args.post:
        return "--repo requires --post"

    if args.integration_token and not args.post:
        return "--integration-token requires --post"

    if args.integration_base_url and not args.post:
        return "--integration-base-url requires --post"

    if args.pr and args.post != "github":
        return "--pr requires --post github"

    if args.mr and args.post != "gitlab":
        return "--mr requires --post gitlab"

    if not args.post:
        return None

    if not args.repo:
        return "--post requires --repo"

    if args.post == "github" and not args.pr:
        return "--post github requires --pr <number>"

    if args.post == "github" and args.mr:
        return "--post github does not accept --mr"

    if args.post == "gitlab" and not args.mr:
        return "--post gitlab requires --mr <iid>"

    if args.post == "gitlab" and args.pr:
        return "--post gitlab does not accept --pr"

    return None


def _resolve_github_head_sha(
    *,
    repo: str,
    pr_number: int,
    token: str | None,
) -> str:
    """Fetch the head SHA of a GitHub PR. Falls back to 'HEAD' on error."""
    token = token or os.getenv("GITHUB_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{repo}/pulls/{pr_number}",
            headers=headers,
            timeout=(10, 30),
        )
        if resp.status_code == 200:
            sha = resp.json().get("head", {}).get("sha")
            if sha:
                return sha
    except requests.RequestException as exc:
        logger.debug("Could not resolve PR head SHA: %s", exc)
    return "HEAD"


def _print_posting_report(*, report, dry_run: bool) -> None:
    mode = "dry-run posted" if dry_run else "posted"
    print(
        f"{report.platform} comments {mode}: {report.posted}/{report.attempted} "
        f"(skipped: {report.skipped})",
        file=sys.stderr,
    )
    for error in report.errors[:8]:
        print(f"  - {error}", file=sys.stderr)


def _extract_config_arg(argv: list[str]) -> str | None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    parsed, _ = bootstrap.parse_known_args(argv)
    return parsed.config or os.getenv("PR_REVIEWER_CONFIG")


def _load_review_config(explicit_path: str | None) -> dict[str, object]:
    config_path = _resolve_config_path(explicit_path)
    if config_path is None:
        return {}

    try:
        with config_path.open("rb") as handle:
            raw_data = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"could not read config file {config_path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc

    if config_path.name == "pyproject.toml":
        tool_data = raw_data.get("tool")
        if not isinstance(tool_data, dict):
            return {}
        config_data = tool_data.get("pr-reviewer")
        if config_data is None:
            return {}
    else:
        config_data = raw_data.get("review", raw_data)

    if not isinstance(config_data, dict):
        raise ConfigError(f"config section in {config_path} must be a TOML table")

    return _validate_review_config(config_data, config_path)


def _resolve_config_path(explicit_path: str | None) -> Path | None:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        return path

    for directory in [Path.cwd(), *Path.cwd().parents]:
        dedicated = directory / ".pr-reviewer.toml"
        if dedicated.is_file():
            return dedicated

        pyproject = directory / "pyproject.toml"
        if pyproject.is_file() and _pyproject_has_review_config(pyproject):
            return pyproject

    return None


def _pyproject_has_review_config(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return False

    tool = data.get("tool")
    return isinstance(tool, dict) and isinstance(tool.get("pr-reviewer"), dict)


def _validate_review_config(config_data: dict[str, object], config_path: Path) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for raw_key, value in config_data.items():
        key = raw_key.replace("-", "_")
        if key not in _CONFIGURABLE_REVIEW_OPTIONS:
            raise ConfigError(f"unsupported config key in {config_path}: {raw_key}")

        if value is None:
            normalized[key] = None
            continue

        if key in _BOOL_VALIDATORS:
            if not isinstance(value, bool):
                raise ConfigError(f"config key {raw_key} in {config_path} must be a boolean")
            normalized[key] = value
            continue

        if key in _INT_VALIDATORS:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigError(f"config key {raw_key} in {config_path} must be an integer")
            normalized[key] = value
            continue

        if not isinstance(value, str):
            raise ConfigError(f"config key {raw_key} in {config_path} must be a string")

        if key in _CHOICE_VALIDATORS and value not in _CHOICE_VALIDATORS[key]:
            allowed = ", ".join(sorted(_CHOICE_VALIDATORS[key]))
            raise ConfigError(
                f"config key {raw_key} in {config_path} must be one of: {allowed}"
            )

        normalized[key] = value

    return normalized
