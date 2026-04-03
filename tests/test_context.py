"""Tests for pr_reviewer.context."""
from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import requests

from pr_reviewer.context import _MAX_FILE_LINES, fetch_github_file_context, fetch_project_context


def _b64(content: str) -> str:
    """Base64-encode a string the same way the GitHub API does."""
    return base64.b64encode(content.encode()).decode()


def _mock_response(status: int, json_data: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data or {}
    return resp


class TestFetchGithubFileContext:
    def test_success_returns_decoded_content(self) -> None:
        content = "def foo():\n    return 42\n"
        mock_resp = _mock_response(200, {"content": _b64(content)})

        with patch("pr_reviewer.context.requests.Session") as mock_session_cls:
            session = MagicMock()
            session.get.return_value = mock_resp
            mock_session_cls.return_value = session

            result = fetch_github_file_context(
                repo="owner/repo",
                file_paths=["src/foo.py"],
                ref="abc123",
                token="tok",
            )

        assert result == {"src/foo.py": content}

    def test_404_is_skipped_gracefully(self) -> None:
        mock_resp = _mock_response(404)

        with patch("pr_reviewer.context.requests.Session") as mock_session_cls:
            session = MagicMock()
            session.get.return_value = mock_resp
            mock_session_cls.return_value = session

            result = fetch_github_file_context(
                repo="owner/repo",
                file_paths=["deleted.py"],
                ref="abc123",
                token="tok",
            )

        assert result == {}

    def test_network_error_is_skipped_gracefully(self) -> None:
        with patch("pr_reviewer.context.requests.Session") as mock_session_cls:
            session = MagicMock()
            session.get.side_effect = requests.ConnectionError("no network")
            mock_session_cls.return_value = session

            result = fetch_github_file_context(
                repo="owner/repo",
                file_paths=["src/foo.py"],
                ref="abc123",
                token=None,
            )

        assert result == {}

    def test_other_http_error_is_skipped(self) -> None:
        mock_resp = _mock_response(500)

        with patch("pr_reviewer.context.requests.Session") as mock_session_cls:
            session = MagicMock()
            session.get.return_value = mock_resp
            mock_session_cls.return_value = session

            result = fetch_github_file_context(
                repo="owner/repo",
                file_paths=["src/foo.py"],
                ref="abc123",
                token="tok",
            )

        assert result == {}

    def test_truncation_at_max_lines(self) -> None:
        # Build a file with more lines than the cap
        lines = [f"line {i}" for i in range(_MAX_FILE_LINES + 50)]
        content = "\n".join(lines)
        mock_resp = _mock_response(200, {"content": _b64(content)})

        with patch("pr_reviewer.context.requests.Session") as mock_session_cls:
            session = MagicMock()
            session.get.return_value = mock_resp
            mock_session_cls.return_value = session

            result = fetch_github_file_context(
                repo="owner/repo",
                file_paths=["big.py"],
                ref="abc123",
                token="tok",
            )

        assert "big.py" in result
        truncated = result["big.py"]
        assert f"truncated at {_MAX_FILE_LINES} lines" in truncated
        result_lines = truncated.splitlines()
        # Should have _MAX_FILE_LINES content lines + 1 truncation notice
        assert len(result_lines) == _MAX_FILE_LINES + 1

    def test_no_truncation_under_limit(self) -> None:
        lines = [f"line {i}" for i in range(10)]
        content = "\n".join(lines)
        mock_resp = _mock_response(200, {"content": _b64(content)})

        with patch("pr_reviewer.context.requests.Session") as mock_session_cls:
            session = MagicMock()
            session.get.return_value = mock_resp
            mock_session_cls.return_value = session

            result = fetch_github_file_context(
                repo="owner/repo",
                file_paths=["small.py"],
                ref="abc123",
                token="tok",
            )

        assert result["small.py"] == content

    def test_multiple_files_partial_failure(self) -> None:
        good_content = "x = 1\n"
        good_resp = _mock_response(200, {"content": _b64(good_content)})
        bad_resp = _mock_response(404)

        with patch("pr_reviewer.context.requests.Session") as mock_session_cls:
            session = MagicMock()
            session.get.side_effect = [good_resp, bad_resp]
            mock_session_cls.return_value = session

            result = fetch_github_file_context(
                repo="owner/repo",
                file_paths=["good.py", "deleted.py"],
                ref="abc123",
                token="tok",
            )

        assert "good.py" in result
        assert "deleted.py" not in result

    def test_empty_content_field_skipped(self) -> None:
        mock_resp = _mock_response(200, {"content": ""})

        with patch("pr_reviewer.context.requests.Session") as mock_session_cls:
            session = MagicMock()
            session.get.return_value = mock_resp
            mock_session_cls.return_value = session

            result = fetch_github_file_context(
                repo="owner/repo",
                file_paths=["empty.py"],
                ref="abc123",
                token="tok",
            )

        assert result == {}

    def test_no_token_omits_auth_header(self) -> None:
        content = "pass\n"
        mock_resp = _mock_response(200, {"content": _b64(content)})

        with patch("pr_reviewer.context.requests.Session") as mock_session_cls:
            session = MagicMock()
            session.get.return_value = mock_resp
            mock_session_cls.return_value = session

            with patch.dict("os.environ", {}, clear=True):
                result = fetch_github_file_context(
                    repo="owner/repo",
                    file_paths=["src/x.py"],
                    ref="HEAD",
                    token=None,
                )

        # Should still work — no auth header
        assert "src/x.py" in result
        call_kwargs = session.headers.update.call_args[0][0]
        assert "Authorization" not in call_kwargs


# ---------------------------------------------------------------------------
# fetch_project_context
# ---------------------------------------------------------------------------


class TestFetchProjectContext:
    def test_fetches_readme_and_config(self) -> None:
        readme_content = "# My Project\nDoes cool stuff."
        config_content = "[tool.myapp]\nversion = '1.0'"

        def _get_side_effect(url, **kwargs):
            if "README.md" in url:
                return _mock_response(200, {"content": _b64(readme_content)})
            if "pyproject.toml" in url:
                return _mock_response(200, {"content": _b64(config_content)})
            return _mock_response(404)

        with patch("pr_reviewer.context.requests.Session") as mock_session_cls:
            session = MagicMock()
            session.get.side_effect = _get_side_effect
            mock_session_cls.return_value = session

            result = fetch_project_context(repo="owner/repo", ref="main", token="tok")

        assert "README.md" in result
        assert readme_content in result["README.md"]

    def test_stops_after_first_readme_hit(self) -> None:
        """Should not fetch README.rst if README.md succeeded."""
        readme_md_content = "# Found me"

        call_log: list[str] = []

        def _get_side_effect(url, **kwargs):
            call_log.append(url)
            if "README.md" in url:
                return _mock_response(200, {"content": _b64(readme_md_content)})
            return _mock_response(404)

        with patch("pr_reviewer.context.requests.Session") as mock_session_cls:
            session = MagicMock()
            session.get.side_effect = _get_side_effect
            mock_session_cls.return_value = session

            result = fetch_project_context(repo="owner/repo", ref="main", token="tok")

        assert "README.md" in result
        assert not any("README.rst" in url for url in call_log), "README.rst should not be fetched after README.md hit"

    def test_truncates_at_per_file_line_cap(self) -> None:
        # README.md has a cap of 150 lines
        lines = [f"line {i}" for i in range(200)]
        content = "\n".join(lines)

        with patch("pr_reviewer.context.requests.Session") as mock_session_cls:
            session = MagicMock()
            session.get.return_value = _mock_response(200, {"content": _b64(content)})
            mock_session_cls.return_value = session

            result = fetch_project_context(
                repo="owner/repo",
                ref="main",
                token="tok",
                candidates=[("README.md", 150)],
            )

        assert "README.md" in result
        assert "truncated at 150 lines" in result["README.md"]

    def test_all_404_returns_empty(self) -> None:
        with patch("pr_reviewer.context.requests.Session") as mock_session_cls:
            session = MagicMock()
            session.get.return_value = _mock_response(404)
            mock_session_cls.return_value = session

            result = fetch_project_context(repo="owner/repo", ref="main", token=None)

        assert result == {}

    def test_network_error_skipped(self) -> None:
        with patch("pr_reviewer.context.requests.Session") as mock_session_cls:
            session = MagicMock()
            session.get.side_effect = requests.ConnectionError("down")
            mock_session_cls.return_value = session

            result = fetch_project_context(repo="owner/repo", ref="main", token=None)

        assert result == {}
