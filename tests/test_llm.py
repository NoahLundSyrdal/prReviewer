from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import requests

from pr_reviewer.llm import OpenAICompatibleProvider

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "verdict": {"type": "string"},
    },
    "required": ["summary", "verdict"],
    "additionalProperties": False,
}


def _make_success_response(content: dict) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps(content)}}],
    }
    return resp


def _make_error_response(status_code: int, message: str = "error") -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = message
    resp.headers = {}
    resp.json.return_value = {"error": {"message": message}}
    return resp


def test_json_schema_mode_included_in_payload() -> None:
    """When json_schema is provided, payload should use response_format type=json_schema."""
    success_resp = _make_success_response({"summary": "ok", "verdict": "looks good"})

    with patch("pr_reviewer.llm.requests.post", return_value=success_resp) as mock_post:
        provider = OpenAICompatibleProvider(api_key="test-key")
        provider.complete_json(
            model="gpt-4",
            system_prompt="You are a reviewer.",
            user_prompt="Review this.",
            json_schema=SIMPLE_SCHEMA,
        )

    sent_payload = mock_post.call_args[1]["json"]
    assert sent_payload["response_format"]["type"] == "json_schema"
    assert sent_payload["response_format"]["json_schema"]["name"] == "review_result"
    assert sent_payload["response_format"]["json_schema"]["strict"] is True
    assert sent_payload["response_format"]["json_schema"]["schema"] == SIMPLE_SCHEMA


def test_json_object_fallback_when_no_schema() -> None:
    """When json_schema is not provided, payload should use response_format type=json_object."""
    success_resp = _make_success_response({"summary": "ok", "verdict": "looks good"})

    with patch("pr_reviewer.llm.requests.post", return_value=success_resp) as mock_post:
        provider = OpenAICompatibleProvider(api_key="test-key")
        provider.complete_json(
            model="gpt-4",
            system_prompt="You are a reviewer.",
            user_prompt="Review this.",
        )

    sent_payload = mock_post.call_args[1]["json"]
    assert sent_payload["response_format"] == {"type": "json_object"}


def test_graceful_fallback_on_400() -> None:
    """If json_schema mode returns 400, retry with json_object and succeed."""
    import copy

    error_resp = _make_error_response(400, "json_schema not supported")
    success_resp = _make_success_response({"summary": "ok", "verdict": "looks good"})

    captured_payloads: list[dict] = []

    def _capture_post(*args, **kwargs):
        captured_payloads.append(copy.deepcopy(kwargs.get("json", {})))
        if len(captured_payloads) == 1:
            return error_resp
        return success_resp

    with patch("pr_reviewer.llm.requests.post", side_effect=_capture_post):
        provider = OpenAICompatibleProvider(api_key="test-key", max_retries=3)
        result = provider.complete_json(
            model="gpt-4",
            system_prompt="You are a reviewer.",
            user_prompt="Review this.",
            json_schema=SIMPLE_SCHEMA,
        )

    assert result == json.dumps({"summary": "ok", "verdict": "looks good"})
    assert len(captured_payloads) == 2

    # First call should have used json_schema
    assert captured_payloads[0]["response_format"]["type"] == "json_schema"

    # Second call should have fallen back to json_object
    assert captured_payloads[1]["response_format"] == {"type": "json_object"}
