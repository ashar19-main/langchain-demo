import sys
import types

import pytest

from guardrails.bedrock_guardrails import (
    BEDROCK_BEARER_TOKEN_ENV_VAR,
    BEDROCK_GUARDRAIL_REFUSAL_MESSAGE,
    BedrockGuardrailConfig,
    _get_blocked_message,
    _get_intervention_category,
    _normalize_bedrock_output_message,
    evaluate_prompt_with_bedrock_guardrail,
    without_bedrock_bearer_token,
)
from guardrails.input_classifier import (
    ABUSIVE_LANGUAGE_REFUSAL_MESSAGE,
    DEFAULT_REFUSAL_MESSAGE,
)


def test_without_bedrock_bearer_token_temporarily_removes_and_restores_env(monkeypatch):
    monkeypatch.setenv(BEDROCK_BEARER_TOKEN_ENV_VAR, "token")

    with without_bedrock_bearer_token():
        assert BEDROCK_BEARER_TOKEN_ENV_VAR not in __import__("os").environ

    assert __import__("os").environ[BEDROCK_BEARER_TOKEN_ENV_VAR] == "token"


def test_without_bedrock_bearer_token_leaves_missing_env_missing(monkeypatch):
    monkeypatch.delenv(BEDROCK_BEARER_TOKEN_ENV_VAR, raising=False)

    with without_bedrock_bearer_token():
        pass

    assert BEDROCK_BEARER_TOKEN_ENV_VAR not in __import__("os").environ


def test_bedrock_guardrail_config_from_env(monkeypatch):
    monkeypatch.setenv("BEDROCK_GUARDRAIL_ID", "gr-1")
    monkeypatch.setenv("BEDROCK_GUARDRAIL_VERSION", "7")
    monkeypatch.setenv("BEDROCK_GUARDRAIL_REGION", "us-east-1")
    monkeypatch.setenv("BEDROCK_GUARDRAIL_FAIL_CLOSED", "false")

    config = BedrockGuardrailConfig.from_env()

    assert config.guardrail_id == "gr-1"
    assert config.guardrail_version == "7"
    assert config.region_name == "us-east-1"
    assert config.fail_closed is False
    assert config.is_configured


def test_bedrock_guardrail_config_uses_aws_region_fallback(monkeypatch):
    monkeypatch.delenv("BEDROCK_GUARDRAIL_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")

    config = BedrockGuardrailConfig.from_env()

    assert config.region_name == "ap-south-1"
    assert not config.is_configured


def test_evaluate_prompt_returns_none_when_not_configured():
    result = evaluate_prompt_with_bedrock_guardrail(
        "hello",
        BedrockGuardrailConfig(guardrail_id="", guardrail_version="DRAFT"),
    )

    assert result is None


@pytest.mark.parametrize("action", ["NONE", "GUARDRAIL_ALLOWED"])
def test_evaluate_prompt_returns_none_when_guardrail_does_not_intervene(
    monkeypatch, action
):
    fake_boto3 = _install_fake_boto3(
        monkeypatch,
        {"action": action},
    )

    result = evaluate_prompt_with_bedrock_guardrail(
        "hello",
        BedrockGuardrailConfig("gr-1", "DRAFT", "us-west-2"),
    )

    assert result is None
    assert fake_boto3.client_calls == [("bedrock-runtime", "us-west-2")]
    assert fake_boto3.runtime.prompts == ["hello"]


def test_evaluate_prompt_returns_classification_when_guardrail_intervenes(monkeypatch):
    _install_fake_boto3(
        monkeypatch,
        {
            "action": "GUARDRAIL_INTERVENED",
            "actionReason": "Blocked by policy",
            "outputs": [{"text": "blocked"}],
            "assessments": [
                {
                    "contentPolicy": {
                        "filters": [{"type": "VIOLENCE", "detected": True}]
                    }
                }
            ],
        },
    )

    classification = evaluate_prompt_with_bedrock_guardrail(
        "bad prompt",
        BedrockGuardrailConfig("gr-1", "DRAFT", "us-west-2"),
    )

    assert classification is not None
    assert not classification.allowed
    assert classification.category == "bedrock_violence"
    assert classification.reason == "Blocked by policy"
    assert classification.refusal_message == "blocked"


def test_evaluate_prompt_fails_open_when_configured(monkeypatch):
    _install_fake_boto3(monkeypatch, RuntimeError("network down"))

    result = evaluate_prompt_with_bedrock_guardrail(
        "hello",
        BedrockGuardrailConfig("gr-1", "DRAFT", fail_closed=False),
    )

    assert result is None


def test_evaluate_prompt_fails_closed_by_default(monkeypatch):
    _install_fake_boto3(monkeypatch, RuntimeError("network down"))

    classification = evaluate_prompt_with_bedrock_guardrail(
        "hello",
        BedrockGuardrailConfig("gr-1", "DRAFT"),
    )

    assert classification is not None
    assert not classification.allowed
    assert classification.category == "bedrock_guardrail_error"
    assert "network down" in classification.reason
    assert classification.refusal_message == DEFAULT_REFUSAL_MESSAGE


@pytest.mark.parametrize(
    ("response", "category"),
    [
        (
            {"assessments": [{"contentPolicy": {"filters": [{"type": "HATE", "detected": True}]}}]},
            "bedrock_hate",
        ),
        ({"assessments": [{"wordPolicy": {"customWords": [{"match": "x"}]}}]}, "bedrock_word_policy"),
        (
            {"assessments": [{"wordPolicy": {"managedWordLists": [{"type": "PROFANITY"}]}}]},
            "bedrock_word_policy",
        ),
        ({"assessments": [{"topicPolicy": {"topics": [{"detected": True}]}}]}, "bedrock_denied_topic"),
        ({}, "bedrock_guardrail"),
    ],
)
def test_get_intervention_category(response, category):
    assert _get_intervention_category(response) == category


@pytest.mark.parametrize("category", ["bedrock_insults", "bedrock_word_policy"])
def test_get_blocked_message_uses_abuse_message_for_abusive_categories(category):
    assert _get_blocked_message({}, category) == ABUSIVE_LANGUAGE_REFUSAL_MESSAGE


def test_get_blocked_message_uses_bedrock_output_or_default():
    assert _get_blocked_message({"outputs": [{"text": "custom blocked"}]}, "other") == "custom blocked"
    assert _get_blocked_message({"outputs": [{"text": ""}]}, "other") == BEDROCK_GUARDRAIL_REFUSAL_MESSAGE


def test_get_blocked_message_normalizes_legacy_capability_refusal():
    legacy_message = (
        "I can't help with that request. This demo can only help with calculator, "
        "project file listing, and approved project file reading."
    )

    assert _get_blocked_message({"outputs": [{"text": legacy_message}]}, "bedrock_sexual") == (
        DEFAULT_REFUSAL_MESSAGE
    )


def test_normalize_bedrock_output_message_preserves_current_capability_refusal():
    assert _normalize_bedrock_output_message(DEFAULT_REFUSAL_MESSAGE) == (
        DEFAULT_REFUSAL_MESSAGE
    )


def _install_fake_boto3(monkeypatch, response_or_error):
    class FakeRuntime:
        def __init__(self):
            self.prompts = []

        def apply_guardrail(self, **kwargs):
            self.prompts.append(kwargs["content"][0]["text"]["text"])
            if isinstance(response_or_error, Exception):
                raise response_or_error
            return response_or_error

    class FakeBoto3(types.SimpleNamespace):
        def __init__(self):
            super().__init__()
            self.runtime = FakeRuntime()
            self.client_calls = []

        def client(self, service_name, region_name=None):
            self.client_calls.append((service_name, region_name))
            return self.runtime

    fake_boto3 = FakeBoto3()
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    return fake_boto3
