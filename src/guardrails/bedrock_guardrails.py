import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from guardrails.input_classifier import (
    ABUSIVE_LANGUAGE_REFUSAL_MESSAGE,
    DEFAULT_REFUSAL_MESSAGE,
    InputClassification,
)


BEDROCK_GUARDRAIL_REFUSAL_MESSAGE = (
    "I can't help with that request because it was blocked by the configured "
    "Amazon Bedrock Guardrail."
)
LEGACY_CAPABILITY_REFUSAL_PREFIX = (
    "I can't help with that request. This demo can only help with calculator,"
)

BEDROCK_BEARER_TOKEN_ENV_VAR = "AWS_BEARER_TOKEN_BEDROCK"


@contextmanager
def without_bedrock_bearer_token():
    """
    Temporarily remove the Bedrock API-key env var for boto3 guardrail calls.

    boto3 Bedrock clients can automatically pick up AWS_BEARER_TOKEN_BEDROCK.
    The demo uses that token for the OpenAI-compatible model endpoint, but
    guardrail management/evaluation should use normal AWS SDK credentials.
    """
    bedrock_bearer_token = os.environ.pop(BEDROCK_BEARER_TOKEN_ENV_VAR, None)

    try:
        yield
    finally:
        if bedrock_bearer_token is not None:
            os.environ[BEDROCK_BEARER_TOKEN_ENV_VAR] = bedrock_bearer_token


@dataclass(frozen=True)
class BedrockGuardrailConfig:
    guardrail_id: str
    guardrail_version: str
    region_name: str | None = None
    fail_closed: bool = True

    @property
    def is_configured(self) -> bool:
        return bool(self.guardrail_id and self.guardrail_version)

    @classmethod
    def from_env(cls) -> "BedrockGuardrailConfig":
        return cls(
            guardrail_id=os.getenv("BEDROCK_GUARDRAIL_ID", ""),
            guardrail_version=os.getenv("BEDROCK_GUARDRAIL_VERSION", "DRAFT"),
            region_name=(
                os.getenv("BEDROCK_GUARDRAIL_REGION")
                or os.getenv("AWS_REGION")
                or os.getenv("AWS_DEFAULT_REGION")
            ),
            fail_closed=os.getenv("BEDROCK_GUARDRAIL_FAIL_CLOSED", "true").lower()
            not in {"0", "false", "no"},
        )


def evaluate_prompt_with_bedrock_guardrail(
    prompt: str,
    config: BedrockGuardrailConfig | None = None,
) -> InputClassification | None:
    """
    Evaluate a prompt with Amazon Bedrock Guardrails.

    Returns None when Bedrock Guardrails are not configured or when the
    guardrail allows the prompt.
    """
    guardrail_config = config or BedrockGuardrailConfig.from_env()

    if not guardrail_config.is_configured:
        return None

    try:
        with without_bedrock_bearer_token():
            import boto3

            bedrock_runtime = boto3.client(
                "bedrock-runtime",
                region_name=guardrail_config.region_name,
            )
            response = bedrock_runtime.apply_guardrail(
                guardrailIdentifier=guardrail_config.guardrail_id,
                guardrailVersion=guardrail_config.guardrail_version,
                source="INPUT",
                content=[{"text": {"text": prompt}}],
                outputScope="FULL",
            )
    except Exception as ex:
        if not guardrail_config.fail_closed:
            return None

        return InputClassification(
            allowed=False,
            category="bedrock_guardrail_error",
            reason=f"Bedrock Guardrail evaluation failed: {ex}",
            refusal_message=DEFAULT_REFUSAL_MESSAGE,
        )

    if response.get("action") != "GUARDRAIL_INTERVENED":
        return None

    category = _get_intervention_category(response)

    return InputClassification(
        allowed=False,
        category=category,
        reason=response.get("actionReason", "Bedrock Guardrail intervened."),
        refusal_message=_get_blocked_message(response, category),
    )


def _get_blocked_message(response: dict[str, Any], category: str) -> str:
    if category in {"bedrock_insults", "bedrock_word_policy"}:
        return ABUSIVE_LANGUAGE_REFUSAL_MESSAGE

    for output in response.get("outputs", []) or []:
        output_text = output.get("text")
        if output_text:
            return _normalize_bedrock_output_message(output_text)

    return BEDROCK_GUARDRAIL_REFUSAL_MESSAGE


def _normalize_bedrock_output_message(output_text: str) -> str:
    """
    Replace older remote Bedrock block messages with the current local wording.
    """
    if (
        output_text.startswith(LEGACY_CAPABILITY_REFUSAL_PREFIX)
        and "image search" not in output_text.lower()
    ):
        return DEFAULT_REFUSAL_MESSAGE

    return output_text


def _get_intervention_category(response: dict[str, Any]) -> str:
    for assessment in response.get("assessments", []) or []:
        content_policy = assessment.get("contentPolicy", {})
        for content_filter in content_policy.get("filters", []) or []:
            if content_filter.get("detected"):
                return f"bedrock_{content_filter.get('type', 'content').lower()}"

        word_policy = assessment.get("wordPolicy", {})
        if word_policy.get("customWords") or word_policy.get("managedWordLists"):
            return "bedrock_word_policy"

        topic_policy = assessment.get("topicPolicy", {})
        for topic in topic_policy.get("topics", []) or []:
            if topic.get("detected"):
                return "bedrock_denied_topic"

    return "bedrock_guardrail"
