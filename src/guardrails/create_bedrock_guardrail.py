import argparse
import os
from contextlib import contextmanager


DEFAULT_GUARDRAIL_NAME = "langchain-demo-basic-guardrail"
DEFAULT_BLOCKED_MESSAGE = (
    "I can't help with that request. This demo can only help with calculator, "
    "project file listing, and approved project file reading."
)

CONTENT_FILTER_TYPES = (
    "SEXUAL",
    "VIOLENCE",
    "HATE",
    "INSULTS",
    "MISCONDUCT",
    "PROMPT_ATTACK",
)

BEDROCK_BEARER_TOKEN_ENV_VAR = "AWS_BEARER_TOKEN_BEDROCK"


@contextmanager
def without_bedrock_bearer_token():
    """
    Temporarily remove the Bedrock API-key env var for guardrail creation.

    This lets boto3 use normal AWS credentials configured with `aws configure`
    instead of an expired OpenAI-compatible Bedrock bearer token.
    """
    bedrock_bearer_token = os.environ.pop(BEDROCK_BEARER_TOKEN_ENV_VAR, None)

    try:
        yield
    finally:
        if bedrock_bearer_token is not None:
            os.environ[BEDROCK_BEARER_TOKEN_ENV_VAR] = bedrock_bearer_token


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an Amazon Bedrock Guardrail for the LangChain demo."
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_GUARDRAIL_NAME,
        help="Name for the Bedrock Guardrail.",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
        help="AWS region where the guardrail should be created.",
    )
    parser.add_argument(
        "--blocked-message",
        default=DEFAULT_BLOCKED_MESSAGE,
        help="Message returned when Bedrock Guardrails blocks input or output.",
    )
    return parser.parse_args()


def build_content_filters():
    filters = []

    for filter_type in CONTENT_FILTER_TYPES:
        filter_config = {
            "type": filter_type,
            "inputStrength": "HIGH",
            "outputStrength": "HIGH",
            "inputAction": "BLOCK",
            "outputAction": "BLOCK",
            "inputEnabled": True,
            "outputEnabled": True,
        }

        if filter_type == "PROMPT_ATTACK":
            filter_config.update(
                {
                    "outputStrength": "NONE",
                    "outputAction": "NONE",
                    "outputEnabled": False,
                }
            )

        filters.append(filter_config)

    return filters


def create_guardrail(name: str, region: str | None, blocked_message: str):
    with without_bedrock_bearer_token():
        import boto3

        bedrock = boto3.client("bedrock", region_name=region)
        guardrail = bedrock.create_guardrail(
            name=name,
            description=(
                "Guardrail for the LangChain demo. Blocks abusive language, "
                "harmful requests, sexual content, violence, hate, misconduct, "
                "and prompt attacks."
            ),
            contentPolicyConfig={
                "filtersConfig": build_content_filters(),
            },
            wordPolicyConfig={
                "managedWordListsConfig": [
                    {
                        "type": "PROFANITY",
                        "inputAction": "BLOCK",
                        "outputAction": "BLOCK",
                        "inputEnabled": True,
                        "outputEnabled": True,
                    }
                ]
            },
            blockedInputMessaging=blocked_message,
            blockedOutputsMessaging=blocked_message,
        )
        version = bedrock.create_guardrail_version(
            guardrailIdentifier=guardrail["guardrailId"],
            description="Initial version for the LangChain demo guardrail.",
        )

    return {
        "guardrail_id": guardrail["guardrailId"],
        "guardrail_version": version["version"],
        "region": region,
    }


def main() -> int:
    args = parse_args()
    result = create_guardrail(args.name, args.region, args.blocked_message)

    print("Created Bedrock Guardrail.")
    print(f"BEDROCK_GUARDRAIL_ID={result['guardrail_id']}")
    print(f"BEDROCK_GUARDRAIL_VERSION={result['guardrail_version']}")
    if result["region"]:
        print(f"BEDROCK_GUARDRAIL_REGION={result['region']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
