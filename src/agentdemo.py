import argparse
import os
import sys

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from tools import get_tools


DEFAULT_MODEL = "openai.gpt-oss-20b"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple LangChain Agent demo using Amazon Bedrock OpenAI-compatible API."
    )

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=True,
        help="User prompt for the agent.",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Amazon Bedrock model ID from AWS Quick Start.",
    )

    return parser.parse_args()


def create_bedrock_chat_model(model: str):
    api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("Missing environment variable: AWS_BEARER_TOKEN_BEDROCK")

    if not base_url:
        raise ValueError("Missing environment variable: OPENAI_BASE_URL")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.2,
        max_tokens=800,
    )


def create_agent_runner(model: str):
    chat_model = create_bedrock_chat_model(model)
    tools = get_tools()

    agent = create_agent(
        model=chat_model,
        tools=tools,
        system_prompt="""
You are a simple helpful Dev Helper Agent.

You can:
- answer simple questions directly
- use calculator for math
- use list_project_files to inspect folders
- use read_file to read text/code files

Use tools only when needed.
Keep answers simple and clear.
""",
    )

    return agent


def main() -> int:
    args = parse_args()

    try:
        agent = create_agent_runner(args.model)

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": args.prompt,
                    }
                ]
            }
        )

        final_message = result["messages"][-1]

        print("\nFinal Agent Response:")
        print("-" * 60)
        print(final_message.content)
        print("-" * 60)

        return 0

    except Exception as ex:
        print("\nERROR:")
        print(str(ex))
        return 1


if __name__ == "__main__":
    sys.exit(main())