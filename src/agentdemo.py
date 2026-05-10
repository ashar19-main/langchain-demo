import argparse
import asyncio
import os
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from guardrails import build_agent_system_prompt, classify_user_prompt
from tools import get_local_tools


DEFAULT_MODEL = "openai.gpt-oss-20b"
MAX_TOOL_ARG_CHARS = 500
PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


async def get_mcp_tools():
    client = MultiServerMCPClient(
        {
            "project_files": {
                "command": sys.executable,
                "args": ["src/mcp_servers/project_files_server.py"],
                "transport": "stdio",
                "cwd": PROJECT_ROOT,
            }
        },
        tool_name_prefix=False,
    )

    return await client.get_tools()


async def get_agent_tools():
    return get_local_tools() + await get_mcp_tools()


async def create_agent_runner(model: str):
    chat_model = create_bedrock_chat_model(model)
    tools = await get_agent_tools()

    agent = create_agent(
        model=chat_model,
        tools=tools,
        system_prompt=build_agent_system_prompt(),
    )

    return agent


def format_tool_args(args) -> str:
    if args is None:
        return ""

    text = str(args)
    if len(text) > MAX_TOOL_ARG_CHARS:
        return text[:MAX_TOOL_ARG_CHARS] + "... [truncated]"

    return text


def extract_tool_calls(messages):
    """
    Extract tool calls from LangChain messages returned by the agent.
    """
    tool_calls = []
    seen_tool_call_ids = set()

    for message in messages:
        for tool_call in getattr(message, "tool_calls", []) or []:
            tool_call_id = tool_call.get("id")
            if tool_call_id and tool_call_id in seen_tool_call_ids:
                continue

            name = tool_call.get("name")
            args = tool_call.get("args")

            if name:
                tool_calls.append(
                    {
                        "id": tool_call_id,
                        "name": name,
                        "args": args,
                    }
                )

            if tool_call_id:
                seen_tool_call_ids.add(tool_call_id)

        additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
        for raw_tool_call in additional_kwargs.get("tool_calls", []) or []:
            tool_call_id = raw_tool_call.get("id")
            if tool_call_id and tool_call_id in seen_tool_call_ids:
                continue

            function_call = raw_tool_call.get("function", {})
            name = function_call.get("name")
            args = function_call.get("arguments")

            if name:
                tool_calls.append(
                    {
                        "id": tool_call_id,
                        "name": name,
                        "args": args,
                    }
                )

            if tool_call_id:
                seen_tool_call_ids.add(tool_call_id)

    return tool_calls


def print_tool_call_summary(tool_calls):
    print("\nTool Calls:")
    print("-" * 60)

    if not tool_calls:
        print("No tool was called.")
        print("-" * 60)
        return

    for index, tool_call in enumerate(tool_calls, start=1):
        print(f"{index}. {tool_call['name']}")

        formatted_args = format_tool_args(tool_call.get("args"))
        if formatted_args:
            print(f"   args: {formatted_args}")

    print("-" * 60)


async def run_agent_demo() -> int:
    args = parse_args()

    try:
        prompt_classification = classify_user_prompt(args.prompt)
        if not prompt_classification.allowed:
            print_tool_call_summary([])

            print("\nFinal Agent Response:")
            print("-" * 60)
            print(prompt_classification.refusal_message)
            print("-" * 60)

            return 0

        agent = await create_agent_runner(args.model)

        result = await agent.ainvoke(
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
        tool_calls = extract_tool_calls(result["messages"])

        print_tool_call_summary(tool_calls)

        print("\nFinal Agent Response:")
        print("-" * 60)
        print(final_message.content)
        print("-" * 60)

        return 0

    except Exception as ex:
        print("\nERROR:")
        print(str(ex))
        return 1


def main() -> int:
    return asyncio.run(run_agent_demo())


if __name__ == "__main__":
    sys.exit(main())
