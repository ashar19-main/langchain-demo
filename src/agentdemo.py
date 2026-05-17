import argparse
import asyncio
import ast
import os
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from guardrails import (
    build_agent_system_prompt,
    classify_user_prompt,
    evaluate_prompt_with_bedrock_guardrail,
)
from tools import get_local_tools


DEFAULT_MODEL = "openai.gpt-oss-20b"
MAX_TOOL_ARG_CHARS = 500
MAX_TOOL_RESULT_CHARS = 1200
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TAVILY_API_KEY_ENV_VAR = "TAVILY_API_KEY"
BEDROCK_TOKEN_ERROR_MESSAGE = (
    "The Amazon Bedrock bearer token appears to be expired or invalid. "
    "Refresh AWS_BEARER_TOKEN_BEDROCK, confirm OPENAI_BASE_URL is correct, "
    "and rerun the command."
)


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


def get_tavily_mcp_env() -> dict[str, str]:
    """
    Return environment variables that must be passed to the Tavily MCP process.

    The MCP stdio client only inherits a small safe-list of environment
    variables by default, so app-specific secrets must be passed explicitly.
    """
    tavily_api_key = os.getenv(TAVILY_API_KEY_ENV_VAR, "")
    return {TAVILY_API_KEY_ENV_VAR: tavily_api_key}


async def get_mcp_tools():
    client = MultiServerMCPClient(
        {
            "project_files": {
                "command": sys.executable,
                "args": ["src/mcp_servers/project_files_server.py"],
                "transport": "stdio",
                "cwd": PROJECT_ROOT,
            },
            "tavily_image_search": {
                "command": sys.executable,
                "args": ["src/mcp_servers/tavily_image_search_server.py"],
                "transport": "stdio",
                "cwd": PROJECT_ROOT,
                "env": get_tavily_mcp_env(),
            },
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


def extract_tool_results(messages):
    """
    Extract tool result messages returned by the agent.
    """
    tool_results = []

    for message in messages:
        message_type = getattr(message, "type", "")
        tool_call_id = getattr(message, "tool_call_id", "")
        content = getattr(message, "content", "")

        if message_type == "tool" or tool_call_id:
            tool_results.append(
                {
                    "name": getattr(message, "name", "") or "tool",
                    "content": content,
                }
            )

    return tool_results


def normalize_tool_result_content(content) -> str:
    """
    Return readable text from LangChain/MCP tool result content.
    """
    if isinstance(content, list):
        text_items = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        if text_items:
            return "\n".join(text_items)

    if isinstance(content, str):
        try:
            parsed_content = ast.literal_eval(content)
        except (SyntaxError, ValueError):
            return content

        if isinstance(parsed_content, list):
            return normalize_tool_result_content(parsed_content)

    return str(content or "")


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


def print_tool_result_summary(tool_results):
    print("\nTool Results:")
    print("-" * 60)

    if not tool_results:
        print("No tool result was returned.")
        print("-" * 60)
        return

    for index, tool_result in enumerate(tool_results, start=1):
        print(f"{index}. {tool_result['name']}")

        content = normalize_tool_result_content(tool_result.get("content"))
        if len(content) > MAX_TOOL_RESULT_CHARS:
            content = content[:MAX_TOOL_RESULT_CHARS] + "... [truncated]"

        if content:
            print(content)

    print("-" * 60)


def print_guardrail_summary(source: str, category: str):
    print("\nGuardrail:")
    print("-" * 60)
    print(f"Source: {source}")
    print(f"Category: {category}")
    print("-" * 60)


def format_agent_error(ex: Exception) -> str:
    """
    Return a user-friendly message for common runtime errors.
    """
    raw_error = str(ex)
    normalized_error = raw_error.lower()

    if (
        "invalid_api_key" in normalized_error
        or "signature expired" in normalized_error
        or "permission_denied_error" in normalized_error
    ):
        return f"{BEDROCK_TOKEN_ERROR_MESSAGE}\n\nRaw error: {raw_error}"

    return raw_error


async def run_agent_demo() -> int:
    args = parse_args()

    try:
        bedrock_guardrail_classification = evaluate_prompt_with_bedrock_guardrail(
            args.prompt
        )
        if bedrock_guardrail_classification:
            print_tool_call_summary([])
            print_tool_result_summary([])
            print_guardrail_summary(
                "bedrock",
                bedrock_guardrail_classification.category,
            )

            print("\nFinal Agent Response:")
            print("-" * 60)
            print(bedrock_guardrail_classification.refusal_message)
            print("-" * 60)

            return 0

        prompt_classification = classify_user_prompt(args.prompt)
        if not prompt_classification.allowed:
            print_tool_call_summary([])
            print_tool_result_summary([])
            print_guardrail_summary("local_regex", prompt_classification.category)

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
        tool_results = extract_tool_results(result["messages"])

        print_tool_call_summary(tool_calls)
        print_tool_result_summary(tool_results)

        print("\nFinal Agent Response:")
        print("-" * 60)
        print(final_message.content)
        print("-" * 60)

        return 0

    except Exception as ex:
        print("\nERROR:")
        print(format_agent_error(ex))
        return 1


def main() -> int:
    return asyncio.run(run_agent_demo())


if __name__ == "__main__":
    sys.exit(main())
