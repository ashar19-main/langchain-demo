import argparse
import sys

from CustomLangChainLLMWrapper import create_bedrock_llm


DEFAULT_MODEL = "openai.gpt-oss-20b"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo CLI program to call Amazon Bedrock through LangChain."
    )

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="Explain in 3 simple bullet points what LangChain does in an agentic AI application.",
        help="Prompt to send to the LLM.",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model ID from Amazon Bedrock Quick Start.",
    )

    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for model response.",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum output tokens.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        llm = create_bedrock_llm(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        response = llm.invoke(args.prompt)

        print("\nLLM Response:")
        print("-" * 60)
        print(response)
        print("-" * 60)

        return 0

    except Exception as ex:
        print("\nERROR:")
        print(str(ex))
        return 1


if __name__ == "__main__":
    sys.exit(main())