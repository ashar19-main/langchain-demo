import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from langchain_core.language_models.llms import LLM


class BedrockOpenAICompatibleLLM(LLM):
    """
    Custom LangChain LLM wrapper around Amazon Bedrock's
    OpenAI-compatible API endpoint.
    """

    model: str
    api_key: str
    base_url: str
    temperature: float = 0.2
    max_tokens: int = 500

    @property
    def _llm_type(self) -> str:
        return "bedrock-openai-compatible"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        response = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        output_text = response.output_text

        if stop:
            for stop_word in stop:
                output_text = output_text.split(stop_word)[0]

        return output_text


def create_bedrock_llm(
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 500,
) -> BedrockOpenAICompatibleLLM:
    """
    Factory function to create the Bedrock LangChain LLM wrapper
    using environment variables.
    """

    api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError(
            "Missing environment variable: AWS_BEARER_TOKEN_BEDROCK"
        )

    if not base_url:
        raise ValueError(
            "Missing environment variable: OPENAI_BASE_URL"
        )

    return BedrockOpenAICompatibleLLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )