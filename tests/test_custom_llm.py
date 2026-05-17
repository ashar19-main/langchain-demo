import pytest

import CustomLangChainLLMWrapper as wrapper
from CustomLangChainLLMWrapper import (
    BedrockOpenAICompatibleLLM,
    create_bedrock_llm,
)


def test_llm_identifying_properties_do_not_include_secret():
    llm = BedrockOpenAICompatibleLLM(
        model="model",
        api_key="secret",
        base_url="https://bedrock.example",
        temperature=0.4,
        max_tokens=42,
    )

    assert llm._llm_type == "bedrock-openai-compatible"
    assert llm._identifying_params == {
        "model": "model",
        "base_url": "https://bedrock.example",
        "temperature": 0.4,
        "max_tokens": 42,
    }


def test_call_invokes_openai_responses_api(monkeypatch):
    fake_openai = FakeOpenAIFactory("model response STOP hidden")
    monkeypatch.setattr(wrapper, "OpenAI", fake_openai)
    llm = BedrockOpenAICompatibleLLM(
        model="model",
        api_key="secret",
        base_url="https://bedrock.example",
        temperature=0.3,
        max_tokens=99,
    )

    result = llm._call("hello", stop=[" STOP"])

    assert result == "model response"
    assert fake_openai.created_with == {
        "api_key": "secret",
        "base_url": "https://bedrock.example",
    }
    assert fake_openai.responses_call == {
        "model": "model",
        "input": [{"role": "user", "content": "hello"}],
        "temperature": 0.3,
        "max_output_tokens": 99,
    }


def test_call_returns_full_text_without_stop_words(monkeypatch):
    monkeypatch.setattr(wrapper, "OpenAI", FakeOpenAIFactory("complete response"))
    llm = BedrockOpenAICompatibleLLM(
        model="model",
        api_key="secret",
        base_url="https://bedrock.example",
    )

    assert llm._call("hello") == "complete response"


def test_create_bedrock_llm_reads_required_environment(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "token")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://bedrock.example")

    llm = create_bedrock_llm("model", temperature=0.1, max_tokens=12)

    assert llm.model == "model"
    assert llm.api_key == "token"
    assert llm.base_url == "https://bedrock.example"
    assert llm.temperature == 0.1
    assert llm.max_tokens == 12


def test_create_bedrock_llm_requires_api_key(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://bedrock.example")

    with pytest.raises(ValueError, match="AWS_BEARER_TOKEN_BEDROCK"):
        create_bedrock_llm("model")


def test_create_bedrock_llm_requires_base_url(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "token")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        create_bedrock_llm("model")


class FakeOpenAIFactory:
    def __init__(self, output_text):
        self.output_text = output_text
        self.created_with = None
        self.responses_call = None

    def __call__(self, **kwargs):
        self.created_with = kwargs
        return self

    @property
    def responses(self):
        return self

    def create(self, **kwargs):
        self.responses_call = kwargs
        return type("Response", (), {"output_text": self.output_text})()
