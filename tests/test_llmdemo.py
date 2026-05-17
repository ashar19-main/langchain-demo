import argparse

import llmdemo


def test_parse_args_reads_cli_values(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "llmdemo.py",
            "--prompt",
            "hello",
            "--model",
            "custom-model",
            "--temperature",
            "0.9",
            "--max-tokens",
            "321",
        ],
    )

    args = llmdemo.parse_args()

    assert args.prompt == "hello"
    assert args.model == "custom-model"
    assert args.temperature == 0.9
    assert args.max_tokens == 321


def test_main_invokes_llm_and_prints_response(monkeypatch, capsys):
    monkeypatch.setattr(
        llmdemo,
        "parse_args",
        lambda: argparse.Namespace(
            prompt="hello",
            model="model",
            temperature=0.7,
            max_tokens=123,
        ),
    )
    fake_factory = FakeLLMFactory("response text")
    monkeypatch.setattr(llmdemo, "create_bedrock_llm", fake_factory)

    assert llmdemo.main() == 0

    assert fake_factory.kwargs == {
        "model": "model",
        "temperature": 0.7,
        "max_tokens": 123,
    }
    assert "response text" in capsys.readouterr().out


def test_main_returns_error_code_on_exception(monkeypatch, capsys):
    monkeypatch.setattr(
        llmdemo,
        "parse_args",
        lambda: argparse.Namespace(
            prompt="hello",
            model="model",
            temperature=0.7,
            max_tokens=123,
        ),
    )

    def raise_error(**kwargs):
        raise RuntimeError("no model")

    monkeypatch.setattr(llmdemo, "create_bedrock_llm", raise_error)

    assert llmdemo.main() == 1

    assert "no model" in capsys.readouterr().out


class FakeLLMFactory:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self

    def invoke(self, prompt):
        assert prompt == "hello"
        return self.response
