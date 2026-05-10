# langchain-demo

Simple Python demo for learning agentic AI development with LangChain and Amazon Bedrock through an OpenAI-compatible API.

## What is included

- A direct LLM demo using a custom LangChain `LLM` wrapper.
- A simple LangChain agent demo using `ChatOpenAI`.
- Local tools for basic math, file listing, and file reading.
- Windows command wrappers under `runnables/`.

## Project layout

```text
.
├── runnables/
│   ├── agentdemo.cmd
│   └── llmdemo.cmd
├── src/
│   ├── CustomLangChainLLMWrapper.py
│   ├── agentdemo.py
│   ├── llmdemo.py
│   └── tools.py
├── poetry.lock
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.11 or later
- Poetry
- Access to Amazon Bedrock with an OpenAI-compatible endpoint

## Environment variables

Set these variables before running the demos:

```powershell
$env:AWS_BEARER_TOKEN_BEDROCK = "your-bedrock-api-token"
$env:OPENAI_BASE_URL = "your-bedrock-openai-compatible-base-url"
```

## Install dependencies

```powershell
poetry install
```

## Run the direct LLM demo

```powershell
runnables\llmdemo.cmd -p "Explain LangChain in three simple bullet points."
```

## Run the agent demo

```powershell
runnables\agentdemo.cmd -p "List the project files and explain what this app does."
```

## Notes

This is a learning project. The included file tools and calculator are intentionally simple and should be tightened before use in a production application.
