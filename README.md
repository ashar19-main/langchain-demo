# langchain-demo

Simple Python demo for learning agentic AI development with LangChain and Amazon Bedrock through an OpenAI-compatible API.

## What is included

- A direct LLM demo using a custom LangChain `LLM` wrapper.
- A simple LangChain agent demo using `ChatOpenAI`.
- Local tools for basic math and file reading.
- FastMCP servers for project file listing and Tavily image search/downloads.
- Reusable guardrails for system prompting, prompt classification, and optional Amazon Bedrock Guardrails integration.
- Windows command wrappers under `runnables/`.

## Project layout

```text
.
|-- runnables/
|   |-- agentdemo.cmd
|   `-- llmdemo.cmd
|-- src/
|   |-- guardrails/
|   |-- mcp_servers/
|   |-- tools/
|   |-- CustomLangChainLLMWrapper.py
|   |-- agentdemo.py
|   `-- llmdemo.py
|-- poetry.lock
|-- pyproject.toml
`-- README.md
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
$env:TAVILY_API_KEY = "your-tavily-api-key"
```

If you are running the `.cmd` wrapper from Command Prompt instead of
PowerShell, set the variables in the same `cmd.exe` window:

```cmd
set AWS_BEARER_TOKEN_BEDROCK=your-bedrock-api-token
set OPENAI_BASE_URL=your-bedrock-openai-compatible-base-url
set TAVILY_API_KEY=your-tavily-api-key
```

Optional Bedrock Guardrails variables:

```powershell
$env:BEDROCK_GUARDRAIL_ID = "your-guardrail-id"
$env:BEDROCK_GUARDRAIL_VERSION = "your-guardrail-version"
$env:BEDROCK_GUARDRAIL_REGION = "your-guardrail-region"
```

When these variables are set, `agentdemo.py` evaluates the prompt with Amazon Bedrock Guardrails before running the local prompt classifier or invoking the agent.

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

Search for a royalty-free image and download the first Tavily image result:

```powershell
runnables\agentdemo.cmd -p "Find and download an image of a mountain lake at sunrise."
```

## Create a Bedrock Guardrail

The helper below creates a Bedrock Guardrail with high-strength content filters for harmful categories and a managed profanity word list:

```powershell
poetry run python src\guardrails\create_bedrock_guardrail.py --region ap-south-1
```

Copy the printed `BEDROCK_GUARDRAIL_ID`, `BEDROCK_GUARDRAIL_VERSION`, and `BEDROCK_GUARDRAIL_REGION` values into your environment before running the agent.

## Run tests and view the report

Run the unit test suite with the configured 90% coverage gate:

```powershell
poetry run pytest
```

Generate a standalone HTML test report with pass/fail details and coverage by file:

```powershell
poetry run python scripts\test_report_visualizer.py
```

Open the generated report at `reports\test-report\index.html`.

## Notes

This is a learning project. The included file tools, calculator, MCP servers, and guardrails are intentionally simple and should be tightened before use in a production application.
