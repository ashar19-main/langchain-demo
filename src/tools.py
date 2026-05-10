import os
from pathlib import Path

from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a simple mathematical expression.
    Example: '25 * 19'
    """
    try:
        # Safe demo use only. We disable built-ins.
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as ex:
        return f"Error evaluating expression: {ex}"


@tool
def list_project_files(directory: str = ".") -> str:
    """
    List files and folders in the specified directory.
    Example: '.', 'src', 'runnables'
    """
    try:
        path = Path(directory)

        if not path.exists():
            return f"Directory does not exist: {directory}"

        items = sorted([item.name for item in path.iterdir()])

        if not items:
            return f"No files found in {directory}"

        return "\n".join(items)

    except Exception as ex:
        return f"Error listing files: {ex}"


@tool
def read_file(file_path: str) -> str:
    """
    Read and return the contents of a text file.
    Example: 'pyproject.toml', 'src/llmdemo.py'
    """
    try:
        path = Path(file_path)

        if not path.exists():
            return f"File does not exist: {file_path}"

        if not path.is_file():
            return f"Not a file: {file_path}"

        content = path.read_text(encoding="utf-8")

        # Limit content size for the demo
        max_chars = 5000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[Content truncated]"

        return content

    except Exception as ex:
        return f"Error reading file: {ex}"


def get_tools():
    """
    Return all tools to be registered with the agent.
    """
    return [
        calculator,
        list_project_files,
        read_file,
    ]