from pathlib import Path

from langchain_core.tools import tool


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

        # Limit content size for the demo.
        max_chars = 5000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[Content truncated]"

        return content

    except Exception as ex:
        return f"Error reading file: {ex}"
