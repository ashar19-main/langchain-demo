from pathlib import Path

from langchain_core.tools import tool


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
