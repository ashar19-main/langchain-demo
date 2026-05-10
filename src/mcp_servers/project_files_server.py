from pathlib import Path

from fastmcp import FastMCP


mcp = FastMCP("project-files")


@mcp.tool
def list_project_files(directory: str = ".") -> str:
    """
    List files and folders in the specified directory.
    Example: '.', 'src', 'runnables'
    """
    try:
        path = Path(directory)

        if not path.exists():
            return f"Directory does not exist: {directory}"

        if not path.is_dir():
            return f"Not a directory: {directory}"

        items = sorted([item.name for item in path.iterdir()])

        if not items:
            return f"No files found in {directory}"

        return "\n".join(items)

    except Exception as ex:
        return f"Error listing files: {ex}"


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False, log_level="ERROR")
