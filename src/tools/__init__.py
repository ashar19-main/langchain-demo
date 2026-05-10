from tools.calculator import calculator
from tools.read_file import read_file


def get_local_tools():
    """
    Return local tools to be registered with the agent.

    list_project_files is provided by the project-files MCP server.
    """
    return [
        calculator,
        read_file,
    ]
