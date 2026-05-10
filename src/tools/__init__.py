from tools.calculator import calculator
from tools.list_project_files import list_project_files
from tools.read_file import read_file


def get_tools():
    """
    Return all tools to be registered with the agent.
    """
    return [
        calculator,
        list_project_files,
        read_file,
    ]
