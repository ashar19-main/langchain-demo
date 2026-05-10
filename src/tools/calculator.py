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
