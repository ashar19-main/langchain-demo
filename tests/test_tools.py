from tools import get_local_tools
from tools.calculator import calculator
from tools.read_file import read_file


def test_calculator_returns_expression_result():
    assert calculator.invoke({"expression": "25 * 19"}) == "475"


def test_calculator_returns_error_for_invalid_expression():
    result = calculator.invoke({"expression": "missing_name + 1"})

    assert result.startswith("Error evaluating expression:")


def test_read_file_returns_file_contents(tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("hello from a test file", encoding="utf-8")

    assert read_file.invoke({"file_path": str(target)}) == "hello from a test file"


def test_read_file_reports_missing_file(tmp_path):
    missing = tmp_path / "missing.txt"

    assert read_file.invoke({"file_path": str(missing)}) == f"File does not exist: {missing}"


def test_read_file_reports_directory(tmp_path):
    assert read_file.invoke({"file_path": str(tmp_path)}) == f"Not a file: {tmp_path}"


def test_read_file_truncates_large_files(tmp_path):
    target = tmp_path / "large.txt"
    target.write_text("x" * 5001, encoding="utf-8")

    result = read_file.invoke({"file_path": str(target)})

    assert result == ("x" * 5000) + "\n\n[Content truncated]"


def test_read_file_returns_read_errors(monkeypatch, tmp_path):
    target = tmp_path / "sample.txt"
    target.write_text("hello", encoding="utf-8")

    def raise_read_error(*args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr("pathlib.Path.read_text", raise_read_error)

    assert read_file.invoke({"file_path": str(target)}) == "Error reading file: nope"


def test_get_local_tools_returns_demo_tools():
    assert get_local_tools() == [calculator, read_file]
