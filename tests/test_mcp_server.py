from mcp_servers.project_files_server import list_project_files


def test_list_project_files_returns_sorted_entries(tmp_path):
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")

    assert list_project_files(str(tmp_path)) == "a.txt\nb.txt"


def test_list_project_files_reports_missing_directory(tmp_path):
    missing = tmp_path / "missing"

    assert list_project_files(str(missing)) == f"Directory does not exist: {missing}"


def test_list_project_files_reports_file_path(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("content", encoding="utf-8")

    assert list_project_files(str(file_path)) == f"Not a directory: {file_path}"


def test_list_project_files_reports_empty_directory(tmp_path):
    assert list_project_files(str(tmp_path)) == f"No files found in {tmp_path}"


def test_list_project_files_returns_errors(monkeypatch, tmp_path):
    def raise_iterdir(_self):
        raise OSError("cannot list")

    monkeypatch.setattr("pathlib.Path.iterdir", raise_iterdir)

    assert list_project_files(str(tmp_path)) == "Error listing files: cannot list"
