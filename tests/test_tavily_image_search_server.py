from pathlib import Path

import pytest

from mcp_servers import tavily_image_search_server as tavily


def test_is_safe_image_prompt_blocks_empty_long_and_explicit_prompts():
    assert tavily.is_safe_image_prompt("mountain lake at sunrise")
    assert not tavily.is_safe_image_prompt("   ")
    assert not tavily.is_safe_image_prompt("x" * (tavily.MAX_PROMPT_LENGTH + 1))
    assert not tavily.is_safe_image_prompt("download a porn image")
    assert not tavily.is_safe_image_prompt("find an offensive racist image")


def test_build_royalty_free_query_adds_reuse_terms():
    assert tavily.build_royalty_free_query("forest trail") == (
        "royalty-free image creative commons forest trail"
    )


def test_format_image_search_result_uses_consistent_success_structure():
    result = tavily.format_image_search_result(
        search_succeeded=True,
        download_succeeded=True,
        reason="completed",
        message="Downloaded the first Tavily image result.",
        image_file="downloads\\tavily_images\\lake.jpg",
        source_url="https://example.com/lake.jpg",
    )

    assert result == "\n".join(
        [
            "Image Search Result:",
            "Search Succeeded: yes",
            "Download Succeeded: yes",
            "Reason: completed",
            "Message: Downloaded the first Tavily image result.",
            "Image File: downloads\\tavily_images\\lake.jpg",
            "Source URL: https://example.com/lake.jpg",
            f"Instruction: {tavily.SUCCESS_DOWNLOAD_INSTRUCTION}",
        ]
    )


def test_format_image_search_result_uses_consistent_failure_structure():
    result = tavily.format_image_search_result(
        search_succeeded=False,
        download_succeeded=False,
        reason="missing_tavily_api_key",
        message="Missing environment variable: TAVILY_API_KEY",
    )

    assert "Search Succeeded: no" in result
    assert "Download Succeeded: no" in result
    assert "Image File: N/A" in result
    assert "Source URL: N/A" in result
    assert f"Instruction: {tavily.FAILED_DOWNLOAD_INSTRUCTION}" in result


def test_extract_first_image_url_handles_string_and_dict_images():
    assert tavily.extract_first_image_url({"images": ["https://example.com/a.jpg"]}) == (
        "https://example.com/a.jpg"
    )
    assert tavily.extract_first_image_url(
        {"images": [{"url": "https://example.com/b.png"}]}
    ) == "https://example.com/b.png"
    assert tavily.extract_first_image_url(
        {"images": [{"image_url": "https://example.com/c.webp"}]}
    ) == "https://example.com/c.webp"


def test_extract_first_image_url_falls_back_to_result_image_url():
    response = {"results": [{"image_url": "https://example.com/result.jpg"}]}

    assert tavily.extract_first_image_url(response) == "https://example.com/result.jpg"


def test_extract_first_image_url_returns_none_when_missing():
    assert tavily.extract_first_image_url({"results": [{"title": "No image"}]}) is None


@pytest.mark.parametrize(
    ("url", "content_type", "extension"),
    [
        ("https://example.com/image", "image/png", ".png"),
        ("https://example.com/image.jpeg", "", ".jpg"),
        ("https://example.com/image.unknown", "", ".jpg"),
    ],
)
def test_get_image_extension(url, content_type, extension):
    assert tavily.get_image_extension(url, content_type) == extension


def test_download_image_writes_unique_image_files(monkeypatch, tmp_path):
    fake_get = FakeGet()
    monkeypatch.setattr(tavily.httpx, "get", fake_get)

    first_path = tavily.download_image(
        "https://example.com/image.png",
        "Mountain Lake!",
        tmp_path,
    )
    second_path = tavily.download_image(
        "https://example.com/image.png",
        "Mountain Lake!",
        tmp_path,
    )

    assert first_path == tmp_path / "mountain-lake.png"
    assert second_path == tmp_path / "mountain-lake-1.png"
    assert first_path.read_bytes() == b"image-bytes"
    assert second_path.read_bytes() == b"image-bytes"
    assert fake_get.kwargs["headers"] == {
        "User-Agent": "langchain-demo-image-downloader/1.0"
    }


def test_download_image_rejects_non_image_content(monkeypatch, tmp_path):
    monkeypatch.setattr(tavily.httpx, "get", lambda *args, **kwargs: FakeTextResponse())

    with pytest.raises(ValueError, match="URL did not return an image"):
        tavily.download_image("https://example.com/page", "page", tmp_path)


def test_search_and_download_image_refuses_unsafe_prompt(monkeypatch):
    monkeypatch.delenv(tavily.TAVILY_API_KEY_ENV_VAR, raising=False)

    result = tavily.search_and_download_image("porn image")

    assert "Search Succeeded: no" in result
    assert "Download Succeeded: no" in result
    assert "Reason: blocked_by_image_prompt_guardrail" in result
    assert tavily.IMAGE_SEARCH_REFUSAL_MESSAGE in result


def test_search_and_download_image_requires_api_key(monkeypatch):
    monkeypatch.delenv(tavily.TAVILY_API_KEY_ENV_VAR, raising=False)

    result = tavily.search_and_download_image("mountain lake")

    assert "Search Succeeded: no" in result
    assert "Download Succeeded: no" in result
    assert "Reason: missing_tavily_api_key" in result
    assert "Missing environment variable: TAVILY_API_KEY" in result


def test_search_and_download_image_posts_to_tavily_and_downloads(
    monkeypatch,
    tmp_path,
):
    fake_post = FakePost()
    monkeypatch.setenv(tavily.TAVILY_API_KEY_ENV_VAR, "token")
    monkeypatch.setattr(tavily.httpx, "post", fake_post)
    monkeypatch.setattr(
        tavily,
        "download_image",
        lambda image_url, prompt, download_dir: Path(download_dir) / "image.jpg",
    )

    result = tavily.search_and_download_image("mountain lake", str(tmp_path))

    assert fake_post.kwargs["json"] == {
        "query": "royalty-free image creative commons mountain lake",
        "search_depth": "basic",
        "include_images": True,
        "include_image_descriptions": True,
        "max_results": 5,
    }
    assert fake_post.kwargs["headers"] == {"Authorization": "Bearer token"}
    assert "Search Succeeded: yes" in result
    assert "Download Succeeded: yes" in result
    assert "Reason: completed" in result
    assert f"Image File: {(tmp_path / 'image.jpg').resolve()}" in result
    assert "Source URL: https://example.com/image.jpg" in result
    assert tavily.SUCCESS_DOWNLOAD_INSTRUCTION in result


def test_search_and_download_image_reports_no_images(monkeypatch):
    monkeypatch.setenv(tavily.TAVILY_API_KEY_ENV_VAR, "token")
    monkeypatch.setattr(tavily.httpx, "post", lambda *args, **kwargs: FakeEmptyPostResponse())

    result = tavily.search_and_download_image("mountain lake")

    assert "Search Succeeded: yes" in result
    assert "Download Succeeded: no" in result
    assert "Reason: no_image_results" in result
    assert "No image results were returned by Tavily." in result


def test_search_and_download_image_reports_download_failure(monkeypatch):
    monkeypatch.setenv(tavily.TAVILY_API_KEY_ENV_VAR, "token")
    monkeypatch.setattr(tavily.httpx, "post", lambda *args, **kwargs: FakePostResponse())
    monkeypatch.setattr(
        tavily,
        "download_image",
        lambda image_url, prompt, download_dir: (_ for _ in ()).throw(
            RuntimeError("403 Forbidden")
        ),
    )

    result = tavily.search_and_download_image("mountain lake")

    assert "Search Succeeded: yes" in result
    assert "Download Succeeded: no" in result
    assert "Reason: download_failed" in result
    assert "403 Forbidden" in result
    assert "Source URL: https://example.com/image.jpg" in result


class FakeImageResponse:
    headers = {"content-type": "image/png"}
    content = b"image-bytes"

    def raise_for_status(self):
        return None


class FakeGet:
    def __init__(self):
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.kwargs = kwargs
        return FakeImageResponse()


class FakeTextResponse:
    headers = {"content-type": "text/html"}
    content = b"<html></html>"

    def raise_for_status(self):
        return None


class FakePost:
    def __init__(self):
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.kwargs = kwargs
        return FakePostResponse()


class FakePostResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"images": ["https://example.com/image.jpg"]}


class FakeEmptyPostResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"images": []}
