import mimetypes
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastmcp import FastMCP


TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_API_KEY_ENV_VAR = "TAVILY_API_KEY"
DEFAULT_DOWNLOAD_DIR = Path("downloads") / "tavily_images"
MAX_PROMPT_LENGTH = 300
IMAGE_SEARCH_REFUSAL_MESSAGE = (
    "I can't help search for that image. Please use a non-offensive, "
    "non-sexually-explicit image prompt."
)
SUCCESS_DOWNLOAD_INSTRUCTION = (
    "Find the downloaded image at the image file path above. By default, "
    "images are saved under downloads\\tavily_images in this project."
)
FAILED_DOWNLOAD_INSTRUCTION = (
    "No image was downloaded. Review the reason above, then try again with a "
    "safe image prompt or a different search phrase."
)

UNSAFE_IMAGE_PROMPT_PATTERNS = (
    r"\bsexually explicit\b",
    r"\bporn\b",
    r"\bpornographic\b",
    r"\berotic\b",
    r"\bnude\b",
    r"\bnudity\b",
    r"\bnsfw\b",
    r"\bfuck\b",
    r"\bfucking\b",
    r"\bcunt\b",
    r"\bbitch\b",
    r"\basshole\b",
    r"\bhate speech\b",
    r"\bracist\b",
    r"\bslur\b",
)


mcp = FastMCP("tavily-image-search")


def is_safe_image_prompt(prompt: str) -> bool:
    """
    Return whether the image prompt passes the local image-search guardrail.
    """
    normalized_prompt = prompt.strip().lower()

    if not normalized_prompt or len(normalized_prompt) > MAX_PROMPT_LENGTH:
        return False

    return not any(
        re.search(pattern, normalized_prompt)
        for pattern in UNSAFE_IMAGE_PROMPT_PATTERNS
    )


def build_royalty_free_query(prompt: str) -> str:
    """
    Bias Tavily toward reusable image sources through the search query text.
    """
    return f"royalty-free image creative commons {prompt.strip()}"


def format_image_search_result(
    *,
    search_succeeded: bool,
    download_succeeded: bool,
    reason: str,
    message: str,
    image_file: str = "N/A",
    source_url: str = "N/A",
    instruction: str | None = None,
) -> str:
    """
    Build a consistent, human-readable MCP result for image search requests.
    """
    resolved_instruction = (
        instruction
        if instruction is not None
        else (
            SUCCESS_DOWNLOAD_INSTRUCTION
            if download_succeeded
            else FAILED_DOWNLOAD_INSTRUCTION
        )
    )

    return "\n".join(
        [
            "Image Search Result:",
            f"Search Succeeded: {'yes' if search_succeeded else 'no'}",
            f"Download Succeeded: {'yes' if download_succeeded else 'no'}",
            f"Reason: {reason}",
            f"Message: {message}",
            f"Image File: {image_file}",
            f"Source URL: {source_url}",
            f"Instruction: {resolved_instruction}",
        ]
    )


def extract_first_image_url(tavily_response: dict) -> str | None:
    """
    Extract the first image URL from Tavily search results.
    """
    images = tavily_response.get("images") or []

    for image in images:
        if isinstance(image, str) and image:
            return image

        if isinstance(image, dict):
            image_url = image.get("url") or image.get("image_url")
            if image_url:
                return image_url

    for result in tavily_response.get("results") or []:
        if isinstance(result, dict):
            image_url = result.get("image_url") or result.get("raw_content")
            if image_url and str(image_url).startswith(("http://", "https://")):
                return image_url

    return None


def get_image_extension(image_url: str, content_type: str | None) -> str:
    """
    Resolve a stable image file extension from the response content type or URL.
    """
    if content_type:
        extension = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if extension in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
            return ".jpg" if extension == ".jpeg" else extension

    url_extension = Path(urlparse(image_url).path).suffix.lower()
    if url_extension in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
        return ".jpg" if url_extension == ".jpeg" else url_extension

    return ".jpg"


def download_image(image_url: str, prompt: str, download_dir: Path) -> Path:
    """
    Download an image URL and return the local file path.
    """
    response = httpx.get(
        image_url,
        headers={"User-Agent": "langchain-demo-image-downloader/1.0"},
        follow_redirects=True,
        timeout=30,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if content_type and not content_type.startswith("image/"):
        raise ValueError(f"URL did not return an image ({content_type}).")

    download_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^a-zA-Z0-9]+", "-", prompt.strip()).strip("-").lower()
    safe_stem = safe_stem[:60] or "tavily-image"
    extension = get_image_extension(image_url, content_type)
    image_path = download_dir / f"{safe_stem}{extension}"

    counter = 1
    while image_path.exists():
        image_path = download_dir / f"{safe_stem}-{counter}{extension}"
        counter += 1

    image_path.write_bytes(response.content)
    return image_path


@mcp.tool
def search_and_download_image(prompt: str, download_directory: str = "") -> str:
    """
    Search Tavily for a royalty-free image matching the prompt and download the
    first image result.
    """
    if not is_safe_image_prompt(prompt):
        return format_image_search_result(
            search_succeeded=False,
            download_succeeded=False,
            reason="blocked_by_image_prompt_guardrail",
            message=IMAGE_SEARCH_REFUSAL_MESSAGE,
        )

    api_key = os.getenv(TAVILY_API_KEY_ENV_VAR)
    if not api_key:
        return format_image_search_result(
            search_succeeded=False,
            download_succeeded=False,
            reason="missing_tavily_api_key",
            message=f"Missing environment variable: {TAVILY_API_KEY_ENV_VAR}",
        )

    target_dir = Path(download_directory) if download_directory else DEFAULT_DOWNLOAD_DIR

    try:
        response = httpx.post(
            TAVILY_SEARCH_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "query": build_royalty_free_query(prompt),
                "search_depth": "basic",
                "include_images": True,
                "include_image_descriptions": True,
                "max_results": 5,
            },
            timeout=30,
        )
        response.raise_for_status()

    except Exception as ex:
        return format_image_search_result(
            search_succeeded=False,
            download_succeeded=False,
            reason="search_failed",
            message=f"Error searching Tavily: {ex}",
        )

    image_url = extract_first_image_url(response.json())
    if not image_url:
        return format_image_search_result(
            search_succeeded=True,
            download_succeeded=False,
            reason="no_image_results",
            message="No image results were returned by Tavily.",
        )

    try:
        image_path = download_image(image_url, prompt, target_dir).resolve()
    except Exception as ex:
        return format_image_search_result(
            search_succeeded=True,
            download_succeeded=False,
            reason="download_failed",
            message=f"Error downloading the first Tavily image result: {ex}",
            source_url=image_url,
        )

    return format_image_search_result(
        search_succeeded=True,
        download_succeeded=True,
        reason="completed",
        message="Downloaded the first Tavily image result.",
        image_file=str(image_path),
        source_url=image_url,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False, log_level="ERROR")
