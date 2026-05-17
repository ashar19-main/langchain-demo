import re
from dataclasses import dataclass


SUPPORTED_CAPABILITIES = (
    "calculator",
    "project file listing",
    "approved project file reading",
    "royalty-free image search",
)

DEFAULT_REFUSAL_MESSAGE = (
    "I can't help with that request. This demo can only help with calculator, "
    "project file listing, approved project file reading, and royalty-free "
    "image search."
)

ABUSIVE_LANGUAGE_REFUSAL_MESSAGE = (
    "Kindly refrain from using abusive language while submitting your query or "
    "request."
)


@dataclass(frozen=True)
class InputClassification:
    allowed: bool
    category: str
    reason: str
    refusal_message: str = ""


UNSAFE_PATTERNS = (
    (
        "sexual_content",
        (
            r"\bsexually explicit\b",
            r"\bporn\b",
            r"\bpornographic\b",
            r"\berotic\b",
            r"\bnude\b",
            r"\bnudity\b",
        ),
        "The request appears to ask for sexually explicit content.",
    ),
    (
        "abuse_or_harassment",
        (
            r"\babuse\b",
            r"\basshole\b",
            r"\bbastard\b",
            r"\bbitch\b",
            r"\bcunt\b",
            r"\bdick\b",
            r"\bfuck\b",
            r"\bfucking\b",
            r"\bharass\b",
            r"\bhate speech\b",
            r"\bmotherfucker\b",
            r"\bracist\b",
            r"\bshit\b",
            r"\bslur\b",
            r"\bthreaten\b",
            r"\buseless\b.*\b(fuck|fucking|cunt|bitch|asshole)\b",
            r"\bkill yourself\b",
        ),
        "The request contains abusive or harassing language.",
        ABUSIVE_LANGUAGE_REFUSAL_MESSAGE,
    ),
    (
        "illegal_or_harmful",
        (
            r"\billegal\b",
            r"\bfraud\b",
            r"\bscam\b",
            r"\bphishing\b",
            r"\bmalware\b",
            r"\bransomware\b",
            r"\bkeylogger\b",
            r"\bexploit\b",
            r"\bhack\b",
            r"\bsteal\b",
            r"\bbypass\b",
            r"\bunauthorized access\b",
        ),
        "The request appears to ask for illegal or harmful assistance.",
    ),
    (
        "secrets_or_private_data",
        (
            r"\bsecret\b",
            r"\btoken\b",
            r"\bpassword\b",
            r"\bcredential\b",
            r"\bprivate key\b",
            r"\bapi key\b",
            r"\bcredit card\b",
            r"\bbank account\b",
            r"(^|\s)\.env(\s|$)",
            r"(^|\s)\.aws(\s|$)",
            r"(^|\s)\.ssh(\s|$)",
        ),
        "The request appears to ask for secrets, credentials, or private data.",
    ),
    (
        "prompt_injection",
        (
            r"\bignore\b.*\b(previous|above|system) instructions\b",
            r"\boverride\b.*\b(previous|above|system) instructions\b",
            r"\b(show|reveal|print|display)\b.*\bsystem prompt\b",
            r"\bdisable\b.*\bguardrails\b",
        ),
        "The request appears to try to bypass the application's instructions.",
    ),
)

SUPPORTED_INTENT_PATTERNS = (
    (
        "calculator",
        (
            r"\bcalculate\b",
            r"\bcalculator\b",
            r"\bmath\b",
            r"\barithmetic\b",
            r"\badd\b",
            r"\bplus\b",
            r"\bsum\b",
            r"\bsubtract\b",
            r"\bminus\b",
            r"\bmultiply\b",
            r"\btimes\b",
            r"\bdivide\b",
            r"\bpercent\b",
            r"\bpercentage\b",
            r"\d+\s*[\+\-\*/]\s*\d+",
        ),
        "The request is within the calculator capability.",
    ),
    (
        "image_search",
        (
            r"\b(search|find|download|get)\b.*\b(image|images|picture|pictures|photo|photos)\b",
            r"\b(image|picture|photo)\b.*\b(search|download)\b",
            r"\bsearch_and_download_image\b",
        ),
        "The request is within the royalty-free image search capability.",
    ),
    (
        "list_project_files",
        (
            r"\blist\b.*\b(files|folders|directories)\b",
            r"\bshow\b.*\b(files|folders|directories)\b",
            r"\binspect\b.*\b(files|folders|directories)\b",
            r"\bwhat files\b",
            r"\bdir\b",
            r"\bls\b",
            r"\blist_project_files\b",
        ),
        "The request is within the project file listing capability.",
    ),
    (
        "read_file",
        (
            r"\bread\b.*\b(file|contents?)\b",
            r"\bshow\b.*\b(file|contents?)\b",
            r"\bdisplay\b.*\b(file|contents?)\b",
            r"\bopen\b.*\b(file|contents?)\b",
            r"\bread_file\b",
        ),
        "The request is within the approved project file reading capability.",
    ),
)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def classify_user_prompt(prompt: str) -> InputClassification:
    """
    Classify whether a user prompt may be sent to the LangChain agent.
    """
    normalized_prompt = prompt.strip().lower()

    if not normalized_prompt:
        return InputClassification(
            allowed=False,
            category="empty_prompt",
            reason="The prompt is empty.",
            refusal_message=DEFAULT_REFUSAL_MESSAGE,
        )

    for unsafe_pattern in UNSAFE_PATTERNS:
        category, patterns, reason = unsafe_pattern[:3]
        refusal_message = (
            unsafe_pattern[3]
            if len(unsafe_pattern) > 3
            else DEFAULT_REFUSAL_MESSAGE
        )
        if _matches_any(normalized_prompt, patterns):
            return InputClassification(
                allowed=False,
                category=category,
                reason=reason,
                refusal_message=refusal_message,
            )

    for category, patterns, reason in SUPPORTED_INTENT_PATTERNS:
        if _matches_any(normalized_prompt, patterns):
            return InputClassification(
                allowed=True,
                category=category,
                reason=reason,
            )

    return InputClassification(
        allowed=False,
        category="unsupported_request",
        reason="The prompt does not match the supported tool capabilities.",
        refusal_message=DEFAULT_REFUSAL_MESSAGE,
    )
