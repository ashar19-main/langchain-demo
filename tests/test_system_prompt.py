from guardrails.system_prompt import (
    AGENT_ROLE_PROMPT,
    SAFETY_SYSTEM_PROMPT,
    TOOL_USE_PROMPT,
    build_agent_system_prompt,
)


def test_build_agent_system_prompt_combines_prompt_sections_in_order():
    prompt = build_agent_system_prompt()

    expected = "\n".join(
        [
            AGENT_ROLE_PROMPT.strip(),
            SAFETY_SYSTEM_PROMPT.strip(),
            TOOL_USE_PROMPT.strip(),
        ]
    )
    assert prompt == expected
    assert "calculator" in prompt
    assert "list_project_files" in prompt
    assert "read_file" in prompt
    assert "search_and_download_image" in prompt
    assert "royalty-free" in prompt
