from tcrb.v02.agent import (
    RECOVERY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    resolve_system_prompt,
)


def test_default_prompt_selection_preserves_builtin_prompt():
    assert resolve_system_prompt(None, "default") == SYSTEM_PROMPT
    assert resolve_system_prompt("", "default") == SYSTEM_PROMPT


def test_recovery_prompt_selection_returns_recovery_instructions():
    prompt = resolve_system_prompt(None, "recovery")

    assert prompt == RECOVERY_SYSTEM_PROMPT
    assert "retry" in prompt.lower()
    assert "fallback" in prompt.lower()
    assert "do not repeat an unchanged failed call" in prompt.lower()


def test_custom_prompt_takes_precedence_over_prompt_variant():
    assert resolve_system_prompt("custom instructions", "recovery") == "custom instructions"


def test_unknown_prompt_variant_is_rejected():
    try:
        resolve_system_prompt(None, "unknown")
    except ValueError as exc:
        assert "unknown prompt variant" in str(exc)
    else:
        raise AssertionError("unknown prompt variants must be rejected")
