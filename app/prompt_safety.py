"""Helpers for putting untrusted text (PDF content, student answers) inside prompts."""
import re


def neutralize_tags(text: str, tag: str) -> str:
    """Break any `<tag>` / `</tag>` in `text` so it cannot close or reopen the wrapper we put around it.

    "</context>" becomes "< /context>": still readable, but no longer a tag."""
    return re.sub(rf"<(/?)\s*({re.escape(tag)})\b", r"< \1\2", text, flags=re.I)
