"""Shared JSON-extractor used by every provider's chat_json implementation."""
from __future__ import annotations

import json
from typing import Any


def extract_json(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model response")
    if text.startswith("```"):
        text = text.strip("`")
        first_nl = text.find("\n")
        if first_nl != -1:
            head = text[:first_nl].strip().lower()
            if head in {"json", "javascript", ""}:
                text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            end = text.rfind(closer)
            if start != -1 and end > start:
                return json.loads(text[start:end + 1])
        raise
