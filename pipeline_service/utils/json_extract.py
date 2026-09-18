from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(raw: str) -> str:
    """Return the JSON object substring from `raw`, or `raw` itself if no match.
    """
    raw = raw.strip()
    raw = _THINK_BLOCK.sub("", raw).strip()
    fenced = _JSON_FENCE.search(raw)
    if fenced:
        raw = fenced.group(1).strip()
    obj = _JSON_OBJECT.search(raw)
    return obj.group(0) if obj else raw
