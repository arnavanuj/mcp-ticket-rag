"""Dynamic payload introspection for MCP live responses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class FieldEntry:
    path: str
    key: str
    value: Any
    value_type: str
    normalized_text_value: str
    parent_path: str


def flatten_payload(payload: Any) -> list[FieldEntry]:
    """Flatten nested dict/list payload into searchable field entries."""

    entries: list[FieldEntry] = []
    _walk(payload=payload, path="", parent_path="", entries=entries)
    return entries


def search_payload_fields(
    payload: Any,
    semantic_target: str | list[str],
    optional_value_filter: str | Callable[[FieldEntry], bool] | None = None,
) -> list[dict[str, Any]]:
    """Search flattened payload by semantic field/path similarity + optional value filter."""

    targets = [semantic_target] if isinstance(semantic_target, str) else semantic_target
    target_tokens = _tokens(" ".join(targets))
    entries = flatten_payload(payload)
    matches: list[dict[str, Any]] = []

    for entry in entries:
        score = _path_score(entry, target_tokens)
        if score <= 0:
            continue

        if optional_value_filter is not None and not _passes_filter(entry, optional_value_filter):
            continue

        matches.append({"score": score, "entry": entry})

    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches


def extract_best_value(
    payload: Any,
    candidate_field_meanings: list[str],
    optional_value_filter: str | Callable[[FieldEntry], bool] | None = None,
) -> dict[str, Any] | None:
    """Return best matching flattened field/value candidate from payload."""

    matches = search_payload_fields(
        payload=payload,
        semantic_target=candidate_field_meanings,
        optional_value_filter=optional_value_filter,
    )
    if not matches:
        return None
    top = matches[0]
    entry: FieldEntry = top["entry"]
    return {
        "path": entry.path,
        "value": entry.value,
        "normalized_text_value": entry.normalized_text_value,
        "score": top["score"],
        "parent_path": entry.parent_path,
    }


def _walk(payload: Any, path: str, parent_path: str, entries: list[FieldEntry]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_str = str(key)
            child_path = f"{path}.{key_str}" if path else key_str
            _walk(payload=value, path=child_path, parent_path=path, entries=entries)
        return

    if isinstance(payload, list):
        for idx, value in enumerate(payload):
            child_path = f"{path}[{idx}]"
            _walk(payload=value, path=child_path, parent_path=path, entries=entries)
        return

    key = path.split(".")[-1] if path else ""
    if "[" in key:
        key = key.split("[", 1)[0]
    entries.append(
        FieldEntry(
            path=path or "$",
            key=key,
            value=payload,
            value_type=type(payload).__name__,
            normalized_text_value=_normalize_value(payload),
            parent_path=parent_path,
        )
    )


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return str(value).lower()
    return str(value).strip().lower()


def _tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9_]+", text.lower()) if tok}


def _path_score(entry: FieldEntry, target_tokens: set[str]) -> float:
    if not target_tokens:
        return 0.0
    path_tokens = _tokens(f"{entry.path} {entry.key}")
    overlap = len(path_tokens.intersection(target_tokens))
    if overlap == 0:
        return 0.0
    return overlap + (overlap / max(1, len(target_tokens)))


def _passes_filter(entry: FieldEntry, value_filter: str | Callable[[FieldEntry], bool]) -> bool:
    if callable(value_filter):
        try:
            return bool(value_filter(entry))
        except Exception:
            return False
    return str(value_filter).strip().lower() in entry.normalized_text_value

