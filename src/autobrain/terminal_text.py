"""Unicode-safe terminal display-width helpers."""

from __future__ import annotations

import unicodedata


def truncate_terminal_text(value: str, width: int) -> str:
    normalized = _normalize_terminal_text(value)
    if terminal_width(normalized) <= width:
        return normalized
    if width <= 1:
        return "…"[:width]
    target_width = width - 1
    result: list[str] = []
    used_width = 0
    for cluster in _terminal_clusters(normalized):
        cluster_width = _cluster_width(cluster)
        if used_width + cluster_width > target_width:
            break
        result.append(cluster)
        used_width += cluster_width
    return "".join(result) + "…"


def terminal_width(value: str) -> int:
    normalized = _normalize_terminal_text(value)
    return sum(_cluster_width(cluster) for cluster in _terminal_clusters(normalized))


def _terminal_clusters(value: str) -> tuple[str, ...]:
    clusters: list[str] = []
    current = ""
    join_next = False
    for character in value:
        if not current:
            current = character
        elif join_next or character == "\u200d" or _is_cluster_extension(character):
            current += character
        else:
            clusters.append(current)
            current = character
        join_next = character == "\u200d"
    if current:
        clusters.append(current)
    return tuple(clusters)


def _cluster_width(cluster: str) -> int:
    measured_width = sum(_character_width(character) for character in cluster)
    if "\ufe0f" in cluster or "\u20e3" in cluster:
        return max(2, measured_width)
    return measured_width


def _is_cluster_extension(character: str) -> bool:
    return (
        bool(unicodedata.combining(character))
        or unicodedata.category(character) in {"Cf", "Me", "Mn"}
        or "\U0001f3fb" <= character <= "\U0001f3ff"
    )


def _character_width(character: str) -> int:
    if unicodedata.combining(character):
        return 0
    if unicodedata.category(character) in {"Cf", "Me", "Mn"}:
        return 0
    if unicodedata.east_asian_width(character) in {"F", "W"}:
        return 2
    return 1


def _normalize_terminal_text(value: str) -> str:
    normalized: list[str] = []
    for character in value:
        if character == "\t":
            normalized.append("    ")
        elif character in {"\n", "\r"}:
            normalized.append(" ")
        elif unicodedata.category(character) == "Cc":
            normalized.append("?")
        else:
            normalized.append(character)
    return "".join(normalized)
