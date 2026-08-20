"""Deprecated style compatibility; curses styling lives in tui_legacy only."""


def line_style(line: str, row: int) -> int:
    del line, row
    return 0
