"""Curses color styles for the AutoBrain cockpit."""

import curses


def line_style(line: str, row: int) -> int:
    if not curses.has_colors():
        return curses.A_BOLD if row == 0 else curses.A_NORMAL
    if row == 0:
        return curses.color_pair(1) | curses.A_BOLD
    if line.startswith("Status") and "not connected" in line:
        return curses.color_pair(4) | curses.A_BOLD
    if "export ready" in line or line.endswith("connected"):
        return curses.color_pair(2)
    if line.startswith("Enter"):
        return curses.color_pair(3) | curses.A_BOLD
    if line.startswith("Step") or line.startswith("[ChatGPT]"):
        return curses.color_pair(1)
    return curses.A_NORMAL
