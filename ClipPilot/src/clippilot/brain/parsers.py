# -*- coding: utf-8 -*-
"""Markdown table parsing and updating utilities for ClipPilot ledgers."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional


def parse_markdown_table(content: str) -> list[dict[str, str]]:
    """Parse a markdown table into a list of row dictionaries mapped by column header.

    Headers are determined from the first table row.
    """
    rows: list[list[str]] = []
    headers: list[str] = []

    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue

        # Split cells and strip whitespace
        cells = [c.strip() for c in line.split("|")[1:-1]]

        # Skip separator rows like |---|---|
        if all(re.match(r"^:?-+:?$", c) for c in cells if c):
            continue

        if not headers:
            headers = cells
        else:
            rows.append(cells)

    # Map rows to headers
    result = []
    for row in rows:
        row_dict = {}
        for idx, header in enumerate(headers):
            val = row[idx] if idx < len(row) else ""
            # Strip standard styling like **Bold** or `code` markers if present
            row_dict[header.lower().replace("#", "num").strip()] = val
        result.append(row_dict)
    return result


def parse_learned_rules(skill_md_content: str) -> str:
    """Extract content between <!-- LEARNED-RULES-START --> and <!-- LEARNED-RULES-END -->."""
    pattern = r"<!--\s*LEARNED-RULES-START\s*-->(.*?)<!--\s*LEARNED-RULES-END\s*-->"
    match = re.search(pattern, skill_md_content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def update_topic_status(file_content: str, topic_num: str, status_value: str) -> str:
    """Find the row with the given topic number and update its Status column to status_value."""
    lines = file_content.splitlines()
    for idx, line in enumerate(lines):
        # We check if the line is a table row starting with the topic number (e.g. | 006 | or | 006 | unused |)
        if line.strip().startswith("|"):
            parts = line.split("|")
            if len(parts) >= 3 and parts[1].strip() == topic_num:
                parts[2] = f" {status_value} "
                lines[idx] = "|".join(parts)
                break
    return "\n".join(lines) + ("\n" if file_content.endswith("\n") else "")
