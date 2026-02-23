from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import VisibilityObj


def write_jsonl(path: Path, objs: Iterable[VisibilityObj]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for o in objs:
            f.write(o.model_dump_json())
            f.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows