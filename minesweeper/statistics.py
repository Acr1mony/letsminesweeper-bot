from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StatisticsSnapshot:
    opened: int = 0
    flagged: int = 0


class StatisticsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(__file__).resolve().parents[1] / "data" / "statistics.json"
        self._lock = threading.Lock()
        self._snapshot = self._load()

    def _load(self) -> StatisticsSnapshot:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return StatisticsSnapshot(
                opened=max(0, int(data.get("opened", 0))),
                flagged=max(0, int(data.get("flagged", 0))),
            )
        except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError):
            return StatisticsSnapshot()

    def snapshot(self) -> StatisticsSnapshot:
        with self._lock:
            return self._snapshot

    def add(self, *, opened: int = 0, flagged: int = 0) -> StatisticsSnapshot:
        if opened < 0 or flagged < 0:
            raise ValueError("统计增量不能为负数")
        with self._lock:
            self._snapshot = StatisticsSnapshot(
                opened=self._snapshot.opened + opened,
                flagged=self._snapshot.flagged + flagged,
            )
            self._save_locked()
            return self._snapshot

    def clear(self) -> StatisticsSnapshot:
        with self._lock:
            self._snapshot = StatisticsSnapshot()
            self._save_locked()
            return self._snapshot

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {"opened": self._snapshot.opened, "flagged": self._snapshot.flagged},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
