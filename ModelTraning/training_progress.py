from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def file_size(path: Path) -> str:
    if not path.exists():
        return "ausente"
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} GB"  # pragma: no cover - inalcancavel, o loop sempre retorna em "GB"


def progress_iter(
    iterable: Iterable[Any],
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
    enabled: bool = True,
) -> Iterator[Any]:
    if not enabled:
        yield from iterable
        return

    try:
        from tqdm import tqdm
    except ImportError:
        yield from iterable
        return

    yield from tqdm(iterable, total=total, desc=desc, unit=unit)


class TrainingProgress:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def write(self, message: str = "") -> None:
        if not self.enabled:
            return
        try:
            from tqdm import tqdm

            tqdm.write(message)
        except ImportError:
            print(message, file=sys.stdout, flush=True)

    def banner(self, title: str, details: list[str] | None = None) -> None:
        self.write("")
        self.write("=" * 72)
        self.write(title)
        if details:
            for item in details:
                self.write(f"  {item}")
        self.write("=" * 72)

    def stage(self, number: int, total: int, title: str, details: list[str] | None = None) -> None:
        self.write("")
        self.write(f"[{number}/{total}] {title}")
        if details:
            for item in details:
                self.write(f"  - {item}")

    def info(self, message: str) -> None:
        self.write(f"[info] {message}")

    def ok(self, message: str) -> None:
        self.write(f"[ok] {message}")

    def skip(self, message: str) -> None:
        self.write(f"[skip] {message}")

    def warn(self, message: str) -> None:
        self.write(f"[warn] {message}")

    def fail(self, message: str) -> None:
        self.write(f"[erro] {message}")

    def summary(self, title: str, items: dict[str, Any]) -> None:
        self.write(title)
        if not items:
            return
        width = max(len(str(key)) for key in items)
        for key, value in items.items():
            self.write(f"  {str(key).ljust(width)} : {value}")

    @contextmanager
    def timed(self, label: str, heartbeat_seconds: int = 0) -> Iterator[None]:
        started = time.monotonic()
        self.info(f"{label}...")
        stop_event = threading.Event()
        thread: threading.Thread | None = None

        def heartbeat() -> None:
            while not stop_event.wait(heartbeat_seconds):
                elapsed = format_duration(time.monotonic() - started)
                self.info(f"{label} ainda em execucao ({elapsed})")

        if heartbeat_seconds > 0 and self.enabled:
            thread = threading.Thread(target=heartbeat, name="training-progress-heartbeat", daemon=True)
            thread.start()

        try:
            yield
        except Exception:
            elapsed = format_duration(time.monotonic() - started)
            self.fail(f"{label} falhou apos {elapsed}")
            raise
        finally:
            stop_event.set()
            if thread is not None:
                thread.join(timeout=1)

        elapsed = format_duration(time.monotonic() - started)
        self.ok(f"{label} concluido em {elapsed}")
