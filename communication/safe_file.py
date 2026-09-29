"""Petits utilitaires de persistance locale interprocessus."""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def interprocess_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
    try:
        if os.name == "nt":
            import msvcrt
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def append_jsonl(
    path: Path,
    record: dict,
    max_bytes: int | None = None,
    max_archives: int = 3,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with interprocess_lock(lock_path):
        if max_bytes and path.exists() and path.stat().st_size >= max_bytes:
            archive = path.with_name(f"{path.stem}.{time.time_ns()}{path.suffix}")
            os.replace(path, archive)
            archives = sorted(
                path.parent.glob(f"{path.stem}.*{path.suffix}"),
                key=lambda candidate: candidate.stat().st_mtime,
                reverse=True,
            )
            for stale_archive in archives[max(0, max_archives):]:
                try:
                    stale_archive.unlink()
                except OSError:
                    pass
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
