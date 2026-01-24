import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable


class DiskCache:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(key_obj: Any) -> str:
        payload = json.dumps(
            key_obj,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _path_for_key(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path_for_key(key)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, entry: dict[str, Any]) -> Path:
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def get_or_compute(
        self,
        key_obj: Any,
        compute: Callable[[], Any],
    ) -> tuple[dict[str, Any], bool]:
        key = self.make_key(key_obj)
        cached = self.get(key)
        if cached is not None:
            return cached, True

        value = compute()
        entry = {
            "key": key,
            "created_at": time.time(),
            "key_obj": key_obj,
            "value": value,
        }
        self.set(key, entry)
        return entry, False

