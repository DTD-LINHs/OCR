import os
import uuid
from datetime import datetime


class LocalStorage:
    """Stores uploads on disk under `root`. Swap for MinIO when files must be shared between hosts."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)

    def save(self, data: bytes, extension: str) -> str:
        relative = os.path.join(datetime.now().strftime("%Y/%m/%d"), f"{uuid.uuid4().hex}{extension}")
        path = self._resolve(relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return relative.replace(os.sep, "/")

    def read(self, relative: str) -> bytes:
        with open(self._resolve(relative), "rb") as f:
            return f.read()

    def _resolve(self, relative: str) -> str:
        path = os.path.abspath(os.path.join(self.root, relative))
        if os.path.commonpath([path, self.root]) != self.root:
            raise ValueError("Invalid storage path")
        return path
