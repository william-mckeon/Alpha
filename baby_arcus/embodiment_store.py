"""Atomic body persistence with exclusive process ownership."""
import os
import time
from pathlib import Path
from baby_arcus.contracts import canonical, decode
from baby_arcus.embodiment import Embodiment
from baby_arcus.process_lock import ProcessLock

class EmbodimentStore:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = ProcessLock(self.root / "body.lock")
        self.path = self.root / "body.json"

    def load(self):
        if self.path.exists():
            raw=self.path.read_bytes()
            record=decode(raw)
            body=Embodiment.restore(record)
            if record["version"]==1:
                backup=self.root/"body.v1.backup.json"
                if not backup.exists():
                    with backup.open("xb") as stream:
                        stream.write(raw);stream.flush();os.fsync(stream.fileno())
                self.save(body)
            return body
        body = Embodiment()
        self.save(body)
        return body

    def save(self, body):
        data = canonical(Embodiment.restore(body.record()).record())
        temp = self.root / "body.pending"
        # Windows readers/sync software can briefly hold a file without delete
        # sharing. Retain atomic replacement and fail after a bounded retry.
        for attempt in range(6):
            try:
                with temp.open("wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp, self.path)
                return
            except PermissionError:
                if attempt==5:raise
                time.sleep(.02*2**attempt)

    def close(self):
        self.lock.close()

