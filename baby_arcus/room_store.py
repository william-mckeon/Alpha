"""Atomic room-object save; caller holds the existing exclusive body-store lock."""
import os,time
from pathlib import Path
from baby_arcus.contracts import canonical,decode
from baby_arcus.playroom import Playroom

class RoomStore:
    def __init__(self,root):self.root=Path(root);self.path=self.root/'room-objects.json'
    def load(self):
        if self.path.exists():return Playroom.restore(decode(self.path.read_bytes()))
        room=Playroom();self.save(room);return room
    def save(self,room):
        data=canonical(Playroom.restore(room.record()).record())
        pending=self.root/'room-objects.pending'
        for attempt in range(6):
            try:
                with pending.open('wb') as stream:
                    stream.write(data);stream.flush();os.fsync(stream.fileno())
                os.replace(pending,self.path);return
            except PermissionError:
                if attempt==5:raise
                time.sleep(.02*2**attempt)
