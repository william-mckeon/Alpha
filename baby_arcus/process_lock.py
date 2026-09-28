"""OS-held controller ownership, automatically released after process death."""
import os
from pathlib import Path
from baby_arcus.artifacts import Conflict

class ProcessLock:
    def __init__(self,path,readonly=False):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        self.handle=open(path,"rb" if readonly else "a+b")
        try:
            self.handle.seek(0,2)
            if self.handle.tell()==0:
                if readonly:
                    raise OSError("Controller ownership file is empty")
                self.handle.write(b"0")
                self.handle.flush()
            self.handle.seek(0)
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise Conflict("Another controller owns this state directory") from exc

    def close(self):
        if getattr(self,'handle',None) is not None and not self.handle.closed:
            self.handle.close()

    def __del__(self):
        self.close()
