"""Read-only per-service storage inventory, including worker caches."""
from pathlib import Path
import shutil

def inventory(root):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    size=0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                size+=path.stat().st_size
        except FileNotFoundError:
            # Atomic temporary files may disappear while a snapshot is sampled.
            continue
    return {"bytes":size,"free_bytes":shutil.disk_usage(root).free}

def monitored(app,root):
    def dispatch(method,path,body):
        if method=="GET" and path=="/v1/storage":
            return 200,inventory(root)
        return app(method,path,body)
    return dispatch
