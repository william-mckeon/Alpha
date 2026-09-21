"""Offline, hash-verified snapshots of all seven service state directories.

Stop all services first. A stopped controller and released GPU lease are checked,
but this module cannot prove that an independently running simulator is idle.
No backup/restore operation overwrites an existing archive or destination.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path,PurePosixPath
import tempfile
import zipfile
from baby_arcus.contracts import canonical,ContractError
from baby_arcus.process_lock import ProcessLock

SERVICES=("controller","artifacts","simulation","inference","training","evaluator","dashboard")
CHUNK=8*1024*1024

def safe_name(name):
    path=PurePosixPath(name)
    if not name or path.is_absolute() or "\\" in name or ":" in name or any(p in (".","..","") for p in name.split("/")):
        raise ContractError("Unsafe snapshot path")
    if len(path.parts)<2 or path.parts[0] not in SERVICES:
        raise ContractError("Unknown snapshot service path")
    for part in path.parts:
        if part.endswith(("."," ")) or any(ord(char)<32 or char in '<>"|?*' for char in part):
            raise ContractError("Nonportable snapshot path")
        if part.split(".")[0].upper() in {"CON","PRN","AUX","NUL",*(f"COM{i}" for i in range(1,10)),*(f"LPT{i}" for i in range(1,10))}:
            raise ContractError("Reserved snapshot path")
    return path

def copy_hash(source,target=None):
    digest=hashlib.sha256()
    size=0
    while chunk:=source.read(CHUNK):
        digest.update(chunk)
        size+=len(chunk)
        if target is not None:
            target.write(chunk)
    return {"size":size,"sha256":digest.hexdigest()}

def export_snapshot(sources,destination,offline=False):
    if not offline:
        raise ContractError("Stop all services and explicitly select offline snapshot mode")
    if set(sources)!=set(SERVICES):
        raise ContractError("Snapshot requires all seven named service roots")
    roots={name:Path(path).resolve() for name,path in sources.items()}
    for name,root in roots.items():
        if not root.is_dir() and name not in ("evaluator","dashboard"):
            raise ContractError("Missing service directory: "+name)
    destination=Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if any(destination.is_relative_to(root) for root in roots.values()):
        raise ContractError("Snapshot output must be outside all source directories")
    controller=roots["controller"]
    lock=ProcessLock(controller/"controller.lock",readonly=os.name!="nt")
    temporary=None
    try:
        run=json.loads((controller/"run.json").read_text())
        resources=json.loads((controller/"resource.json").read_text())
        if run["status"] not in ("idle","paused","completed","failed") or resources.get("lease") is not None:
            raise ContractError("Controller must be settled with no GPU lease")
        destination.parent.mkdir(parents=True,exist_ok=True)
        descriptor,temporary=tempfile.mkstemp(prefix=".snapshot-",dir=destination.parent)
        os.close(descriptor)
        manifest={"format":1,"services":list(SERVICES),"run_id":run.get("run_id"),
                  "checkpoint_id":run.get("checkpoint_id"),"files":{}}
        with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            for name,root in roots.items():
                for path in sorted(root.rglob("*")):
                    if path.is_symlink() or not path.resolve().is_relative_to(root):
                        raise ContractError("Snapshot refuses links outside service state")
                    if not path.is_file() or path.name=="controller.lock":
                        continue
                    relative=name+"/"+path.relative_to(root).as_posix()
                    safe_name(relative)
                    before=path.stat()
                    with path.open("rb") as source,archive.open(relative,"w",force_zip64=True) as target:
                        manifest["files"][relative]=copy_hash(source,target)
                    after=path.stat()
                    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                        raise ContractError("Service data changed during offline snapshot")
            archive.writestr("manifest.json",canonical(manifest))
        with open(temporary,"r+b") as handle:
            os.fsync(handle.fileno())
        with zipfile.ZipFile(temporary) as archive:
            read_manifest(archive)
        # Atomic no-clobber publication on the same filesystem.
        os.link(temporary,destination)
        return {"archive":str(destination),"files":len(manifest["files"]),
                "bytes":sum(row["size"] for row in manifest["files"].values()),
                "checkpoint_id":manifest["checkpoint_id"]}
    finally:
        lock.close()
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)

def read_manifest(archive):
    entries=archive.infolist()
    names=[entry.filename for entry in entries]
    if len(names)!=len(set(names)) or len(names)>1_000_000:
        raise ContractError("Duplicate or excessive snapshot entries")
    if "manifest.json" not in names or archive.getinfo("manifest.json").file_size>128*1024**2:
        raise ContractError("Missing or excessive snapshot manifest")
    manifest=json.loads(archive.read("manifest.json"))
    if manifest.get("format")!=1 or manifest.get("services")!=list(SERVICES) or not isinstance(manifest.get("files"),dict):
        raise ContractError("Incompatible snapshot format")
    if set(names)!=set(manifest["files"])|{"manifest.json"}:
        raise ContractError("Snapshot inventory mismatch")
    total=0
    portable=set()
    for name,row in manifest["files"].items():
        safe_name(name)
        folded=name.casefold()
        if folded in portable:
            raise ContractError("Case-colliding snapshot paths")
        portable.add(folded)
        info=archive.getinfo(name)
        if info.is_dir() or ((info.external_attr>>16)&0o170000)==0o120000:
            raise ContractError("Snapshot links/directories are not file entries")
        if not isinstance(row,dict) or set(row)!={"size","sha256"} or not isinstance(row["sha256"],str) or not re.fullmatch("[0-9a-f]{64}",row["sha256"]):
            raise ContractError("Invalid snapshot hash record")
        if type(row.get("size")) is not int or row["size"]<0 or info.file_size!=row["size"]:
            raise ContractError("Snapshot size mismatch")
        total+=row["size"]
    for name in portable:
        if any(str(parent) in portable for parent in PurePosixPath(name).parents):
            raise ContractError("Snapshot file/directory collision")
    if total>128*1024**3:
        raise ContractError("Snapshot exceeds initial restore size limit")
    return manifest


def publish_directory(source,destination):
    """Atomic no-replace rename on the supported Windows and Linux hosts."""
    if os.name=="nt":
        os.rename(source,destination)
    else:
        import ctypes
        library=ctypes.CDLL(None,use_errno=True)
        rename=library.renameat2
        rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
        rename.restype=ctypes.c_int
        if rename(-100,os.fsencode(source),-100,os.fsencode(destination),1):
            error=ctypes.get_errno()
            raise OSError(error,os.strerror(error),str(destination))

def verify_snapshot(path):
    with zipfile.ZipFile(path) as archive:
        manifest=read_manifest(archive)
        for name,expected in manifest["files"].items():
            with archive.open(name) as source:
                if copy_hash(source)!=expected:
                    raise ContractError("Snapshot integrity failure: "+name)
    return manifest

def restore_snapshot(path,destination):
    destination=Path(destination).absolute()
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    # The staging directory is newly allocated within the intended destination parent.
    with tempfile.TemporaryDirectory(prefix=".baby-restore-",dir=destination.parent) as temporary:
        staging=Path(temporary).resolve()
        if staging.parent!=destination.parent.resolve():
            raise ContractError("Restore staging escaped destination parent")
        with zipfile.ZipFile(path) as archive:
            manifest=read_manifest(archive)
            for service in SERVICES:
                (staging/service).mkdir()
            for name,expected in manifest["files"].items():
                target=staging.joinpath(*safe_name(name).parts)
                target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(name) as source,target.open("xb") as output:
                    measured=copy_hash(source,output)
                    output.flush()
                    os.fsync(output.fileno())
                if measured!=expected:
                    raise ContractError("Snapshot integrity failure: "+name)
        run=json.loads((staging/"controller/run.json").read_text())
        resource=json.loads((staging/"controller/resource.json").read_text())
        if run["status"] not in ("idle","paused","completed","failed") or resource.get("lease") is not None:
            raise ContractError("Snapshot contains active service ownership")
        if run.get("checkpoint_id")!=manifest.get("checkpoint_id"):
            raise ContractError("Snapshot checkpoint identity mismatch")
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        publish_directory(staging,destination)
    return {"destination":str(destination),"files":len(manifest["files"]),"checkpoint_id":manifest.get("checkpoint_id")}

def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    export=sub.add_parser("export")
    export.add_argument("--sources",required=True,help="JSON mapping all seven service names to state directories")
    export.add_argument("--output",required=True)
    export.add_argument("--offline",action="store_true")
    verify=sub.add_parser("verify")
    verify.add_argument("archive")
    restore=sub.add_parser("restore")
    restore.add_argument("archive")
    restore.add_argument("destination")
    args=parser.parse_args()
    if args.command=="export":
        result=export_snapshot(json.loads(Path(args.sources).read_text()),args.output,args.offline)
    elif args.command=="restore":
        result=restore_snapshot(args.archive,args.destination)
    else:
        manifest=verify_snapshot(args.archive)
        result={"verified":True,"files":len(manifest["files"]),"checkpoint_id":manifest.get("checkpoint_id")}
    print(json.dumps(result))

if __name__=="__main__":
    main()
