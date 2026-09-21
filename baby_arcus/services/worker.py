"""Process-isolated model execution with a killable absolute deadline."""
import multiprocessing as mp
from pathlib import Path
import threading
import time
from baby_arcus.contracts import ContractError
from baby_arcus.artifacts import Conflict
from baby_arcus.transport import Client

def child(connection,root,artifact_url,token,device,role):
    from baby_arcus.audit import AuditLog,TRACE
    audit=AuditLog(Path(root)/"audit",role+"-worker")
    import torch
    from baby_arcus.services.inference import InferenceEngine
    from baby_arcus.services.training import TrainingEngine
    torch.set_num_threads(2)
    engine = (InferenceEngine if role=="inference" else TrainingEngine)(
        root,Client(artifact_url,token,timeout=30),device)
    try:
        while True:
            trace,command = connection.recv()
            TRACE.set(trace)
            audit.emit("model.command",{"command":command},durable=True)
            try:
                result=engine(command)
                audit.emit("model.result",{"request_id":command.get("request_id"),"result":result},durable=True)
                connection.send((True,result))
            except Exception as exc:
                audit.emit("model.error",{"request_id":command.get("request_id"),"exception_type":type(exc).__name__},durable=True)
                from baby_arcus.contracts import canonical
                (Path(root)/"last_error.json").write_bytes(canonical({"operation":command.get("operation"),
                    "request_id":command.get("request_id"),"error":type(exc).__name__+": "+str(exc),"time":time.time()}))
                connection.send((False,type(exc).__name__+": "+str(exc)))
    except (EOFError,BrokenPipeError):
        pass
    finally:
        audit.close()

class WorkerApplication:
    def __init__(self,role,root,artifact_url,controller_url,token="",device="cpu"):
        self.role,self.root,self.artifact_url,self.device=role,str(root),artifact_url,device
        Path(root).mkdir(parents=True,exist_ok=True)
        self.controller=Client(controller_url,token,timeout=5)
        self.token=token
        self.lock=threading.RLock()
        self.process=None
        self.pipe=None
        self.lease=None
        self.cancel=threading.Event()

    def unload(self):
        if self.process:
            self.process.terminate()
            self.process.join(5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(5)
            if self.process.is_alive():
                raise RuntimeError("Worker could not be stopped; resource remains owned")
            self.process.close()
            self.process=None
            self.pipe.close()
            self.pipe=None
        self.lease=None
        self.cancel.clear()
        return {"unloaded":True}

    def __call__(self,method,path,body):
        if method=="POST" and path=="/v1/cancel":
            if self.lease is not None and body.get("lease_id")!=self.lease:
                raise Conflict("Cancellation lease mismatch")
            self.cancel.set()
            return 200,{"cancel_requested":True}
        if method=="GET" and path in ("/health","/ready"):
            return 200,{"service":self.role,"ready":True,"loaded":self.process is not None}
        if method!="POST":
            raise KeyError(path)
        with self.lock:
            if path=="/v1/unload":
                return 200,self.unload()
            if path!="/v1/execute":
                raise KeyError(path)
            deadline=body["deadline"]
            if not isinstance(deadline,(int,float)) or not time.time()<deadline<=time.time()+43200:
                raise ContractError("Invalid operation deadline")
            lease_id=body["lease_id"]
            self.controller.request("POST","/v1/resources/check",{"lease_id":lease_id,"owner":self.role,"renew":True})
            if self.cancel.is_set():
                raise Conflict("Worker cancellation is pending; unload before reuse")
            if self.lease is not None and self.lease!=lease_id:
                raise Conflict("Loaded worker belongs to another lease")
            if self.process is None:
                ctx=mp.get_context("spawn")
                self.pipe,remote=ctx.Pipe()
                self.process=ctx.Process(target=child,args=(remote,self.root,self.artifact_url,self.token,self.device,self.role),daemon=True)
                self.process.start()
                remote.close()
                self.lease=lease_id
            from baby_arcus.audit import TRACE
            self.pipe.send((TRACE.get(),body))
            renewed=time.monotonic()
            while not self.pipe.poll(min(.1,max(0,deadline-time.time()))):
                if self.cancel.is_set() or time.time()>=deadline or not self.process.is_alive():
                    self.unload()
                    raise RuntimeError("Worker deadline or process failure; candidate not accepted")
                if time.monotonic()-renewed>15:
                    try:
                        self.controller.request("POST","/v1/resources/check",{"lease_id":lease_id,"owner":self.role,"renew":True})
                    except Exception:
                        self.unload()
                        raise
                    renewed=time.monotonic()
            ok,result=self.pipe.recv()
            if not ok:
                self.unload()
                raise RuntimeError(result)
            return 200,result
