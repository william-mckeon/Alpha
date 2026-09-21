"""Human-started standing policy runner; the model proposes joint actions only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid
from baby_arcus.contracts import ContractError
from baby_arcus.body_tools import BodyToolsApplication
class LiveBodyPolicy:
    def __init__(self,app,python,checkpoint,workdir,label="Standing pilot · 48,351 parameters",expected_hash=None,goals=("standing",)):
        self.app=app;self.python=Path(python);self.checkpoint=Path(checkpoint);self.workdir=Path(workdir)
        self.thread=None;self.process=None;self.cancel=threading.Event()
        self.expected_hash=expected_hash
        self.goals=tuple(goals)
        self.info={"status":"idle","policy":label,"checkpoint":self.checkpoint.name,
                   "checkpoint_sha256":None,"actions":0,"hold_seconds":0,"reason":"Ready to start","training":False,
                   "goal":"standing","available_goals":list(self.goals)}
    def snapshot(self):
        return deepcopy(self.info)
    def on_tick(self):
        if self.info["status"] not in ("loading","running"):return
        w=self.app.world
        if w.paused or w.view.held or w.view.region!="playpen" or w.body.sleep_state!="awake":
            self.stop("Body paused, asleep, held or outside playpen");return
        if self.info["status"]=="running":
            from baby_arcus.body_senses import observe_body_senses
            senses=observe_body_senses(w.body,w.view.held)
            from baby_arcus.posture_goals import achieved
            self.info["hold_seconds"]=round(self.info["hold_seconds"]+.1,1) if achieved(senses,self.info["goal"]) else 0
    def emit(self,event,details):
        if self.app.audit:self.app.audit.emit(event,details,durable=True)
    def start(self,goal="standing"):
        # Called while the application's RLock is held.
        if goal not in self.goals:raise ContractError("Posture goal not available in this checkpoint")
        if self.thread and self.thread.is_alive():raise ContractError("Policy is already running or stopping")
        world=self.app.world
        if world.paused or world.view.held or world.view.region!="playpen" or world.body.sleep_state!="awake":
            raise ContractError("Start requires an awake, unpaused body inside the playpen and not held")
        if not self.python.is_file() or not self.checkpoint.is_file():raise ContractError("Standing runtime or checkpoint unavailable")
        # Python 3.10 is the pinned Ubuntu runtime; file_digest requires 3.11.
        checkpoint_digest=hashlib.sha256()
        with self.checkpoint.open("rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):checkpoint_digest.update(chunk)
        checkpoint_hash=checkpoint_digest.hexdigest()
        if self.expected_hash and checkpoint_hash!=self.expected_hash:raise ContractError("Checkpoint differs from qualified model")
        self.cancel=threading.Event()
        self.info.update(status="loading",actions=0,hold_seconds=0,goal=goal,reason="Loading learned "+goal+" policy",
                         checkpoint_sha256=checkpoint_hash)
        candidate=deepcopy(world);candidate.body.motor_mode="independent";candidate.body.eyelid_openness=0;candidate.view.epoch+=1
        self.app.commit(candidate)
        self.emit("live_policy.start",self.snapshot())
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()
        return self.snapshot()
    def stop(self,reason="Stopped by you"):
        self.cancel.set()
        if self.info["status"] in ("loading","running"):
            self.info.update(status="stopped",reason=reason)
            self.emit("live_policy.stop",self.snapshot())
        if self.process and self.process.poll() is None:
            try:self.process.terminate()
            except OSError:pass
        return self.snapshot()
    def close(self):
        self.stop("Desktop closed")
        if self.thread:self.thread.join(timeout=5)
    def _run(self):
        process=None
        try:
            process=subprocess.Popen([str(self.python),"-u","-m","baby_arcus.services.body_predictor",
                "--checkpoint",str(self.checkpoint)],cwd=self.workdir,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,text=True,bufsize=1,
                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            self.process=process
            replies=queue.Queue()
            def reader():
                for line in process.stdout:replies.put(line)
                replies.put(None)
            threading.Thread(target=reader,daemon=True).start()
            def receive(timeout):
                deadline=time.monotonic()+timeout
                while not self.cancel.is_set():
                    try:
                        line=replies.get(timeout=min(.1,max(.001,deadline-time.monotonic())))
                        if line is None:raise RuntimeError("Predictor exited")
                        return json.loads(line)
                    except queue.Empty:
                        if time.monotonic()>=deadline:raise TimeoutError("Predictor deadline")
                raise InterruptedError("Stopped")
            ready=receive(40)
            if ready.get("ready") is not True:raise RuntimeError("Predictor could not load")
            if self.info["goal"] not in ready.get("goals",["standing"]):raise RuntimeError("Checkpoint does not support selected goal")
            from baby_arcus.body_vocabulary import ACTIONS
            tools=BodyToolsApplication(self.app)
            with self.app.lock:
                if self.cancel.is_set():return
                self.info.update(status="running",reason="Choosing joint movements",parameters=ready["parameters"],device=ready.get("device","cpu"))
                self.emit("live_policy.loaded",{"parameters":ready["parameters"],**self.snapshot()})
            last_tick=-1
            deadline=time.monotonic()+45
            while time.monotonic()<deadline and not self.cancel.is_set():
                with self.app.lock:
                    w=self.app.world
                    if w.paused or w.view.held or w.view.region!="playpen" or w.body.sleep_state!="awake":
                        self.stop("Body paused, asleep, held or outside playpen");break
                    if w.tick==last_tick:
                        senses=None
                    else:
                        status,senses=tools("POST","/v1/tools/call",{"request_id":uuid.uuid4().hex,"name":"observe_senses","arguments":{}})
                        last_tick=w.tick
                        if self.info["hold_seconds"]>=5:
                            self.info.update(status="completed",reason=self.info["goal"].capitalize()+" held for five simulated seconds")
                            self.emit("live_policy.completed",self.snapshot());break
                if senses is None:
                    self.cancel.wait(.03);continue
                process.stdin.write(json.dumps({"senses":senses,"goal":self.info["goal"]})+"\n");process.stdin.flush()
                response=receive(3)
                index=response.get("action")
                if type(index) is not int or not 0<=index<len(ACTIONS):raise RuntimeError("Invalid policy action")
                action=ACTIONS[index]
                with self.app.lock:
                    if self.cancel.is_set():break
                    w=self.app.world
                    if w.paused or w.view.held or w.view.region!="playpen" or w.body.sleep_state!="awake":
                        self.stop("Body became unavailable");break
                    self.emit("live_policy.decision",{"senses":senses,"action":action,"tick":w.tick,"goal":self.info["goal"]})
                    if action:
                        status,result=tools("POST","/v1/tools/call",{"request_id":uuid.uuid4().hex,"name":"body_action","arguments":action})
                        if status!=200:raise RuntimeError("Body rejected policy command")
                        self.info["actions"]+=1
                self.cancel.wait(.1)
            with self.app.lock:
                if self.info["status"]=="running":self.info.update(status="stopped",reason="Bounded session ended")
        except InterruptedError:
            pass
        except Exception as exc:
            with self.app.lock:
                if not self.cancel.is_set():
                    self.info.update(status="error",reason=type(exc).__name__+": "+str(exc))
                    self.emit("live_policy.error",{"exception_type":type(exc).__name__})
        finally:
            if process:
                if process.poll() is None:process.terminate()
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=2)
                if process.stdin:process.stdin.close()
                if process.stdout:process.stdout.close()
            self.process=None
