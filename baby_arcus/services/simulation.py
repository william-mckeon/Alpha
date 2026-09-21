"""Durable episodes and idempotent joint commands. No inference/training in Phase 1."""
from pathlib import Path
import sqlite3
import hashlib
from contextlib import contextmanager
from baby_arcus.actions import Action
from baby_arcus.artifacts import Conflict
from baby_arcus.contracts import EpisodeStart, canonical, decode, digest, envelope, integer, WORLD_VERSION, ContractError
from baby_arcus.lessons import generate, validate_solvable
from baby_arcus.observations import observations
from baby_arcus.world import World

class SimulationApplication:
    def __init__(self, root, artifact_client):
        self.root = Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.artifacts = artifact_client
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, state BLOB NOT NULL, replay BLOB NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS receipts (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, response BLOB NOT NULL)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root/"episodes.sqlite",timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def read(self, episode_id):
        with self.connect() as db:
            row = db.execute("SELECT state,replay FROM episodes WHERE id=?",(episode_id,)).fetchone()
        if not row:
            raise KeyError(episode_id)
        return World.restore(decode(row[0])),decode(row[1])

    def public(self, episode_id, world):
        return {"schema_version":1,"world_version":world.world_version,"episode_id":episode_id,
                "step":world.step,"terminated":world.terminated,"truncated":world.truncated,
                "observations":observations(world)}

    def publish(self, request_id, kind, payload):
        return self.artifacts.request("POST","/v1/artifacts",
            {"schema_version":1,"request_id":request_id,"kind":kind,"payload":payload})["artifact_id"]

    def __call__(self, method, path, body):
        if method == "GET" and path == "/health":
            return 200,{"service":"simulation","schema_version":1,"world_version":WORLD_VERSION}
        if method == "GET" and path == "/ready":
            readiness = self.artifacts.request("GET","/ready")
            if readiness.get("schema_version") != 1:
                raise ContractError("Artifact protocol mismatch")
            return 200,{"ready":True,"service":"simulation","schema_version":1,"world_version":WORLD_VERSION}
        prefix = "/v1/episodes/"
        if method == "GET" and path.startswith(prefix):
            parts = path[len(prefix):].split("/")
            world,replay = self.read(parts[0])
            if len(parts) == 1:
                return 200,self.public(parts[0],world)
            if parts[1:] == ["replay"]:
                return 200,{"schema_version":1,"episode_id":parts[0],"artifact_ids":replay}
            raise KeyError(path)
        if method != "POST":
            raise KeyError(path)
        creating = path == "/v1/episodes"
        if creating:
            start = EpisodeStart.parse(body)
        elif path.startswith(prefix) and path.endswith("/step"):
            envelope(body,("step","actions"))
            integer(body["step"],0,256)
            if not isinstance(body["actions"],dict) or set(body["actions"]) != {"a","b"}:
                raise ContractError("Actions must contain exactly a and b")
            actions = {key:Action.parse(value) for key,value in body["actions"].items()}
        else:
            raise KeyError(path)
        fingerprint = digest({"path":path,"body":body})
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT fingerprint,response FROM receipts WHERE id=?",(body["request_id"],)).fetchone()
            if old:
                if old[0] != fingerprint:
                    raise Conflict("Request ID reused with different content")
                return 200,decode(old[1])
            if creating:
                episode_id = hashlib.sha256(("episode:"+start.request_id).encode()).hexdigest()[:32]
                world = generate(start.family,start.seed,start.max_steps,start.layout_id,start.split,start.difficulty)
                if not validate_solvable(world):
                    raise RuntimeError("Generated lesson failed solvability check")
                initial = {"schema_version":1,"episode_id":episode_id,"state":world.snapshot(),
                           "run_id":start.run_id,"checkpoint_id":start.checkpoint_id}
                ref = self.publish("init-"+episode_id,"episode_initial",initial)
                replay = [ref]
                result = self.public(episode_id,world)
                result["artifact_id"] = ref
                db.execute("INSERT INTO episodes VALUES (?,?,?)",(episode_id,canonical(world.snapshot()),canonical(replay)))
            else:
                episode_id = path[len(prefix):-len("/step")]
                row = db.execute("SELECT state,replay FROM episodes WHERE id=?",(episode_id,)).fetchone()
                if not row:
                    raise KeyError(episode_id)
                world = World.restore(decode(row[0]))
                replay = decode(row[1])
                if body["step"] != world.step:
                    raise Conflict("Step is stale or out of order")
                transition = world.advance(actions)
                result = self.public(episode_id,world)
                result["transition"] = transition
                record = {"schema_version":1,"episode_id":episode_id,"request_id":body["request_id"],
                          "actions":{k:v.wire() for k,v in actions.items()},"transition":transition,
                          "state":world.snapshot(),"observations":result["observations"]}
                ref = self.publish("step-"+fingerprint,"episode_transition",record)
                replay.append(ref)
                result["artifact_id"] = ref
                db.execute("UPDATE episodes SET state=?,replay=? WHERE id=?",
                           (canonical(world.snapshot()),canonical(replay),episode_id))
            db.execute("INSERT INTO receipts VALUES (?,?,?)",(body["request_id"],fingerprint,canonical(result)))
        return 200,result
