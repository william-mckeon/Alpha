"""Independent playroom simulation and browser gateway; no training imports."""
import argparse
from collections import OrderedDict
from copy import deepcopy
import json
import os
from pathlib import Path
import secrets
import threading
from urllib.parse import urlsplit
import uuid

from baby_arcus.contracts import ContractError, fields, identifier
from baby_arcus.playroom import DT
from baby_arcus.play_session import PlaySession
from baby_arcus.embodiment_store import EmbodimentStore
from baby_arcus.body_tools import BodyToolsApplication, validate_body_action
from baby_arcus.transport import Client, StaticResponse, serve
from baby_arcus.conversation_store import ConversationStore
from baby_arcus.body_senses import observe_body_senses
from baby_arcus.audit import AuditLog


class PlayroomApplication:
    def __init__(self, state_root=None):
        self.store = EmbodimentStore(state_root) if state_root is not None else None
        from baby_arcus.room_store import RoomStore
        self.room_store=RoomStore(state_root) if state_root is not None else None
        try:
            self.world = PlaySession(body=self.store.load() if self.store else None,
                                     environment=self.room_store.load() if self.room_store else None)
            self.conversation = ConversationStore(state_root)
        except Exception:
            if self.store:
                self.store.close()
            raise
        self.failure = None
        self.lock = threading.RLock()
        self.session = uuid.uuid4().hex
        self.audit = AuditLog(Path(state_root)/"audit","playroom",compressed=True,max_bytes=4*1024**3) if state_root else None
        if self.audit:self.audit.emit("body.loaded",{"session_id":self.session,"state":self.world.snapshot()},durable=True)
        self.events = []
        self.requests = OrderedDict()
        self.stop = threading.Event()
        self.on_view_change = lambda: None
        self.policy = None
        self.language = None
        self.visual = None
        self.rest = None
        self.curiosity = None
        self.shared = None
        from baby_arcus.interaction_store import InteractionStore
        self.interactions=InteractionStore(state_root)
        for row in self.conversation.rows:
            key='message-'+row['request_id']
            if not row.get('model_read') and key not in self.interactions.by_id:
                self.interactions.append(key,'message',{k:row[k] for k in ('request_id','sender','text')},self.world.tick,self.world.body.entity_id)

    def publish(self,request_id,kind,payload):
        if kind in ('call','message','feedback'):
            candidate=deepcopy(self.world);candidate.body.stimulation=1.;self.commit(candidate)
        if kind=='call':
            payload={**payload,'hearing':{'schema':'arcus-symbolic-hearing-v1',
                'utterance':'Come here, Arcus','sender':self.world.environment.human['name'],
                'source':deepcopy(self.world.environment.human),'localization':'ideal_simulated_location',
                'audio_waveform':False}}
        row=self.interactions.append(request_id,kind,payload,self.world.tick,self.world.body.entity_id)
        if self.shared and self.shared.info['enabled']:return row
        goals=getattr(self.policy,'goals',())
        if kind=='call' and isinstance(goals,(list,tuple)) and 'approach' in goals:
            self.policy.on_event(row);return row
        if kind=='call' and self.visual and self.visual.handles_calls():
            self.visual.on_call(row);return row
        if self.policy and hasattr(self.policy,'on_event'):self.policy.on_event(row)
        return row

    def expire_view(self):
        if self.world.view.expire():
            if self.audit:self.audit.emit("view.expired",{"state":self.world.snapshot()},durable=True)
            self.on_view_change()

    def desktop_event(self, body):
        if self.audit:self.audit.emit("desktop.event",{"request":body})
        fields(body, ("request_id", "kind"), ("inside",))
        identifier(body["request_id"])
        kind = body["kind"]
        fields(body, ("request_id", "kind", "inside") if kind in ("carry", "drop") else ("request_id", "kind"))
        with self.lock:
            self.expire_view()
            if self.rest and kind!='heartbeat':self.rest.stop('Human desktop interaction')
            if self.curiosity and kind!='heartbeat':self.curiosity.stop('Human desktop interaction')
            if self.visual and kind!='heartbeat' and self.visual.snapshot()['enabled']:
                self.visual.stop('Human desktop interaction')
            key = "desktop-" + body["request_id"]
            if key in self.interactions.by_id and key not in self.requests:
                old=self.interactions.by_id[key]
                if old['kind']!=kind or old['payload'].get('inside')!=body.get('inside'):return 409,{'error':'Desktop request ID conflict'}
                return 200,{'state':self.world.snapshot()}
            if key in self.requests:
                old, reply = self.requests[key]
                if old != body:
                    return 409, {"error": "Desktop request ID conflict"}
                return 200, {"state": self.world.snapshot()}
            if not self.audit and kind not in ("heartbeat", "return") and len(self.events) >= 1000:
                return 409, {"error": "Session full; export and restart"}
            candidate = deepcopy(self.world)
            candidate.view.apply(kind, body.get("inside"))
            if self.policy and (candidate.view.held or candidate.view.region!="playpen"):
                self.policy.stop("Picked up or outside playpen")
            if kind != "heartbeat":
                candidate.last_result = candidate.view.event.replace("_", " ")
            self.commit(candidate)
            if kind!='heartbeat':self.publish(key,kind,{'inside':body.get('inside')})
            response = {"state": self.world.snapshot()}
            if kind != "heartbeat":
                self.events.append({"tick": self.world.tick, "source": "human",
                    "action": {"kind": kind}, "result": candidate.last_result})
                self.requests[key] = (deepcopy(body), response)
                self.events=self.events[-1000:]
            return 200, response

    def run_clock(self):
        while not self.stop.wait(DT):
            with self.lock:
                try:
                    self.advance()
                except OSError:
                    if self.audit:self.audit.emit("clock.error",{"exception_type":"OSError"},durable=True)
                    self.failure = "Body persistence failed; restart after checking storage"
                    self.stop.set()

    def commit(self, candidate):
        if self.audit and not self.audit.status()['healthy']:
            self.failure='Audit storage unavailable; archive logs and restart before continuing'
            self.stop.set()
            raise OSError(self.failure)
        if self.room_store and candidate.environment.record()!=self.world.environment.record():
            self.room_store.save(candidate.environment)
        if self.store and candidate.body.record() != self.world.body.record():
            self.store.save(candidate.body)
        changed = candidate.view.stamp() != self.world.view.stamp()
        self.world = candidate
        if self.audit and not self.audit.emit("body.state",{"session_id":self.session,"state":candidate.snapshot()}):
            self.failure='Audit storage unavailable; archive logs and restart before continuing'
            self.stop.set()
            raise OSError(self.failure)
        if changed:
            self.on_view_change()

    def advance(self):
        if self.audit:
            while len(self.requests)>1000:self.requests.popitem(last=False)
        self.expire_view()
        candidate = deepcopy(self.world)
        candidate.step()
        self.commit(candidate)
        if self.shared and self.shared.info['enabled']:
            self.shared.on_tick();return
        if self.shared and self.shared.cfg.get('shared_only'):return
        if self.policy:self.policy.on_tick()
        if self.language:self.language.on_tick()
        if self.visual:self.visual.on_tick()
        if self.rest:self.rest.on_tick()
        if self.curiosity:self.curiosity.on_tick()

    def close(self):
        if self.shared:self.shared.close()
        if self.curiosity:self.curiosity.close()
        if self.rest:self.rest.close()
        if self.visual:self.visual.close()
        if self.language:self.language.close()
        if self.policy:self.policy.close()
        self.stop.set()
        with self.lock:
            if self.store:
                self.store.close()
            if self.audit:self.audit.close()

    def __call__(self, method, path, body):
        with self.lock:
            self.expire_view()
            if method == "GET" and path in ("/health", "/ready"):
                return (503 if self.failure else 200), {"service": "playroom", "ready": not self.failure,
                    "model_connected": bool(self.policy and (self.policy.info["status"]=="running" or self.policy.snapshot().get("receiver")=="ready")), "error": self.failure,"audit":self.audit.status() if self.audit else {"enabled":False}}
            if self.failure:
                return 503, {"error": self.failure}
            if method=='POST' and path=='/v1/shared/control':
                fields(body,('request_id','action'));identifier(body['request_id'])
                if not self.shared:return 409,{'error':'Shared runtime unavailable'}
                key='shared-'+body['request_id']
                if key in self.requests:
                    if self.requests[key][0]!=body:return 409,{'error':'Shared request conflict'}
                    return 200,self.shared.snapshot()
                result=self.shared.control(body['action']);self.requests[key]=(deepcopy(body),result)
                return 200,result
            if method=='POST' and path=='/v1/policy/start' and self.shared and self.shared.info['enabled']:
                fields(body,('request_id',),('goal',));identifier(body['request_id'])
                goal=body.get('goal','standing')
                if goal not in ('standing','lying','sitting','approach'):raise ContractError('Unknown shared task')
                self.publish('shared-task-'+body['request_id'],'task',{'goal':goal})
                return 200,{'status':'queued','goal':goal,'reason':'Task delivered through simulated hearing; no movement forced'}
            if method=='POST' and path=='/v1/policy/stop' and self.shared and self.shared.info['enabled']:
                self.shared.stop('Caregiver stopped movement');return 200,self.shared.snapshot()
            if method=='POST' and path=='/v1/language/control' and self.shared and self.shared.info['enabled']:
                fields(body,('request_id','action'));identifier(body['request_id'])
                text={'start':'listen','listen':'listen','pause':'pause listening','resume':'resume listening','replay':'replay that','restart':'restart listening'}.get(body['action'])
                if body['action']=='stop':self.shared.stop('Caregiver stopped shared learning');return 200,self.shared.snapshot()
                if text is None:raise ContractError('Unknown shared hearing request')
                self.publish('shared-hearing-'+body['request_id'],'hearing_task',{'text':text})
                return 200,{'queued':True,'channel':'simulated_hearing','text':text}
            if method=='POST' and (path.endswith('/control') or path.startswith('/v1/policy/')) and self.shared and self.shared.info['enabled']:
                raise ContractError('Stop shared mode before selecting a legacy controller')
            if method=='POST' and self.shared and self.shared.cfg.get('shared_only') and (path.endswith('/control') or path=='/v1/policy/start'):
                raise ContractError('Start the qualified shared model; legacy learners cannot bypass the fixed 0.25 depth requirement')
            if method == "GET" and path == "/v1/state":
                state=self.world.snapshot()
                state['shared']=self.shared.snapshot() if self.shared else {'enabled':False,'status':'unavailable'}
                state["capabilities"]={"native_body":True,"independent_joints":True,"human_messages":True,
                                       "model_connected":bool(self.policy and (self.policy.info["status"]=="running" or self.policy.snapshot().get("receiver")=="ready"))}
                state["policy"]=self.policy.snapshot() if self.policy else {"status":"unavailable","reason":"Start the desktop host"}
                state['controller']=state['policy']
                state["audit"]=self.audit.status() if self.audit else {"enabled":False}
                state['interactions']=self.interactions.snapshot()
                state['language']=self.language.snapshot() if self.language else {'status':'unavailable'}
                state['visual']=self.visual.snapshot() if self.visual else {'status':'unavailable'}
                state['rest']=self.rest.snapshot() if self.rest else {'status':'unavailable','enabled':False}
                state['curiosity']=self.curiosity.snapshot() if self.curiosity else {'status':'unavailable','enabled':False}
                if state['curiosity'].get('enabled'):
                    state['controller']=state['curiosity'];state['capabilities']['model_connected']=True
                if state['rest'].get('enabled'):
                    state['controller']=state['rest'];state['capabilities']['model_connected']=True
                state['visual']['navigation_ready']=self.visual.handles_calls() if self.visual else False
                if state['visual'].get('enabled'):
                    state['controller']=state['visual']
                    state['capabilities']['model_connected']=(state['visual']['status']=='running' or
                        (state['visual']['status']=='preparing' and state['policy']['status']=='running'))
                if state['shared'].get('enabled'):
                    state['controller']=state['shared'];state['capabilities']['model_connected']=state['shared']['status']=='running'
                return 200, state
            if method=='POST' and path=='/v1/curiosity/control':
                fields(body,('request_id','action'));identifier(body['request_id'])
                if not self.curiosity:return 409,{'error':'Exploration runtime unavailable'}
                key='curiosity-'+body['request_id']
                if key in self.requests:
                    if self.requests[key][0]!=body:return 409,{'error':'Exploration request conflict'}
                    return 200,self.curiosity.snapshot()
                result=self.curiosity.control(body['action']);self.requests[key]=(deepcopy(body),result)
                if self.audit:self.audit.emit('curiosity.control',body,durable=True)
                return 200,result
            if method=='POST' and path=='/v1/rest/control':
                if self.curiosity:self.curiosity.stop('Rest controller selected')
                fields(body,('request_id','action'));identifier(body['request_id'])
                if not self.rest:return 409,{'error':'Rest runtime unavailable'}
                key='rest-'+body['request_id']
                if key in self.requests:
                    if self.requests[key][0]!=body:return 409,{'error':'Rest request conflict'}
                    return 200,self.rest.snapshot()
                result=self.rest.control(body['action']);self.requests[key]=(deepcopy(body),result)
                return 200,result
            if method=='POST' and path=='/v1/visual/control':
                fields(body,('request_id','action'));identifier(body['request_id'])
                if not self.visual:return 409,{'error':'Visual runtime unavailable'}
                key='visual-'+body['request_id']
                if key in self.requests:
                    if self.requests[key][0]!=body:return 409,{'error':'Visual request ID conflict'}
                    return 200,self.visual.snapshot()
                result=self.visual.control(body['action']);self.requests[key]=(deepcopy(body),result)
                if self.curiosity:self.curiosity.stop('Visual controller selected')
                if self.rest:self.rest.stop('Visual controller selected')
                if self.audit:self.audit.emit('visual.control',body,durable=True)
                return 200,result
            if method == 'POST' and path == '/v1/language/control':
                fields(body,('request_id','action'));identifier(body['request_id'])
                if not self.language:return 409,{'error':'Language runtime unavailable'}
                key='language-'+body['request_id']
                if key in self.requests:
                    if self.requests[key][0]!=body:return 409,{'error':'Language request ID conflict'}
                    return 200,self.language.snapshot()
                result=self.language.control(body['action']);self.requests[key]=(deepcopy(body),result)
                if self.audit:self.audit.emit('language.control',{'request_id':body['request_id'],'action':body['action']},durable=True)
                return 200,result
            if method == "POST" and path in ("/v1/policy/start","/v1/policy/stop"):
                if self.curiosity:self.curiosity.stop('Body controller selected')
                if self.rest:self.rest.stop('Body controller selected')
                fields(body,("request_id",),("goal",) if path.endswith("/start") else ());identifier(body["request_id"])
                if not self.policy:return 409,{"error":"Standing controller unavailable"}
                if self.visual and self.visual.snapshot()['enabled']:self.visual.stop('Movement controller selected')
                key="policy-"+body["request_id"]
                command=(path,body.get("goal","standing"))
                if key in self.interactions.by_id and key not in self.requests:
                    old=self.interactions.by_id[key]
                    if old['kind']!=('task' if path.endswith('/start') else 'stop') or old['payload']['goal']!=command[1]:return 409,{'error':'Policy request ID conflict'}
                    return 200,self.policy.snapshot()
                if key in self.requests:
                    if self.requests[key][0]!=command:return 409,{"error":"Policy request ID conflict"}
                    return 200,self.policy.snapshot()
                result=self.policy.start(body.get("goal","standing")) if path.endswith("/start") else self.policy.stop()
                self.publish(key,'task' if path.endswith('/start') else 'stop',{'goal':body.get('goal','standing')})
                self.requests[key]=(command,result)
                return 200,result
            if method == "GET" and path == "/v1/senses":
                return 200, {**observe_body_senses(self.world.body,self.world.view.held),
                             "paused":self.world.paused}
            if method=='GET' and path=='/v1/interactions':
                return 200,{'events':self.interactions.pending() if self.world.body.sleep_state=='awake' else []}
            if method == "GET" and path in ("/v1/messages","/v1/messages/available"):
                awake=self.world.body.sleep_state=="awake"
                pending=sum(row["status"]=="queued" for row in self.conversation.rows)
                self.conversation.release(awake)
                if self.audit and awake and pending:self.audit.emit("messages.released",{"count":pending},durable=True)
                rows=self.conversation.snapshot(model=path.endswith("/available"))
                if path.endswith("/available") and not awake: rows=[]
                language=self.language.snapshot() if self.language else {}
                for row in rows:row['language_receipt']=language.get('message_receipts',{}).get(row['request_id'])
                shared=self.shared.snapshot() if self.shared else {}
                return 200, {"messages":rows,'expressions':language.get('expressions',[])+shared.get('expressions',[]),
                             "model_connected":bool(shared.get('enabled') or (self.policy and self.policy.snapshot().get('receiver')=='ready'))}
            if method == "POST" and path == "/v1/messages":
                result=self.conversation.send(body,self.world.body.sleep_state=="awake")
                self.publish('message-'+body['request_id'],'message',body)
                if self.audit:self.audit.emit("message.saved",{"message":result},durable=True)
                return 200, {"message":result}
            if method == "GET" and path == "/v1/session":
                return 200, {"session": self.session, "state": self.world.snapshot(),
                             "messages":self.conversation.snapshot(),
                             "language":self.language.snapshot() if self.language else None,
                             "events": deepcopy(self.events), "interactions":self.interactions.snapshot(),"limit": 1000,
                             "note": "In-memory session; export before closing. Events beyond limit are rejected."}
            if method == "POST" and path == "/v1/action":
                if self.audit:self.audit.emit("action.requested",{"session_id":self.session,"request":body},durable=True)
                fields(body, ("request_id", "action", "source"))
                identifier(body["request_id"])
                if body["source"] not in ("human", "policy"):
                    raise ContractError("Unknown action source")
                if body["source"] == "policy":
                    if self.shared and self.shared.info['enabled'] and body['request_id']!=self.shared.pending_request:
                        raise ContractError('Shared controller owns body actions')
                    validate_body_action(body["action"])
                previous = self.requests.get(body["request_id"])
                old=self.interactions.by_id.get('action-'+body['request_id']) if body['source']=='human' else None
                if old and not previous:
                    if any(old['payload'].get(k)!=v for k,v in body['action'].items()):return 409,{'error':'Interaction ID conflict'}
                    return 200,{'state':self.world.snapshot(),'event':{'tick':old['tick'],'source':'human','result':'Previously recorded interaction'}}
                if previous:
                    if previous[0] != body:
                        return 409, {"error": "Request ID reused with different content"}
                    return 200, deepcopy(previous[1])
                if not self.audit and len(self.events) >= 1000:
                    return 409, {"error": "Session full. Export and restart the playroom."}
                candidate = deepcopy(self.world)
                result = candidate.action(body["action"])
                if self.shared and body['source']=='human' and body['action']['kind'] not in ('call','feedback'):
                    self.shared.stop('Human control took over')
                if self.curiosity and body['source']=='human':self.curiosity.stop('Human control took over')
                if self.rest and body['source']=='human':
                    # A sleeping hearing cue can inform voluntary wake. It never
                    # queues a movement command to replay after waking.
                    if not (body['action']['kind'] in ('call','feedback') and self.world.body.sleep_state=='sleeping'):
                        self.rest.stop('Human control took over')
                if self.visual and body['source']=='human' and self.visual.snapshot()['enabled']:
                    self.visual.stop('Human control took over')
                if self.policy and body["source"]=="human" and body["action"]["kind"] not in ("human","call","feedback"):
                    self.policy.stop("Human control took over")
                self.commit(candidate)
                if body['source']=='human':
                    if self.policy and body['action']['kind']=='human':self.policy.stop('Caller marker changed; call again')
                    payload=deepcopy(body['action'])
                    payload['sender']=self.world.environment.human['name']
                    if payload['kind']=='feedback':payload['about_event']=getattr(self.policy,'event_sequence',None)
                    self.publish('action-'+body['request_id'],body['action']['kind'],payload)
                self.events.append({"tick": self.world.tick, "source": body["source"],
                                    "action": deepcopy(body["action"]), "result": result})
                response = {"state": self.world.snapshot(), "event": deepcopy(self.events[-1])}
                # One entry per accepted command, retained for the entire bounded session.
                self.requests[body["request_id"]] = (deepcopy(body), deepcopy(response))
                if self.audit:self.events=self.events[-1000:]
                if self.audit:self.audit.emit("action.completed",{"request_id":body["request_id"],"event":response["event"],
                                                                "state":response["state"]},durable=True)
                return 200, response
            raise KeyError(path)


class PlayroomViewer:
    def __init__(self, url, token):
        self.client = Client(url, token)
        self.web = Path(__file__).resolve().parent.parent / "web"

    def __call__(self, method, path, body):
        if method=='POST' and path=='/api/shared/control':
            from baby_arcus.transport import RemoteError
            try:return 200,self.client.request(method,'/v1/shared/control',body)
            except RemoteError as exc:
                if exc.status in (400,409):
                    try:message=json.loads(str(exc)).get('error',str(exc))
                    except ValueError:message=str(exc)
                    return exc.status,{'error':message}
                raise
        if method=='POST' and path=='/api/curiosity/control':
            from baby_arcus.transport import RemoteError
            try:return 200,self.client.request(method,'/v1/curiosity/control',body)
            except RemoteError as exc:
                if exc.status in (400,409):return exc.status,{'error':str(exc)}
                raise
        if method=='POST' and path=='/api/rest/control':
            from baby_arcus.transport import RemoteError
            try:return 200,self.client.request(method,'/v1/rest/control',body)
            except RemoteError as exc:
                if exc.status in (400,409):return exc.status,{'error':str(exc)}
                raise
        if method=='POST' and path=='/api/visual/control':
            from baby_arcus.transport import RemoteError
            try:return 200,self.client.request(method,'/v1/visual/control',body)
            except RemoteError as exc:
                if exc.status in (400,409):return exc.status,{'error':str(exc)}
                raise
        if method=='POST' and path=='/api/language/control':
            from baby_arcus.transport import RemoteError
            try:return 200,self.client.request(method,'/v1/language/control',body)
            except RemoteError as exc:
                if exc.status in (400,409):return exc.status,{'error':str(exc)}
                raise
        if method=="POST" and path in ("/api/policy/start","/api/policy/stop"):
            from baby_arcus.transport import RemoteError
            try:return 200,self.client.request(method,path.replace("/api/","/v1/"),body)
            except RemoteError as exc:
                if exc.status in (400,409):return exc.status,{"error":str(exc)}
                raise
        if method == "GET" and path in ("/health", "/ready"):
            return 200, self.client.request("GET", "/health")
        if method == "GET" and path in ("/api/state", "/api/session"):
            return 200, self.client.request("GET", "/v1/" + path.rsplit("/", 1)[-1])
        if path == "/api/messages" and method in ("GET","POST"):
            from baby_arcus.transport import RemoteError
            try:
                return 200,self.client.request(method,"/v1/messages",body)
            except RemoteError as exc:
                if exc.status in (400,409): return exc.status,{"error":str(exc)}
                raise
        if method == "POST" and path == "/api/action":
            fields(body, ("request_id", "action"))
            from baby_arcus.transport import RemoteError
            try:
                return 200, self.client.request("POST", "/v1/action", {**body, "source": "human"})
            except RemoteError as exc:
                if exc.status in (400, 409):
                    return exc.status, {"error": str(exc)}
                raise
        files = {"/": ("playroom.html", "text/html; charset=utf-8"),
                 "/conversation.js": ("conversation.js", "text/javascript; charset=utf-8"),
                 "/arcus-renderer.js": ("arcus-renderer.js", "text/javascript; charset=utf-8"),
                 "/desktop-bridge.js": ("desktop-bridge.js", "text/javascript; charset=utf-8"),
                 "/environment-renderer.js": ("environment-renderer.js", "text/javascript; charset=utf-8"),
                 "/playroom.js": ("playroom.js", "text/javascript; charset=utf-8"),
                 "/playroom.css": ("playroom.css", "text/css; charset=utf-8"),
                 "/arcus-body.png": ("arcus-body.png", "image/png"),
                 "/arcus-lying.png": ("arcus-lying.png", "image/png"),
                 "/arcus-sitting.png": ("arcus-sitting.png", "image/png")}
        if method == "GET" and path in files:
            name, content_type = files[path]
            return 200, StaticResponse((self.web / name).read_bytes(), content_type)
        raise KeyError(path)


def viewer_server(port, application, audit=None, host="127.0.0.1", internal_host=None, internal_token=None):
    # LocalHandler below enforces the published localhost Host/Origin boundary.
    # Container binding may be 0.0.0.0 while the published host port is loopback-only.
    server = serve(host, port, application,audit=audit,application_auth=True)
    base = server.RequestHandlerClass

    class LocalHandler(base):
        def handle_command(self):
            expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            host = self.headers.get("Host", "")
            if internal_host and host == internal_host and internal_token:
                import hmac
                if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+internal_token):
                    return self.reply(403, {'error':'Service authentication required'})
                if self.headers.get('Origin'):
                    return self.reply(403, {'error':'Internal service route does not accept browser origins'})
            elif host not in expected:
                return self.reply(403, {"error": "Local host required"})
            origin = self.headers.get("Origin")
            if self.command == "POST" and origin and origin != "http://" + host:
                return self.reply(403, {"error": "Same-origin interaction required"})
            if self.command == "POST" and self.headers.get("Sec-Fetch-Site") == "cross-site":
                return self.reply(403, {"error": "Same-origin interaction required"})
            return super().handle_command()

        do_GET = handle_command
        do_POST = handle_command

    server.RequestHandlerClass = LocalHandler
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("all", "simulation", "viewer"), default="all")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--simulation-port", type=int, default=8891)
    parser.add_argument("--simulation-host", default="127.0.0.1")
    parser.add_argument("--simulation-url", default=None)
    parser.add_argument("--state-root", default="runs/arcus_playroom/entity-state")
    parser.add_argument("--tool-port", type=int, default=8892)
    parser.add_argument('--connect-model',action='store_true')
    args = parser.parse_args(argv)
    token = os.environ.get("ARCUS_PLAYROOM_TOKEN", "")
    if args.role != "all" and not token:
        parser.error("Separate services require ARCUS_PLAYROOM_TOKEN in their environment")
    if args.role == "all" and args.simulation_host != "127.0.0.1":
        parser.error("Combined local mode binds to loopback only")
    token = token or secrets.token_urlsafe(32)
    tool_token = os.environ.get("ARCUS_BODY_TOOL_TOKEN", "")
    if tool_token and secrets.compare_digest(tool_token, token):
        parser.error("Body tools require a different credential from human controls")
    app = None
    servers = []
    try:
        if args.role in ("all", "simulation"):
            app = PlayroomApplication(args.state_root)
            if args.connect_model:
                import sys
                from baby_arcus.live_interaction_policy import LiveInteractionPolicy
                from baby_arcus.body_controller_config import configuration
                project=Path(__file__).resolve().parents[2]
                python=os.environ.get('ARCUS_MODEL_PYTHON',str(project/'.venv/Scripts/python.exe') if os.name=='nt' else sys.executable)
                app.policy=LiveInteractionPolicy(app,python,workdir=project,**configuration(project))
                from baby_arcus.language_runtime import LanguageRuntime
                app.language=LanguageRuntime(app,python,project)
                from baby_arcus.visual_runtime import VisualRuntime
                app.visual=VisualRuntime(app,python,project)
                from baby_arcus.rest_runtime import RestRuntime
                app.rest=RestRuntime(app,project)
                from baby_arcus.curiosity_runtime import CuriosityRuntime
                app.curiosity=CuriosityRuntime(app,project)
                from baby_arcus.shared_runtime import SharedRuntime
                app.shared=SharedRuntime(app,python,project)
            server = serve(args.simulation_host, args.simulation_port, app, token,audit=app.audit)
            servers.append(server)
            threading.Thread(target=app.run_clock, daemon=True).start()
            if tool_token:
                servers.append(serve(args.simulation_host, args.tool_port, BodyToolsApplication(app), tool_token,audit=app.audit))
        if args.role in ("all", "viewer"):
            url = args.simulation_url or f"http://127.0.0.1:{args.simulation_port}"
            if args.role == "all":
                url = f"http://127.0.0.1:{servers[0].server_port}"
            servers.append(viewer_server(args.port, PlayroomViewer(url, token),audit=app.audit if app else AuditLog(Path(args.state_root)/"audit","viewer")))
        for server in servers[:-1]:
            threading.Thread(target=server.serve_forever, daemon=True).start()
        print(json.dumps({"role": args.role, "url": f"http://127.0.0.1:{servers[-1].server_port}",
                          "model_connected": False}), flush=True)
        servers[-1].serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if app:
            app.close()
        for server in servers[:-1]:
            server.shutdown()
        for server in servers:
            server.server_close()


if __name__ == "__main__":
    main()
