"""Read-only, loopback viewer. Service credentials stay on the server."""
from pathlib import Path
from urllib.parse import urlsplit,parse_qs
from baby_arcus.transport import Client,StaticResponse
from baby_arcus.contracts import identifier,ContractError

class DashboardApplication:
    def __init__(self,urls,token=""):
        self.clients={name:Client(url,token,timeout=10) for name,url in urls.items()}
        self.web=Path(__file__).resolve().parent.parent/"web"

    def __call__(self,method,path,body):
        if method!="GET":
            raise KeyError(path)
        parsed=urlsplit(path)
        route=parsed.path
        if route in ("/health","/ready"):
            return 200,{"service":"dashboard","ready":True}
        if route=="/api/status":
            return 200,self.clients["controller"].request("GET","/v1/status")
        if route=="/api/reports":
            return 200,self.clients["controller"].request("GET","/v1/reports")
        if route.startswith("/api/reports/"):
            run_id=identifier(route.removeprefix("/api/reports/"))
            return 200,self.clients["controller"].request("GET","/v1/reports/"+run_id)
        if route=="/api/frame":
            query=parse_qs(parsed.query)
            episode=identifier(query["episode"][0])
            index=int(query.get("index",["-1"])[0])
            replay=self.clients["simulation"].request("GET","/v1/episodes/"+episode+"/replay")
            refs=replay["artifact_ids"]
            if not -len(refs)<=index<len(refs):
                raise ContractError("Frame outside replay")
            artifact=self.clients["artifact"].request("GET","/v1/artifacts/"+refs[index])
            initial=artifact if refs[index]==refs[0] else self.clients["artifact"].request("GET","/v1/artifacts/"+refs[0])
            from baby_arcus.world import World
            from baby_arcus.observations import observations
            state=artifact["payload"]["state"]
            return 200,{"state":state,"observations":observations(World.restore(state)),
                        "frame_count":len(refs),"artifact_id":refs[index],
                        "checkpoint_id":initial["payload"].get("checkpoint_id"),"run_id":initial["payload"].get("run_id")}
        name="index.html" if route=="/" else route.removeprefix("/")
        allowed={"index.html":"text/html; charset=utf-8","styles.css":"text/css; charset=utf-8",
                 **{n:"text/javascript; charset=utf-8" for n in ("app.js","api.js","world.js","agent-panel.js","replay.js","training-panel.js")}}
        if name not in allowed:
            raise KeyError(path)
        return 200,StaticResponse((self.web/name).read_bytes(),allowed[name])
