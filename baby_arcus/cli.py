"""Run isolated services or deterministic lesson diagnostics."""
import argparse
import json
import os
from pathlib import Path
from baby_arcus.config import Settings
from baby_arcus.contracts import ContractError
from baby_arcus.messages import validate_catalog
from baby_arcus.transport import Client, serve
from baby_arcus.services.artifacts import ArtifactApplication
from baby_arcus.services.simulation import SimulationApplication
from baby_arcus.lessons import generate, validate_solvable
from baby_arcus.baselines import scripted

def main(argv=None):
    parser = argparse.ArgumentParser(prog="baby-arcus")
    sub = parser.add_subparsers(dest="command",required=True)
    service = sub.add_parser("serve")
    service.add_argument("service",choices=("simulation","artifacts","inference","training","controller","evaluator","dashboard"))
    service.add_argument("--config")
    service.add_argument("--host")
    service.add_argument("--port",type=int)
    service.add_argument("--state-root")
    service.add_argument("--artifact-url")
    service.add_argument("--simulation-url")
    service.add_argument("--inference-url")
    service.add_argument("--training-url")
    service.add_argument("--controller-url")
    service.add_argument("--evaluator-url")
    service.add_argument("--device",choices=("cpu","cuda"))
    service.add_argument("--read-only-network",action="store_true",help="Container viewer only; publish its port on loopback")
    diagnostic = sub.add_parser("diagnose")
    diagnostic.add_argument("--episodes",type=int,default=100)
    args = parser.parse_args(argv)
    if args.command == "diagnose":
        if not 1 <= args.episodes <= 10000:
            parser.error("episodes must be between 1 and 10000")
        results = {}
        for family in ("switch_delivery","clue_search"):
            wins = 0
            for seed in range(args.episodes):
                world = generate(family,seed)
                if not validate_solvable(world):
                    raise RuntimeError("Unsolvable lesson")
                while not (world.terminated or world.truncated):
                    world.advance(scripted(world))
                wins += int(world.success)
            results[family] = {"successes":wins,"episodes":args.episodes,"actor":"scripted_diagnostic"}
        print(json.dumps(results))
        return 0 if all(v["successes"] == args.episodes for v in results.values()) else 1
    cfg = Settings.load(args.config)
    if args.service in ("controller","evaluator"):
        from baby_arcus.evaluation import validate_definition
        validate_definition(cfg.evaluation_config)
    if args.config:
        validate_catalog(cfg.signal_config)
        curriculum = json.loads(Path(cfg.curriculum_config).read_text())
        if curriculum != {"schema_version":1,"world_version":"baby-grid-v2",
                           "families":["switch_delivery","clue_search"],"max_steps":64}:
            raise ContractError("Unsupported curriculum configuration")
    host = args.host or cfg.host
    port = args.port if args.port is not None else getattr(cfg,"artifact_port" if args.service == "artifacts" else args.service+"_port")
    if not 0 <= port <= 65535:
        raise ContractError("Port outside range")
    root = Path(args.state_root or cfg.state_root)
    token = os.environ.get("BABY_ARCUS_TOKEN","")
    urls={name:getattr(args,name+"_url") or getattr(cfg,name+"_url") for name in ("artifact","simulation","inference","training","controller","evaluator")}
    if args.service == "artifacts":
        app=ArtifactApplication(root/"artifacts")
    elif args.service == "simulation":
        app=SimulationApplication(root/"simulation",Client(urls["artifact"],token,timeout=cfg.request_timeout))
    elif args.service in ("inference","training"):
        from baby_arcus.services.worker import WorkerApplication
        app=WorkerApplication(args.service,root/args.service,urls["artifact"],urls["controller"],token,args.device or cfg.device)
    elif args.service == "controller":
        from baby_arcus.services.controller import ControllerApplication
        app=ControllerApplication(root/"controller",{("artifacts" if k=="artifact" else k):v for k,v in urls.items()},token)
    elif args.service == "evaluator":
        from baby_arcus.services.evaluator import EvaluatorApplication
        # Logging can be deployed alongside an older, stateless evaluator.
        from inspect import signature
        options={"root":root/"evaluator"} if "root" in signature(EvaluatorApplication).parameters else {}
        app=EvaluatorApplication(urls["simulation"],urls["inference"],token,**options)
    else:
        from baby_arcus.services.dashboard import DashboardApplication
        if host not in ("127.0.0.1","localhost","::1") and not args.read_only_network:
            raise ContractError("Dashboard is loopback-only until an authenticated gateway is configured")
        app=DashboardApplication(urls,token)
    from baby_arcus.storage import monitored
    dispatch=app if args.service=="dashboard" else monitored(app,root/args.service)
    from baby_arcus.audit import AuditLog
    audit=AuditLog(root/"audit",args.service)
    original_dispatch=dispatch
    def dispatch(method,path,body):
        status,result=original_dispatch(method,path,body)
        if path in ("/health","/ready") and isinstance(result,dict):
            result={**result,"audit":audit.status()}
        return status,result
    server = serve(host,port,dispatch,"" if args.service=="dashboard" else token,
                   readonly_network=args.service=="dashboard" and args.read_only_network,audit=audit)
    print(json.dumps({"service":args.service,"host":host,"port":server.server_address[1]}),flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        audit.close()
        server.server_close()
        if hasattr(app,"unload"):
            app.unload()
        if hasattr(app,"close"):
            app.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
