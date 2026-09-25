"""Live loopback HTTP inference using a read-only 39k parent and isolated cache."""
import json
import sys
import threading
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from baby_arcus.services.shared_trainer import Learner
    from baby_arcus.transport import serve,Client
    from baby_arcus.shared_curriculum import example
    from baby_arcus.shared_checkpoint import digest
    torch.set_num_threads(2)
    parent=Path('/parent');root=Path('/app/runs/test2/efficiency-session');root.mkdir(parents=True)
    manifest=json.loads((parent/'candidate.json').read_text())
    if manifest['updates']!=39000:raise ValueError('Wrong checkpoint')
    (root/(manifest['generation']+'.pt')).symlink_to(parent/(manifest['generation']+'.pt'))
    (root/'candidate.json').write_text(json.dumps(manifest))
    cfg=json.loads((parent/'experiment.json').read_text());cfg['root']=str(root)
    for name in ('experiment.json','config.json'):(root/name).write_text(json.dumps(cfg))
    app=Learner(str(root/'config.json'))
    server=serve('127.0.0.1',0,app,'fixture-session-token')
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    client=Client('http://127.0.0.1:'+str(server.server_port),'fixture-session-token',timeout=180,attempts=1)
    report={'candidate':manifest,'requests':[],'complete':False}
    try:
        row,_,_=example(1,'training','commands')
        for index in range(2):
            start=time.perf_counter();result=client.request('POST','/infer',{'row':row})
            report['requests'].append({'seconds':time.perf_counter()-start,'result':result})
            assert next(app.session.model.parameters()).device.type=='cpu'
        assert report['requests'][0]['result']==report['requests'][1]['result']
        report['health']=client.request('GET','/health')
        report['memory']=client.request('GET','/memory')
        assert report['memory']['memory']['parameter_count']==151946954
        assert report['memory']['generation']==manifest['generation']
        from types import SimpleNamespace
        from baby_arcus.services.test2_playroom import Application
        viewer=Application(SimpleNamespace(client=client))
        status,payload=viewer('GET','/api/test2/memory',{})
        assert status==200 and payload['memory']['parameter_count']==151946954
        report['viewer_memory_proxy_verified']=True
        report['checkpoint_bytes']=(parent/(manifest['generation']+'.pt')).stat().st_size
        report['inference_artifact_bytes']=(root/'inference-artifacts'/(manifest['generation']+'.pt')).stat().st_size
        report['checkpoint_unchanged']=digest(parent/(manifest['generation']+'.pt'))==manifest['sha256']
        report['complete']=report['checkpoint_unchanged']
    finally:
        server.shutdown();server.server_close();app.session.invalidate()
        Path('/evidence/session-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
