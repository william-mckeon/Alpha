"""Isolated live HTTP service qualification using a fresh tiny CPU fixture only."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.runtime_contract import require_container
require_container()
from baby_arcus.shared_factory import initialize
from baby_arcus.shared_curriculum import example
from baby_arcus.language_stream import atomic_json
from baby_arcus.transport import Client


def qualify(root):
    root=Path(root).resolve()
    base=Path('/app/runs/test2').resolve()
    if base not in root.parents or (root.exists() and any(root.iterdir())):
        raise ValueError('Use a new empty isolated fixture directory')
    root.mkdir(parents=True,exist_ok=True)
    run=root/'learner'; config=root/'config.json'
    cfg={'schema':'arcus-test2-v1','initialization':'random','root':str(run),'seed':2101,
         'preset':'tiny','text_dim':16,'depth_capacity':1.,'encoding':'o200k_base',
         'tiktoken_version':importlib.metadata.version('tiktoken'),'learning_rate':.00001,
         'max_storage_bytes':1024**3,'max_graph_records':100,
         'idle_learning':{'auto_resume':True,'idle_seconds':60,'chunk_updates':3,'checkpoint_every':3,'session_updates':3}}
    atomic_json(config,cfg); initial=initialize(str(config))
    secret=secrets.token_urlsafe(32); env=dict(os.environ,ARCUS_TEST2_TOKEN=secret)
    processes=[]; streams=[]
    def start(name,args):
        stream=(root/(name+'.log')).open('w'); streams.append(stream)
        process=subprocess.Popen([sys.executable,'-m',*args],stdout=stream,stderr=subprocess.STDOUT,env=env)
        processes.append(process)
    def ready(client,path='/health'):
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            if any(p.poll() is not None for p in processes): raise RuntimeError('Fixture service exited; inspect logs')
            try: return client.request('GET',path)
            except Exception: time.sleep(.2)
        raise TimeoutError('Fixture readiness timed out')
    try:
        start('learner',['baby_arcus.services.shared_trainer','--config',str(config),'--port','18931'])
        learner=Client('http://127.0.0.1:18931',secret,timeout=30,attempts=1)
        health=ready(learner)
        assert health['runtime']['container'] is True
        start('playroom',['baby_arcus.services.test2_playroom','--config',str(config),
                         '--port','18930','--learner-url','http://127.0.0.1:18931'])
        viewer=Client('http://127.0.0.1:18930',timeout=30,attempts=1)
        ready(viewer)
        snapshot=viewer.request('GET','/api/test2')
        assert snapshot['idle_learning']['enabled'] is False
        row,_,_=example(0,'training','commands')
        decision=learner.request('POST','/infer',{'row':row})
        assert decision['generation']==initial['generation']
        paused=viewer.request('POST','/api/test2/idle',{'action':'pause'})
        assert paused['enabled'] is False
        after=json.loads((run/'candidate.json').read_text())
        assert after['generation']==initial['generation'] and after['updates']==0
        report={'fixture':True,'cpu_only':True,'live_learner_http':True,'live_playroom_http':True,
                'inference_generation_verified':True,'explicit_pause_preserved':True,
                'production_checkpoint_accessed':False,'updates':0,'mastery_established':False}
        atomic_json(root/'report.json',report)
        return report
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        for stream in streams: stream.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--root',required=True)
    print(json.dumps(qualify(parser.parse_args().root)))
