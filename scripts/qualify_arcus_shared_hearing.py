"""Learned hearing choices and isolated resumable DatasetForge playback over HTTP."""
import argparse,json,sys,tempfile,threading,uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.services.shared_worker import Worker
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.shared_experience import capture
from baby_arcus.transport import Client,serve
from baby_arcus.language_stream import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());root=Path(cfg['root']);manifest=json.loads((root/'candidate.json').read_text())
    with tempfile.TemporaryDirectory(prefix='arcus-hearing-') as folder:
        state=Path(folder)/'hearing.json';worker=Worker(a.config,manifest,hearing_state_path=state)
        # The actual authenticated endpoint supplies the action and cursor receipts.
        token=uuid.uuid4().hex;server=serve('127.0.0.1',0,worker,token)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        client=Client(f'http://127.0.0.1:{server.server_port}',token,timeout=60,attempts=1)
        app=PlayroomApplication();row=capture(app);app.close();choices=[]
        try:
            requests=[('listen','listen'),('pause listening','pause'),('resume listening','resume')]
            if worker.model.version>=7:requests.extend([('replay that','replay'),('restart listening','restart')])
            for text,expected in requests:
                row['hearing']=[{'text':text,'source':'simulated_hearing'}]
                reply=client.request('POST','/v1/shared/observe',row);action=reply['hearing_action']
                receipt=client.request('POST','/v1/shared/hearing',{'action':action}) if action else None
                choices.append({'text':text,'expected':expected,'chosen':action,'correct':action==expected,'receipt':receipt})
            first=client.request('POST','/v1/shared/hearing',{'action':'listen'})
            client.request('POST','/v1/shared/hearing',{'action':'pause'})
            paused=client.request('POST','/v1/shared/hearing',{'action':'listen'})
            client.request('POST','/v1/shared/hearing',{'action':'resume'})
            resumed=client.request('POST','/v1/shared/hearing',{'action':'listen'})
            replay=client.request('POST','/v1/shared/hearing',{'action':'replay'})
            worker.close();reloaded=Worker(a.config,manifest,hearing_state_path=state)
            try:
                restored=reloaded.hearing('replay')
                recovery=restored['cursor']==replay['cursor'] and restored['passage']==replay['passage']
            finally:reloaded.close()
            playback=bool(first['passage'] and resumed['passage'] and paused['passage'] is None and resumed['passage']['id']!=first['passage']['id'] and replay['passage']==resumed['passage'])
            report={'candidate':manifest,'choices':choices,'playback_passed':playback,'recovery':recovery,
                'hearing_passed':playback and recovery and all(item['correct'] for item in choices),'desktop_cursor_changed':False,'training_updates':0}
            atomic_json(root/'hearing-live.json',report);print(json.dumps({key:value for key,value in report.items() if key!='choices'}),flush=True)
        finally:server.shutdown();server.server_close();worker.close()
        if not report['hearing_passed']:raise SystemExit(1)

if __name__=='__main__':main()
