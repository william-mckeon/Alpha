"""Accelerated simulation at unchanged tick rates; not wall-clock live endurance."""
import argparse,hashlib,json,sys,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.embodiment import Embodiment
from baby_arcus.play_session import PlaySession
from baby_arcus.body_dynamics import pose
from baby_arcus.rest_runtime import predict
from baby_arcus.rest_environment import features,NAMES

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=4)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not 1<=args.hours<=24:parser.error('Hours must be between 1 and 24')
    project=Path(__file__).resolve().parents[1]
    cfg=json.loads((project/'configs/baby_arcus/rest.json').read_text());root=project/cfg['output']
    path=root/'policy.json';artifact=json.loads(path.read_text())
    w=PlaySession(Embodiment(height=.25,target_posture='lying',joint_positions=pose(0),previous_joints=pose(0),motor_mode='independent'))
    changes=[];restarts=0;decisions=0;last_sleep='awake'
    actions={'rest':'rest','alert':'alert','sleep':'sleep_when_ready','wake':'wake_voluntarily'}
    for tick in range(int(args.hours*36000)):
        w.step()
        if tick and tick%18000==0:
            before=w.body.record();w.body=Embodiment.restore(before)
            assert w.body.record()==before;restarts+=1
        if tick%cfg['interval_ticks']==0:
            name=NAMES[predict(artifact,features(w.body))];decisions+=1
            if name in actions:w.action({'kind':actions[name]})
            if w.body.sleep_state!=last_sleep:
                changes.append({'tick':tick,'sleep':w.body.sleep_state,'rest_need':w.body.rest_need})
                last_sleep=w.body.sleep_state
        assert 0<=w.body.rest_need<=1 and 0<=w.body.alertness<=1
        assert w.body.height<=.30
    sleep=sum(r['sleep']=='sleeping' for r in changes);wake=sum(r['sleep']=='awake' for r in changes)
    report={'passed':sleep>=2 and wake>=2 and len(changes)<=12*args.hours,
            'platform':platform.system(),'python':platform.python_version(),
            'simulated_hours':args.hours,'decisions':decisions,'sleep_entries':sleep,'wake_entries':wake,
            'record_roundtrip_checks':restarts,'changes':changes,
            'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope':'Accelerated quiet, already-lying policy simulation at unchanged 0.1-second dynamics. '
                    'Does not bypass or extend the live controller 600-decision limit; does not prove unattended live endurance.'}
    (args.output or root/'endurance-simulation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
