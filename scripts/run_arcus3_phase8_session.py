"""One foreground training window with sequential evaluation handoffs. No scheduler."""
import argparse,json,subprocess,sys,uuid
from datetime import datetime,timezone,timedelta
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read
from arcus3.campaign import in_window,window_end
from baby_arcus.language_stream import atomic_json

def run(a):
    cfg=read('configs/arcus3/backbone_adaptation.json');windows=read('configs/arcus3/training_windows.json')
    if not cfg['campaign_enabled'] or not in_window(windows):raise ValueError('Campaign/windows not enabled')
    if not read('configs/arcus3/phase8_sources.json')['ready']:raise ValueError('Combined sources not ready')
    end=datetime.fromisoformat(a.stop_at.replace('Z','+00:00'))
    if end.tzinfo is None or not 0<(end-datetime.now(timezone.utc)).total_seconds()<=86400:raise ValueError('Session deadline required within 24 hours')
    end=min(end,window_end(windows))
    root=Path(a.root)
    if root.exists():raise ValueError('Fresh session root required')
    root.mkdir(parents=True)
    resume=a.resume;evaluation=None
    def invoke(mode,child,extra):
        remaining=end-datetime.now(timezone.utc)
        if remaining.total_seconds()<600:raise RuntimeError('Insufficient checkpoint/evaluation deadline margin')
        stop=min(end,datetime.now(timezone.utc)+timedelta(minutes=29)) if mode=='baseline' else end
        cmd=['powershell','-NoProfile','-File','scripts/start_arcus3.ps1','-Mode',mode,'-Root',str(child).replace('\\','/'),'-StopAt',stop.isoformat(),'-ConvertedPath',a.converted]+extra
        atomic_json(root/'session.json',{'active_run':str(child),'mode':mode,'resume':resume,'deadline':end.isoformat()})
        subprocess.run(cmd,check=True)
    while in_window(read('configs/arcus3/training_windows.json')) and datetime.now(timezone.utc)<end:
        if (root/'pause-training').exists():break
        child=Path('runs/arcus3')/('adaptation-session-'+uuid.uuid4().hex)
        extra=['-DataRoot',a.data,'-TeacherPath',a.teacher,'-PreflightReport',a.qualification_report]
        if resume:extra+=['-ResumePath',resume]
        if evaluation:extra+=['-EvaluationRoot',evaluation]
        invoke('adaptation',child,extra)
        report=read(child/'report.json');resume=str(child/'checkpoints'/Path(report['checkpoint']).name)
        atomic_json(root/'latest-session-checkpoint.json',{'checkpoint':resume,'manifest_sha256':report['checkpoint_manifest_sha256']})
        if report['complete']:break
        pending=report['state']['evaluation_pending']
        if not pending or (root/'pause-training').exists():break
        tier='full' if pending[0]=='baseline-full' else pending[0]
        evaluation=str(Path('runs/arcus3')/('baseline-phase8-session-'+uuid.uuid4().hex))
        invoke('baseline',Path(evaluation),['-ExpandedPath',resume,'-EvaluationTier',tier])
    atomic_json(root/'session-result.json',{'last_checkpoint':resume,'last_evaluation':evaluation,'stopped':True,'automatic_next_window':False})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','data','teacher','converted','qualification-report','stop-at'):p.add_argument('--'+name,required=True)
    p.add_argument('--resume');run(p.parse_args())
