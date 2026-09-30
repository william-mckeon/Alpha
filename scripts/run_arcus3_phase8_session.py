"""One foreground training window with sequential evaluation handoffs. No scheduler."""
import argparse,json,subprocess,sys,uuid
from datetime import datetime,timezone,timedelta
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read
from arcus3.campaign import in_window,window_end,session_deadline
from baby_arcus.language_stream import atomic_json

def run(a):
    cfg=read('configs/arcus3/backbone_adaptation.json');windows=read('configs/arcus3/training_windows.json')
    storage=read('configs/arcus3/phase8_storage.json')
    if not storage['ready']:raise ValueError('External storage and migration verification are pending')
    if not cfg['campaign_enabled']:raise ValueError('Campaign not enabled')
    chat=windows.get('mode')=='chat-deadline'
    if not chat and not in_window(windows):raise ValueError('Training window not enabled')
    if not read('configs/arcus3/phase8_sources.json')['ready']:raise ValueError('Combined sources not ready')
    end=session_deadline(a.stop_at,windows)
    if not chat:end=min(end,window_end(windows))
    root=Path(a.root)
    if root.exists():raise ValueError('Fresh session root required')
    root.mkdir(parents=True)
    if chat:
        windows={**windows,'enabled':True,'start_at':datetime.now(timezone.utc).isoformat(),'stop_at':end.isoformat()}
        atomic_json(root/'session-policy.json',windows)
    checkpoint_root=Path(storage['checkpoint_root']).resolve()
    if not checkpoint_root.is_dir():raise ValueError('Checkpoint folder missing')
    resume=a.resume;evaluation=None;stop_reason='window_or_user_pause'
    if not resume and (checkpoint_root/'latest.json').exists():
        from arcus3.checkpoint import digest
        from arcus3.expanded_checkpoint import verify
        pointer=read(checkpoint_root/'latest.json');selected=checkpoint_root/pointer['generation']
        if selected.resolve().parent!=checkpoint_root or digest(selected/'manifest.json')!=pointer['manifest_sha256']:raise ValueError('Invalid checkpoint pointer')
        verify(selected,cfg['parent_sha256']);resume=str(selected)
    def invoke(mode,child,extra):
        nonlocal stop_reason
        remaining=end-datetime.now(timezone.utc)
        if remaining.total_seconds()<600:
            stop_reason='insufficient_deadline_margin';return False
        stop=min(end,datetime.now(timezone.utc)+timedelta(minutes=29)) if mode=='baseline' else end
        cmd=['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File','scripts/start_arcus3.ps1','-Mode',mode,'-Root',str(child).replace('\\','/'),'-StopAt',stop.isoformat(),'-ConvertedPath',a.converted]+extra
        if chat and mode=='adaptation':cmd+=['-WindowPolicy',str(root/'session-policy.json')]
        atomic_json(root/'session.json',{'active_run':str(child),'mode':mode,'resume':resume,'deadline':end.isoformat()})
        subprocess.run(cmd,check=True)
        return True
    while in_window(windows if chat else read('configs/arcus3/training_windows.json')) and datetime.now(timezone.utc)<end:
        if (root/'pause-training').exists():break
        child=Path('runs/arcus3')/('adaptation-session-'+uuid.uuid4().hex)
        extra=['-DataRoot',a.data,'-TeacherPath',a.teacher,'-PreflightReport',a.qualification_report,'-CheckpointRoot',str(checkpoint_root)]
        if resume:extra+=['-ResumePath',resume]
        if evaluation:extra+=['-EvaluationRoot',evaluation]
        if not invoke('adaptation',child,extra):break
        report=read(child/'report.json');resume=str(checkpoint_root/Path(report['checkpoint']).name)
        atomic_json(root/'latest-session-checkpoint.json',{'checkpoint':resume,'manifest_sha256':report['checkpoint_manifest_sha256']})
        if report['complete']:stop_reason='stage_complete';break
        pending=report['state']['evaluation_pending']
        if not pending or (root/'pause-training').exists():break
        tier='full' if pending[0]=='baseline-full' else pending[0]
        evaluation=str(Path('runs/arcus3')/('baseline-phase8-session-'+uuid.uuid4().hex))
        if not invoke('baseline',Path(evaluation),['-ExpandedPath',resume,'-EvaluationTier',tier]):
            evaluation=None;break
    atomic_json(root/'session-result.json',{'last_checkpoint':resume,'last_evaluation':evaluation,'stopped':True,'reason':stop_reason,'automatic_next_window':False})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','data','teacher','converted','qualification-report'):p.add_argument('--'+name,required=True)
    p.add_argument('--stop-at',help='Optional timezone-aware deadline; otherwise use the configured two-hour default')
    p.add_argument('--resume');run(p.parse_args())
