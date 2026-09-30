"""User-controlled, sequential production coordinator. No retries or stage advancement."""
import argparse,json,subprocess,sys,uuid
from datetime import datetime,timezone,timedelta
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read,REVISION
from arcus3.checkpoint import digest
from arcus3.expanded_checkpoint import verify
from arcus3.production import validate_policy,identity,batch_receipt,accept_donor_receipt,disk_budget
from baby_arcus.language_stream import atomic_json

def run(a):
    policy=validate_policy(read(a.policy));runtime=read(a.runtime)
    proof=read(a.qualification)
    if not proof.get('qualified') or proof.get('image_id')!=runtime['image_id'] or proof.get('policy_sha256')!=identity(policy):
        raise ValueError('Matching production qualification required')
    for name,expected in proof['host_files'].items():
        if digest(name)!=expected:raise ValueError('Qualified host source changed: '+name)
    if (runtime['memory'],runtime['cpus'],runtime['pids'],runtime['cuda_fraction'],runtime['memory_watchdog'])!=('8g',2,128,.7,False):
        raise ValueError('Production runtime limits changed')
    root=Path(a.root).resolve();workspace=Path(__file__).resolve().parents[1]
    if root.parent!=workspace/'runs'/'arcus3':raise ValueError('Owned run root required')
    root.mkdir(parents=True,exist_ok=False)
    settings=read('configs/arcus3/phase8_storage.json');base=Path(settings['external_root']).resolve()
    cache=base/'production-cache-v1';checkpoints=base/'checkpoints-phase8-production-001'
    checkpoints.mkdir(exist_ok=True);benchmarks=cache/'benchmarks'
    donor=workspace/'artifacts'/'arcus3'/'donor'/REVISION
    converted=workspace/'runs/arcus3/conversion-phase5-001/converted'
    if a.continue_from:
        saved=read(Path(a.continue_from)/'controller-state.json')
        if saved['policy_sha256']!=identity(policy):raise ValueError('Changed continuation policy')
    else:
        report=read(a.resume_report)
        saved={'checkpoint':a.resume,'data':str(workspace/policy['initial_data']),'teacher':str(workspace/policy['initial_teacher']),
               'state':report['state'],'policy_sha256':identity(policy),'transition':None,'evaluation':None,
               'queued_batches':[str(Path(a.first_batch).resolve())] if a.first_batch else []}
        if digest(Path(a.resume)/'manifest.json')!=report['checkpoint_manifest_sha256']:raise ValueError('Resume report mismatch')
    checkpoint=Path(saved['checkpoint']);verify(checkpoint,saved['state']['parent_sha256'])
    def persist():atomic_json(root/'controller-state.json',saved)
    persist()
    def stopped():
        return (root/'pause-training').exists() or (a.stop_at and datetime.now(timezone.utc)>=datetime.fromisoformat(a.stop_at.replace('Z','+00:00')))
    def invoke(mode,extra):
        if stopped():raise InterruptedError('User pause or requested deadline')
        child=workspace/'runs/arcus3'/('adaptation-production-'+uuid.uuid4().hex)
        end=datetime.now(timezone.utc)+timedelta(hours=23 if mode in ('adaptation','teacher-production','donor-baseline') else 0,minutes=0 if mode in ('adaptation','teacher-production','donor-baseline') else 29)
        if a.stop_at:end=min(end,datetime.fromisoformat(a.stop_at.replace('Z','+00:00')))
        window=root/'worker-window.json';atomic_json(window,{'mode':'chat-deadline','enabled':True,'start_at':datetime.now(timezone.utc).isoformat(),'stop_at':end.isoformat()})
        atomic_json(root/'session.json',{'active_run':str(child),'mode':mode,'deadline':a.stop_at})
        cmd=['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File','scripts/start_arcus3.ps1',
             '-Mode',mode,'-Root',str(child.relative_to(workspace)).replace('\\','/'),'-StopAt',end.isoformat(),'-RuntimeConfig',a.evaluation_runtime if mode=='donor-baseline' else a.runtime]+extra
        if mode=='adaptation':cmd+=['-WindowPolicy',str(window)]
        with (root/'launcher.log').open('a') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            if stopped():raise InterruptedError('User pause or requested deadline')
            raise RuntimeError('Worker failed; inspect '+str(child)+' before any retry')
        return child
    def donor_eval(cp,tier):
        args=['-BenchmarksPath',str(benchmarks),'-EvaluationTier',tier]
        if cp:args+=['-ConvertedPath',str(converted),'-ExpandedPath',str(cp)]
        result=invoke('donor-baseline',args);scores=read(result/'donor-scores.json')
        accept_donor_receipt(scores,digest(Path(cp)/'manifest.json') if cp else 'donor',tier,policy)
        if scores['benchmark_manifest_sha256']!=digest(benchmarks/'manifest.json'):raise ValueError('Benchmark identity mismatch')
        return scores
    def evaluate(cp,pending):
        tier='full' if pending==['baseline-full'] else pending[0]
        result=invoke('baseline',['-ConvertedPath',str(converted),'-ExpandedPath',str(cp),'-EvaluationTier',tier])
        atomic_json(result/'donor-scores.json',donor_eval(cp,tier))
        return str(result)
    try:
        if not saved['state'].get('production') and not saved.get('transition'):
            receipt=batch_receipt(checkpoint,saved['data'],saved['teacher'],policy,'pilot-migration')
            receipt['baseline_donor']=donor_eval(None,'light')
            receipt['baseline_arcus']=donor_eval(checkpoint,'light')
            path=root/'migration.json';atomic_json(path,receipt);saved['transition']=str(path);persist()
        while not stopped():
            checkpoint=Path(saved['checkpoint']);state=saved['state']
            if state['evaluation_pending'] and not saved.get('evaluation'):
                saved['evaluation']=evaluate(checkpoint,state['evaluation_pending']);persist()
            if state.get('batch_complete') and not state['evaluation_pending'] and not saved.get('transition'):
                from scripts.prepare_arcus3_production import catalog
                from arcus3.production_data import prepare
                from scripts.verify_arcus3_teacher_cache import verify_cache
                from arcus3.production_cache import reclaim
                protected=[state['data_sha256']]+[digest(Path(p)/'manifest.json') for p in saved.get('queued_batches',[])]
                if saved.get('preparing') and (Path(saved['preparing']['data'])/'manifest.json').exists():protected.append(digest(Path(saved['preparing']['data'])/'manifest.json'))
                reclaim(cache,checkpoints,protected=protected)
                disk_budget(cache,policy['cache_limit_bytes'],policy['minimum_free_bytes'])
                pending=saved.get('preparing')
                if not pending:
                    data_path=Path(saved['queued_batches'].pop(0)) if saved.get('queued_batches') else cache/('batch-'+uuid.uuid4().hex)
                    if data_path.resolve().parent!=cache:raise ValueError('Batch must be inside managed cache')
                    pending={'data':str(data_path),'teacher':str(cache/('teacher-'+data_path.name.removeprefix('batch-')))}
                    saved['preparing']=pending;persist()
                data,teacher=Path(pending['data']),Path(pending['teacher'])
                if not (data/'manifest.json').exists():
                    sources=catalog(cache/'sources',base/'local-data-inbox')
                    prepare(data,donor,sources,cache/'production-exclusions.json',policy,cache)
                import sqlite3
                with sqlite3.connect(cache/'acquisition.sqlite') as db:
                    registered=db.execute('SELECT manifest FROM batches WHERE path=?',(str(data.resolve()),)).fetchone()
                db.close()
                if not registered or registered[0]!=digest(data/'manifest.json'):raise ValueError('Batch acquisition transaction was not committed')
                teacher.mkdir(exist_ok=True)
                if not (teacher/'manifest.json').exists():
                    invoke('teacher-production',['-DataRoot',str(data),'-TeacherOutput',str(teacher)])
                checked=verify_cache(data,teacher,donor);atomic_json(root/('teacher-verified-'+data.name+'.json'),checked)
                receipt=batch_receipt(checkpoint,data,teacher,policy,data.name)
                path=root/('transition-'+data.name+'.json');atomic_json(path,receipt)
                saved.update(data=str(data),teacher=str(teacher),transition=str(path));saved.pop('preparing',None);persist()
            args=['-DataRoot',saved['data'],'-TeacherPath',saved['teacher'],'-ConvertedPath',str(converted),
                  '-ResumePath',saved['checkpoint'],'-CheckpointRoot',str(checkpoints),'-ProductionPolicy',a.policy,
                  '-PreflightReport',policy['qualification_report']]
            if saved.get('transition'):args+=['-TransitionPath',saved['transition']]
            if saved.get('evaluation'):args+=['-EvaluationRoot',saved['evaluation']]
            child=invoke('adaptation',args);report=read(child/'report.json')
            cp=checkpoints/Path(report['checkpoint']).name
            verify(cp,report['state']['parent_sha256'])
            if digest(cp/'manifest.json')!=report['checkpoint_manifest_sha256'] or not report['frozen_unchanged']:raise ValueError('Checkpoint/frozen verification failed')
            saved.update(checkpoint=str(cp),state=report['state'],transition=None,evaluation=None);persist()
            if report['complete']:
                atomic_json(root/'session-result.json',{'stopped':True,'reason':'100m_review_required','checkpoint':str(cp),
                    'updates':report['state']['updates'],'input_tokens':report['state']['input_tokens'],'target_tokens':report['state']['target_tokens']})
                return
            if report.get('reason') not in ('batch_complete','evaluation_required','stage_boundary_no_overshoot'):
                # Only renew a completed internal lease; never turn another pause into a resume.
                worker=read(child/'runtime.json')
                lease_end=datetime.fromisoformat(worker['deadline'].replace('Z','+00:00'))
                flag=child/'pause-training'
                lease_pause=flag.exists() and 'Graceful deadline pause' in flag.read_text(encoding='utf-8-sig')
                if report.get('reason')=='Manual training pause' and lease_pause and datetime.now(timezone.utc)>=lease_end-timedelta(minutes=5) and not stopped():continue
                raise InterruptedError(report.get('reason','worker paused'))
        raise InterruptedError('User pause or requested deadline')
    except InterruptedError as e:
        atomic_json(root/'session-result.json',{'stopped':True,'reason':str(e),'checkpoint':saved['checkpoint'],
            'updates':saved['state']['updates'],'input_tokens':saved['state']['input_tokens']})
    except Exception as e:
        atomic_json(root/'session-result.json',{'stopped':True,'reason':'failure_requires_review','type':type(e).__name__,'detail':str(e)})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--runtime',required=True);p.add_argument('--evaluation-runtime',default='configs/arcus3/production_evaluation_runtime.json');p.add_argument('--qualification',required=True)
    p.add_argument('--policy',default='configs/arcus3/production.json');p.add_argument('--resume');p.add_argument('--resume-report');p.add_argument('--continue-from');p.add_argument('--stop-at');p.add_argument('--first-batch')
    run(p.parse_args())
