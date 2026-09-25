"""Read-only release evaluation in controlled Docker, with durable stage reports."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--capacity',type=float,required=True)
    args=p.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    from baby_arcus.runtime_contract import require_gpu
    require_gpu()
    import torch
    from baby_arcus.gpu_job_control import gpu_job
    from baby_arcus.language_stream import atomic_json
    from baby_arcus.shared_checkpoint import load,digest
    from scripts.alpha_evaluation_capacity import configure
    cfgpath='configs/baby_arcus/alpha_idle.container.json'
    cfg=json.loads(Path(cfgpath).read_text()); root=Path(cfg['root'])
    candidate=json.loads((root/'candidate.json').read_text())
    expected='9e6f8e21bc4b5e70d94d92cfcdff2872980e28c6585ebcaabad11a0fbb810520'
    if candidate['sha256']!=expected or candidate['updates']!=37000:
        raise ValueError('Expected immutable Alpha-1.0.0 release')
    report={'complete':False,'candidate':candidate,'capacity':args.capacity,'training_updates':0,
            'stages':{},'limitations':['Single checkpoint and small synthetic cohorts; not general intelligence.',
            'GPU telemetry is whole-device sampling, not per-model energy accounting.',
            'Object continuity/restart and desktop GUI qualification are separate legacy suites.']}
    stop=threading.Event()
    def sample():
        with (out/'gpu-telemetry.jsonl').open('a') as log:
            while not stop.is_set():
                try:
                    r=subprocess.run(['nvidia-smi','--query-gpu=timestamp,power.draw,utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                    log.write(json.dumps({'time':time.time(),'sample':r.stdout.strip(),'exit':r.returncode})+'\n');log.flush()
                except Exception as e: log.write(json.dumps({'error':str(e)})+'\n');log.flush()
                stop.wait(2)
    monitor=threading.Thread(target=sample,daemon=True); monitor.start()
    torch.set_num_threads(2); begin=time.monotonic()
    def stage(name,fn):
        report['current_stage']=name; atomic_json(out/'report.json',report)
        print(json.dumps({'starting':name}),flush=True)
        t=time.monotonic(); torch.cuda.reset_peak_memory_stats()
        try:
            result=fn()
            report['stages'][name]={'complete':True,'result':result}
        except Exception as e:
            traceback.print_exc()
            report['stages'][name]={'complete':False,'error':str(e)}
        report['stages'][name].update(seconds=time.monotonic()-t,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        atomic_json(out/'report.json',report)
        gc.collect();torch.cuda.empty_cache()
    try:
        with gpu_job():
            if digest(root/(candidate['generation']+'.pt'))!=expected:raise ValueError('Release hash mismatch')
            torch.cuda.set_per_process_memory_fraction(.5,0)
            from scripts.evaluate_arcus_baseline_parity import evaluate
            stage('retention',lambda:evaluate(cfgpath,'confirmation',20,60,report_directory=out,evaluation_capacity=args.capacity))
            from scripts.evaluate_alpha_coding import evaluate as coding
            stage('coding',lambda:coding(cfgpath,out/'coding',evaluation_capacity=args.capacity))
            from scripts.evaluate_alpha_tool_discovery import evaluate as discovery
            stage('tool_retriever',discovery)
            def pixels():
                from baby_arcus.object_perception_learning import dataset,assess
                model,data=load(root,candidate,'cuda'); del data
                routing=configure(model,args.capacity)
                result=assess(model.perception,model.core,dataset(78291,1000),'cuda')
                result.update(seed=78291,split='confirmation',routing=routing)
                result['passed']=result['count_accuracy']>=.9 and result['ball_iou']>=.75 and result['surface_color_accuracy']>=.9 and result['count_accuracy']-result['blank_count_accuracy']>=.2
                atomic_json(out/'pixels.json',result);return result
            stage('pixels',pixels)
            def curiosity():
                from scripts.evaluate_arcus_shared_curiosity import main as run
                old=sys.argv
                try:
                    sys.argv=['curiosity','--config',cfgpath,'--output',str(out/'curiosity.json'),'--evaluation-capacity',str(args.capacity)]
                    try:run()
                    except SystemExit as e:
                        if e.code!=1 or not (out/'curiosity.json').exists():raise
                    return json.loads((out/'curiosity.json').read_text())
                finally:sys.argv=old
            stage('curiosity',curiosity)
            report['checkpoint_unchanged']=digest(root/(candidate['generation']+'.pt'))==expected
            report['complete']=all(s['complete'] for s in report['stages'].values()) and report['checkpoint_unchanged']
    finally:
        stop.set();monitor.join(timeout=6)
        report['seconds']=time.monotonic()-begin; report['current_stage']='finished'
        atomic_json(out/'report.json',report)
    if not report['complete']:raise SystemExit(1)


if __name__=='__main__':main()
