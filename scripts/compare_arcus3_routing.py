"""Sequential disposable fresh-model arms; never writes production checkpoints."""
import argparse,gc,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main(a):
    if not Path('/.dockerenv').exists() or os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1':
        raise RuntimeError('Docker CUDA only')
    import torch
    from safetensors.torch import load_file
    from arcus3.config import read,deadline,check_live
    from arcus3.checkpoint import digest
    from arcus3.donor import load,verify
    from arcus3.adapters import train_added_experts
    from arcus3.routing_objectives import configure
    from arcus3.backbone_adaptation import update
    from arcus3.evaluation import masked_nll,aggregate,load_suite
    from scripts.qualify_arcus3_training import frozen_digest
    from baby_arcus.gpu_job_control import gpu_job
    from baby_arcus.language_stream import atomic_json
    end=deadline(a.deadline);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cfg=read(a.config);data=Path(a.data);teacher=Path(a.teacher)
    manifest=read(data/'manifest.json');tm=read(teacher/'manifest.json')
    if tm['data_sha256']!=digest(data/'manifest.json'):raise ValueError('Teacher/data mismatch')
    rows=[];counts={}
    for shard in manifest['shards']:
        if digest(data/shard['path'])!=shard['sha256']:raise ValueError('Data changed')
        with (data/shard['path']).open() as source:
            for line in source:
                row=json.loads(line);category=row['source']
                if len(row['input_ids'])<=512 and counts.get(category,0)<6:
                    rows.append(row);counts[category]=counts.get(category,0)+1
    if len(rows)!=30 or len(counts)!=5:raise ValueError('Expected six short records from each source')
    verify(a.donor);suite,_=load_suite('/app')
    report={'complete':False,'campaign_updates':0,'disposable_updates_per_arm':len(rows),
            'training_row_hashes':[r['sha256'] for r in rows],'source_counts':counts,'arms':{},
            'limitations':'Small deterministic diagnostic: not a representative benchmark or proof of long-term recovery. Same seeded routers in all arms.'}
    atomic_json(out/'report.json',report)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_math_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    with gpu_job():
        torch.cuda.set_per_process_memory_fraction(.7)
        for name,objective,weights in [('control','selected-probability-v1',[1/6]*6),
                                       ('balance-only','selected-probability-v1',[1.]*6),
                                       ('paired-balance','paired-output-v2',[1.]*6)]:
            check_live(end,out);torch.manual_seed(cfg['seed']);started=time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            settings={**cfg,'routing_objective':objective,'router_layer_weights':weights}
            model,tokenizer=load(a.donor,converted=a.converted);train_added_experts(model)
            configure(model,settings,initialize=True);frozen=frozen_digest(model)
            def language():
                model.eval();result=[]
                with torch.no_grad():
                    for item in suite['language']:
                        check_live(end,out)
                        ids=tokenizer(item['text'],return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
                        result.append(masked_nll(model(input_ids=ids,use_cache=False).logits,ids,1))
                return aggregate(result)
            before=language();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],foreach=False)
            metrics=[]
            for row in rows:
                check_live(end,out);file=row['sha256']+'.safetensors'
                if tm['files'].get(file)!=digest(teacher/file):raise ValueError('Teacher shard changed')
                metrics.append(update(model,optimizer,row,load_file(str(teacher/file)),settings))
            after=language()
            report['arms'][name]={'before':before,'after':after,'input_tokens':sum(x['input_tokens'] for x in metrics),
                'updates':len(metrics),'metrics':metrics,'frozen_unchanged':frozen_digest(model)==frozen,
                'seconds':time.monotonic()-started,'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
            atomic_json(out/'report.json',report)
            print(json.dumps({'arm':name,'before':before,'after':after}),flush=True)
            del optimizer,model,tokenizer,metrics;gc.collect();torch.cuda.empty_cache()
        report['complete']=True
        report['bounded_check_passed']=all(v['frozen_unchanged'] and v['after']['nll']<=v['before']['nll']+.2 for v in report['arms'].values())
        atomic_json(out/'report.json',report)
        if not report['bounded_check_passed']:raise RuntimeError('Comparison requires review')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key,default in [('donor','/donor'),('converted','/converted'),('teacher','/teacher'),('data','/data'),('output','/output'),('config','/app/configs/arcus3/backbone_adaptation_alpha321.json')]:
        p.add_argument('--'+key,default=default)
    p.add_argument('--deadline',required=True)
    main(p.parse_args())
