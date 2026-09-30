"""Sequential frozen donor teacher generation in controlled Docker CUDA."""
import os,json,argparse,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read
from arcus3.checkpoint import digest
from arcus3.donor import load,verify
from arcus3.distillation import targets
from baby_arcus.language_stream import atomic_json
from baby_arcus.gpu_job_control import gpu_job

def main(a):
    if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1' or not Path('/.dockerenv').exists():raise RuntimeError('Docker CUDA only')
    import torch
    from safetensors.torch import save_file
    from arcus3.corpus_stream import CorpusStream
    from arcus3.config import deadline,check_live
    if getattr(a,'production',False):
        from datetime import datetime,timezone
        end=datetime.fromisoformat(a.deadline.replace('Z','+00:00'))
        if end.tzinfo is None or not 0<(end-datetime.now(timezone.utc)).total_seconds()<=86400:raise ValueError('Invalid worker lease')
    else:end=deadline(a.deadline)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():raise ValueError('Immutable teacher cache exists')
    verify(a.donor);files={};reuse={}
    cache_identity={'donor':digest(Path(a.donor)/'manifest.json'),'data':digest(Path(a.data)/'manifest.json'),'top_k':a.top_k}
    if (root/'identity.json').exists() and read(root/'identity.json')!=cache_identity:raise ValueError('Partial cache identity changed')
    atomic_json(root/'identity.json',cache_identity)
    journal=root/'completed.jsonl'
    if journal.exists():
        for line in journal.read_text().splitlines():
            item=json.loads(line);reuse[item['name']]=(root/item['name'],item['sha256'])
    from arcus3.config import safe_child
    for cache in a.reuse:
        metadata=read(Path(cache)/'manifest.json')
        if metadata['donor_manifest_sha256']!=digest(Path(a.donor)/'manifest.json') or metadata['top_k']!=a.top_k:
            raise ValueError('Reuse teacher identity mismatch')
        for name,sha in metadata['files'].items():reuse[name]=(safe_child(cache,name),sha)
    with gpu_job():
        torch.cuda.set_per_process_memory_fraction(.7);torch.set_num_threads(2)
        model,tok=load(a.donor);start=time.monotonic()
        if a.qualification:
            from arcus3.data import encode
            from scripts.prepare_arcus3_phase8_data import seal
            legacy=read(Path(a.data)/'manifest.json')
            for name,sha in legacy['files'].items():
                from arcus3.config import safe_child
                if digest(safe_child(a.data,name))!=sha:raise ValueError('Qualification source changed')
            prepared=root/'data';prepared.mkdir(exist_ok=False);rows=[]
            for line in (Path(a.data)/'train.jsonl').open():
                row=encode(tok,json.loads(line)['messages'],128)
                if row is not None:rows.append(row)
                if len(rows)==a.max_records:break
            (prepared/'train-000.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            atomic_json(prepared/'provenance.json',{'tokenizer_sha256':digest(Path(a.donor)/'files/tokenizer.json'),
                'sources_sha256':digest(Path(a.data)/'manifest.json'),'evaluation_exclusions_sha256':digest(Path(a.data)/'test.jsonl'),
                'reviewed':True,'scope':'qualification-subset-not-combined-corpus'})
            seal(prepared,prepared/'provenance.json',True);a.data=str(prepared)
        stream=CorpusStream(a.data)
        from arcus3.tokenizer_contract import validate_data
        validate_data(a.donor,stream.manifest)
        for _ in range(min(a.max_records,stream.manifest['records'])):
            check_live(end,root);row=stream.next();key=row['sha256']
            name=key+'.safetensors'
            if name in files:continue
            if name in reuse:
                import shutil
                from safetensors import safe_open
                previous,expected=reuse[name]
                if digest(previous)!=expected:raise ValueError('Reused teacher file changed')
                with safe_open(str(previous),framework='pt',device='cpu') as f:
                    if f.get_slice('indices').get_shape()!=[len(row['input_ids'])-1,a.top_k]:raise ValueError('Reused target shape mismatch')
                if previous.resolve()!=(root/name).resolve():shutil.copyfile(previous,root/name)
                files[name]=expected
            else:
                with torch.no_grad():
                    logits=model(torch.tensor([row['input_ids']],device='cuda'),use_cache=False).logits[0,:-1]
                    target=targets(logits,a.top_k)
                temporary=root/(name+'.tmp');save_file(target,str(temporary));temporary.replace(root/name)
                files[name]=digest(root/name)
                del logits,target
            with journal.open('a') as log:
                log.write(json.dumps({'name':name,'sha256':files[name]})+'\n');log.flush();os.fsync(log.fileno())
            if len(files)%100==0:
                atomic_json(root/'progress.json',{'completed_records':len(files),'requested_records':min(a.max_records,stream.manifest['records'])})
        atomic_json(root/'manifest.json',{'schema':'arcus3-teacher-v1','donor_manifest_sha256':digest(Path(a.donor)/'manifest.json'),
            'data_sha256':stream.sha,'top_k':a.top_k,'precision':'bf16','approximation':'top-k plus residual probability bucket',
            'teacher_input_tokens':sum(len(json.loads(l)['input_ids']) for item in stream.manifest['shards'] for l in (Path(a.data)/item['path']).open()) if a.max_records>=stream.manifest['records'] else None,
            'seconds':time.monotonic()-start,'files':files})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--donor',default='/donor');p.add_argument('--data',default='/data');p.add_argument('--output',default='/output');p.add_argument('--deadline',required=True);p.add_argument('--max-records',type=int,default=32);p.add_argument('--top-k',type=int,default=32);p.add_argument('--reuse',action='append',default=[]);p.add_argument('--qualification',action='store_true');p.add_argument('--production',action='store_true');p.add_argument('--max-new-tokens',type=int,default=128);main(p.parse_args())
