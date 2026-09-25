"""Stream full local review shards and the cached HF source into 64K SFT windows."""
import json
import re
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.contracts import digest
from baby_arcus.data_manifest import file_digest
from baby_arcus.sft_dataset import windows
from baby_arcus.identity_normalization import normalize
from baby_arcus.local_agent_dataset import IDENTITY
from baby_arcus.codebase_corpus import SECRET
from baby_arcus.dataset_split_audit import split_for,content_hash


def main():
    import pyarrow.parquet as pq
    from arcus.tokenizer import get_tokenizer
    root=Path('/dataset');tokenizer=get_tokenizer('o200k_base')
    if not json.loads((root/'report.json').read_text()).get('complete'):raise ValueError('Source build not complete')
    output=root/'packed64';output.mkdir(exist_ok=False)
    report={'context_tokens':65536,'approved':False,'training_enabled':False,'counts':{},'rejections':{},'complete':False}
    counts=Counter();rejects=Counter();seen={};files={}
    records=(output/'records.jsonl').open('w');provenance=(output/'identity-edits.jsonl').open('w')
    for split in ('training','validation','test'):files[split]=(output/(split+'.windows.jsonl')).open('w')
    def admit(record,source):
        counts[source+'_examined']+=1
        try:
            if SECRET.search(json.dumps(record)):raise ValueError('suspected_credential')
            identity=content_hash('\n'.join(m['content'] for m in record['messages']))
            if identity in seen:rejects[source+':duplicate']+=1;return
            # Full preflight first; never partially admit an invalid trajectory.
            n=0;targets=0
            for w in windows(record,tokenizer,65536):n+=1;targets+=sum(w['mask'][1:])
            if not n:raise ValueError('no_targets')
            seen[identity]=record['split']
            for w in windows(record,tokenizer,65536):
                files[record['split']].write(json.dumps({'record_sha256':digest(record),'group':record['group'],**w},separators=(',',':'))+'\n')
            records.write(json.dumps(record)+'\n');counts[source+'_accepted']+=1
            counts[record['split']+'_windows']+=n;counts[record['split']+'_target_tokens']+=targets
        except (ValueError,KeyError,TypeError) as e:rejects[source+':'+str(e)]+=1
    def update():
        report['counts']=dict(counts);report['rejections']=dict(rejects)
        (output/'report.json').write_text(json.dumps(report,indent=2))
    try:
        for line in (root/'logs.jsonl').open():
            admit(json.loads(line),'local_logs')
            if counts['local_logs_examined']%100==0:update()
        for split in ('training','validation'):
            for line in Path('/prior',f'react-{split}.jsonl').open():admit(json.loads(line),'react')
        source=Path('/hf/swegym.parquet');receipt=json.loads(Path(str(source)+'.receipt.json').read_text())
        if file_digest(source)!=receipt['sha256']:raise ValueError('HF source hash mismatch')
        group='hf:'+receipt['repo'];report['hf_source']=receipt
        for batch in pq.ParquetFile(source).iter_batches(batch_size=1):
            for row in batch.to_pylist():
                messages=[{'role':'system','content':IDENTITY}];changes=[]
                for message in row['messages']:
                    if message['role']=='system':continue
                    message=dict(message)
                    if message['role']=='assistant':
                        # Foreign action syntax is retained only as historical context.
                        action=bool(re.search(r'<(?:execute_bash|execute_ipython|file_edit|tool_call)\b',message['content']))
                        message['train']=not action
                        if not action:message['content'],edits=normalize(message['content']);changes+=edits
                    messages.append(message)
                record={'version':1,'source':group,'group':group,'split':split_for(group),'messages':messages,
                    'provenance':{'adapter':'hf-phase2b-v1','input_sha256':digest(row),'target_kinds':['text'],
                    'terminal_result_observed':False,'dataset':{'repo':receipt['repo'],'revision':receipt['revision'],
                    'license':'MIT','outcome':'source_reports_success','identity_changes':changes}}}
                admit(record,'hf');update()
        report['complete']=True
    finally:
        records.close();provenance.close()
        for f in files.values():f.close()
        report['files']=[{'path':p.name,'bytes':p.stat().st_size,'sha256':file_digest(p)} for p in output.glob('*.jsonl')]
        report['limitations']=['External tool calls are context only; original ReAct lessons supply executable tool targets.',
            'Only the locally pinned SWE-Gym HF source is included; other HF sources have not been acquired.',
            'Exact duplicates are removed; semantic near-duplicate and benchmark-contamination review remains required.',
            'Packed windows are review artifacts, not an approved trainer release.']
        update()
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
