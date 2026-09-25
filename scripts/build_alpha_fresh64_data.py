"""Build local code/log review shards and freeze full existing corpus references."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.codebase_corpus import collect,EXCLUDED,SUFFIXES
from baby_arcus.local_agent_dataset import extract,episode
from baby_arcus.dataset_split_audit import content_hash


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def build(config,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    report={'approved':False,'training_enabled':False,'sources':[], 'counts':{},'rejections':{},'identity_edits':0}
    counts=Counter();rejects=Counter();seen={}
    streams={name:(output/(name+'.jsonl')).open('w',encoding='utf-8') for name in ('code','logs','provenance')}
    def emit(kind,value):
        text=value.get('text') or json.dumps(value.get('messages'),sort_keys=True)
        key=content_hash(text)
        if key in seen:
            rejects['duplicate_'+kind]+=1;return
        seen[key]=value['split'];streams[kind].write(json.dumps(value,ensure_ascii=False)+'\n');counts[kind+'_'+value['split']]+=1
    def progress():
        report['counts']=dict(counts);report['rejections']=dict(rejects)
        (output/'report.json').write_text(json.dumps(report,indent=2))
    try:
        for project in config['projects']:
            root=Path(project['root']);paths=[]
            if not root.is_dir():rejects['missing_project']+=1;continue
            for folder,dirs,files in os.walk(root,followlinks=False):
                dirs[:]=[d for d in dirs if d.lower() not in EXCLUDED|{'data','output','benchmarks','calc-bench','evaluation-results'}
                         and not d.startswith('.') and not Path(folder,d).is_symlink()]
                paths.extend(str(Path(folder,n).relative_to(root)) for n in files if Path(n).suffix.lower() in SUFFIXES)
            admitted=0
            for record in collect(root,paths,project['group'],max_bytes=1024**2) if paths else ():
                emit('code',record);admitted+=1
            report['sources'].append({'kind':'code','root':str(root),'group':project['group'],'eligible_paths':len(paths),'admitted_before_dedup':admitted})
            progress()
        for source in config['logs']:
            for path in sorted(Path(source['root']).rglob('*.jsonl')):
                before=path.stat();cwd=None;current=[];accepted=[];edits=0;local_rejects=Counter()
                def flush():
                    nonlocal current,edits
                    if current:
                        try:
                            record,changes=episode(current,'local:'+source['provider'],group)
                            accepted.append(record);edits+=len(changes)
                        except (ValueError,KeyError,TypeError) as e:local_rejects[str(e)]+=1
                    current=[]
                group=None
                with path.open(encoding='utf-8') as stream:
                    consumed=0
                    for line in stream:
                        consumed+=len(line.encode('utf-8'))
                        if consumed>before.st_size:break
                        try:raw=json.loads(line)
                        except ValueError:local_rejects['invalid_json']+=1;continue
                        meta=raw.get('payload',{}) if raw.get('type') in ('session_meta','turn_context') else raw
                        cwd=cwd or meta.get('cwd')
                        if group is None and cwd:
                            canonical=cwd.replace('\\','/').lower().rstrip('/')
                            project=next((p for p in config['projects'] if canonical==p['root'].replace('\\','/').lower().rstrip('/') or canonical.startswith(p['root'].replace('\\','/').lower().rstrip('/')+'/')),None)
                            if project:group=project['group']
                            else:group='external-project:'+hashlib.sha256(canonical.encode()).hexdigest()[:20]
                        if group is None:continue
                        for message in extract(raw,source['provider']):
                            if message['role']=='user':flush()
                            current.append(message)
                flush()
                after=path.stat()
                if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                    rejects['active_log_changed']+=1;continue
                if not group:rejects['log_without_project']+=1;continue
                fingerprint=sha(path)
                final=path.stat()
                if (after.st_size,after.st_mtime_ns)!=(final.st_size,final.st_mtime_ns):rejects['active_log_changed']+=1;continue
                for record in accepted:emit('logs',record)
                rejects.update(local_rejects);report['identity_edits']+=edits
                streams['provenance'].write(json.dumps({'path':str(path),'sha256':fingerprint,'bytes':before.st_size,'group':group,'episodes':len(accepted)})+'\n')
                counts['log_files_scanned']+=1
                if counts['log_files_scanned']%50==0:progress()
        # Pin complete shards, not the earlier million-token coding pilot.
        corpus=[]
        for family,patterns in config['corpus_patterns'].items():
            for pattern in patterns:
                for path in sorted(Path(config['dataset_root']).glob(pattern)):
                    before=path.stat();fingerprint=sha(path);after=path.stat()
                    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Corpus changed while hashing')
                    corpus.append({'family':family,'path':str(path.resolve()),'bytes':before.st_size,'sha256':fingerprint})
                    progress()
        report['full_corpus_shards']=corpus
        report['complete']=True
    finally:
        for stream in streams.values():stream.close()
        progress()
    (output/'source-config.json').write_text(json.dumps(config,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=build(json.loads(Path(a.config).read_text()),a.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','full_corpus_shards')}))
