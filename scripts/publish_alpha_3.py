"""Upload an explicitly selected inference manifest, then verify private immutable revision."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'

def staged_weights(api,repo,package,names,expected):
    """One durable shard commit at a time; metadata/manifest remain the final commit."""
    info=api.model_info(repo,files_metadata=True)
    if not info.private:raise ValueError('Repository must remain private')
    remote={f.rfilename:f for f in info.siblings}
    for name in names:
        if not name.endswith('.safetensors'):continue
        found=remote.get(name)
        if found is not None and found.lfs and found.lfs.sha256==expected[name]['sha256']:
            print(json.dumps({'stage':'shard_already_verified','file':name}),flush=True);continue
        if not api.model_info(repo).private:raise ValueError('Repository must remain private')
        print(json.dumps({'stage':'uploading_shard','file':name}),flush=True)
        commit=api.upload_file(repo_id=repo,path_or_fileobj=str(package/name),path_in_repo=name,
                               commit_message='Stage Alpha 3.0 weight shard (release incomplete)')
        current=api.model_info(repo,revision=commit.oid,files_metadata=True)
        match=next(f for f in current.siblings if f.rfilename==name)
        if not current.private or not match.lfs or match.lfs.sha256!=expected[name]['sha256']:
            raise ValueError('Staged shard verification failed')
        print(json.dumps({'stage':'shard_verified','file':name,'revision':commit.oid}),flush=True)

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main(package):
    p=Path(package);m=json.loads((p/'manifest.json').read_text());s=m['release']
    if s['repo_id']!='Islanderintel/Alpha-3.0' or s['private'] is not True or s['authorization']!='user-requested-alpha-3-publication':raise ValueError('Release scope mismatch')
    v=json.loads((p/'verification.json').read_text())
    if not v['complete'] or not all(r['exact'] for r in v['parity'].values()):raise ValueError('Unverified package')
    for n,item in m['files'].items():
        if Path(n).name!=n or digest(p/n)!=item['sha256']:raise ValueError('Package tamper')
    if s.get('training_updates')!=0 or s.get('unique_parameters')!=2013390848 or v['release']!=s:
        raise ValueError('Release identity mismatch')
    from dotenv import load_dotenv
    load_dotenv('.env',override=True)
    from huggingface_hub import HfApi,hf_hub_download
    from huggingface_hub.errors import RepositoryNotFoundError
    api=HfApi();repo=s['repo_id']
    try:info=api.model_info(repo)
    except RepositoryNotFoundError:
        api.create_repo(repo,private=True,repo_type='model');info=api.model_info(repo)
    if not info.private:raise ValueError('Repository must be private before upload')
    names=sorted(m['files'])+['manifest.json']
    staged_weights(api,repo,p,names,m['files'])
    metadata=[n for n in names if not n.endswith('.safetensors')]
    commit=api.upload_folder(repo_id=repo,folder_path=str(p),allow_patterns=metadata,num_threads=1,commit_message='Complete verified Alpha 3.0 expert initialization')
    revision=commit.oid;info=api.model_info(repo,revision=revision,files_metadata=True)
    if not info.private:raise ValueError('Privacy verification failed')
    siblings={f.rfilename:f for f in info.siblings}
    verified={}
    for n in names:
        expected=digest(p/n);f=siblings[n]
        if f.lfs:
            actual=f.lfs.sha256
        else:
            actual=digest(hf_hub_download(repo,n,revision=revision))
        if actual!=expected:raise ValueError('Remote hash mismatch: '+n)
        verified[n]=actual
    receipt={'repo_id':repo,'revision':revision,'private':True,'files':verified,'verification':'LFS server SHA256 for large objects; downloaded immutable bytes for Git objects'}
    target=p.parent/('publication-'+revision+'.json');target.write_text(json.dumps(receipt,indent=2))
    print(json.dumps({'verified':True,'revision':revision,'private':True,'receipt':str(target)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);args=p.parse_args()
    try:main(args.package)
    except Exception as exc:
        chain=[];seen=set();error=exc
        while error is not None and id(error) not in seen:
            seen.add(id(error));response=getattr(error,'response',None)
            chain.append({'type':type(error).__name__,'http_status':getattr(response,'status_code',None)})
            error=error.__cause__ or error.__context__
        # Never emit signed URLs, authorization headers or credential-bearing tracebacks.
        result={'verified':False,'error_chain':chain}
        (Path(args.package).parent/'upload-error.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True);raise SystemExit(1)
