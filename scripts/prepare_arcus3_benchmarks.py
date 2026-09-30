"""CPU-only snapshot of the pinned donor suite, including training exclusions."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read,safe_child
from arcus3.checkpoint import digest
from arcus3.production import identity
from baby_arcus.language_stream import atomic_json

def prepare(root):
    from lighteval.tasks.registry import Registry,taskinfo_selector
    from lighteval.utils.utils import download_dataset_worker
    from huggingface_hub import HfApi
    import vendor.smollm2.tasks as tasks
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():raise ValueError('Sealed benchmark cache exists')
    registry=Registry(cache_dir=str(root/'download'),custom_tasks=tasks)
    names,_=taskinfo_selector('vendor/smollm2/smollm2_instruct.txt',registry)
    configs=registry.get_task_dict(names);items={};exclusions=set()
    def strings(value):
        if isinstance(value,str) and len(value)>=24:exclusions.add(value)
        elif isinstance(value,dict):
            for v in value.values():strings(v)
        elif isinstance(value,(list,tuple)):
            for v in value:strings(v)
    for name,task in configs.items():
        key=identity([task.dataset_path,task.dataset_config_name]);folder=root/key
        revision=task.dataset_revision or HfApi().dataset_info(task.dataset_path).sha
        if task.dataset_path=='Rowan/hellaswag':
            # The donor-era builder points to deleted GitHub master JSONL URLs.
            # Use the author's immutable Parquet conversion; preserve this deviation.
            from datasets import load_dataset
            revision='218ec52e09a7e7462a5400043bb9a69a41d06b76'
            dataset=load_dataset(task.dataset_path,'default',revision=revision)
        else:
            dataset=download_dataset_worker(task.dataset_path,task.dataset_config_name,task.trust_dataset,
                                            task.dataset_filter,revision)
        if not folder.exists():dataset.save_to_disk(str(folder))
        for split in dataset.values():
            for row in split:strings(row)
        items[key]={'repo':task.dataset_path,'subset':task.dataset_config_name,'revision':revision,
                    'original_revision':task.dataset_revision,'directory':key}
        atomic_json(root/'progress.json',{'task':name,'prepared':len(items),'total_tasks':len(configs)})
    atomic_json(root/'exclusions.json',sorted(exclusions))
    files={str(p.relative_to(root)).replace('\\','/'):digest(p) for item in items.values() for p in (root/item['directory']).rglob('*') if p.is_file()}
    files['exclusions.json']=digest(root/'exclusions.json')
    result={'schema':'arcus3-donor-benchmark-cache-v1','protocol_sha256':digest('vendor/smollm2/manifest.json'),
            'tasks':names,'datasets':items,'files':files}
    atomic_json(root/'manifest.json',result);return result

def install(root):
    """Replace network acquisition only. Prompting, splits and scoring stay upstream."""
    from datasets import load_from_disk
    import lighteval.tasks.lighteval_task as module
    root=Path(root);manifest=read(root/'manifest.json')
    if manifest['protocol_sha256']!=digest('/app/vendor/smollm2/manifest.json'):raise ValueError('Benchmark protocol changed')
    for name,h in manifest['files'].items():
        if digest(safe_child(root,name))!=h:raise ValueError('Benchmark snapshot changed')
    def download(repo,subset,trust,filter_fn,revision,*args,**kwargs):
        item=manifest['datasets'][identity([repo,subset])]
        if revision!=item['original_revision']:raise ValueError('Benchmark revision changed')
        return load_from_disk(str(safe_child(root,item['directory'])))
    module.download_dataset_worker=download
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    result=prepare(a.output);print(json.dumps({'tasks':len(result['tasks']),'datasets':len(result['datasets'])}))
