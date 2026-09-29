"""Read-only remote metadata inventory, no corpus files or model execution."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import read
from baby_arcus.language_stream import atomic_json

def main(output):
    from huggingface_hub import HfApi
    api=HfApi();rows=[]
    for source in read('configs/arcus3/phase8_sources.json')['sources']:
        row=dict(source)
        if 'repo_id' in row:
            try:
                info=api.dataset_info(row['repo_id'],revision=row.get('revision'),files_metadata=True,timeout=45)
                sizes=[f.size for f in info.siblings]
                row.update(resolved_revision=info.sha,repository_bytes=sum(x for x in sizes if x is not None),
                    files=len(sizes),missing_file_sizes=sum(x is None for x in sizes),gated=info.gated,
                    license=(info.card_data or {}).get('license'),size_scope='entire repository superset; may contain overlapping subsets')
            except Exception as e:row.update(error_type=type(e).__name__,size_verified=False)
        else:
            p=Path(row['local']);row['bytes']=p.stat().st_size
        rows.append(row);print(json.dumps(row),flush=True)
    atomic_json(output,{'schema':'arcus3-source-storage-v1','complete_donor_reproduction':False,'sources':rows})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
