"""Compile measured gates; publication remains a separate explicit operation."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.shared_qualification import compile_report
from baby_arcus.language_stream import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    from baby_arcus.shared_qualification import REQUIRED
    for name in sorted(REQUIRED):p.add_argument('--'+name,required=True)
    p.add_argument('--continuity')
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());root=Path(cfg['root'])
    manifest=json.loads((root/'candidate.json').read_text())
    paths={name:getattr(a,name) for name in REQUIRED}
    if a.continuity:paths['continuity']=a.continuity
    report=compile_report(root,manifest,paths)
    atomic_json(root/'reviewed-qualification.json',report);print(json.dumps(report))
    if not report['qualified']:raise SystemExit(1)

if __name__=='__main__':main()
