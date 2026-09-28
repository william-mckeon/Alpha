import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.foundation_data import audit

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('corpus');p.add_argument('--sources',default='configs/baby_arcus/arcus_128m_smollm2_sources.json');p.add_argument('--output',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.sources).read_text());report=audit(a.corpus,[s['name'] for s in cfg['sources']])
    Path(a.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
