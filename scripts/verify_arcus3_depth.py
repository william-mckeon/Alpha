"""Production parity: same trained checkpoint, full-capacity depth on versus off."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arcus3.config import authorize,read,deadline,check_live
from arcus3.donor import verify,load
from arcus3.checkpoint import digest
from scripts.verify_arcus3_conversion import measure,compare
from baby_arcus.gpu_job_control import gpu_job

def main(a):
    if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1' or not Path('/.dockerenv').exists():raise RuntimeError('Docker CUDA required')
    authorize(read('/app/configs/arcus3/project.json'),'inference');end=deadline(a.deadline);out=Path(a.output);verify(a.donor)
    import torch
    torch.set_num_threads(2)
    with gpu_job():
        torch.cuda.set_per_process_memory_fraction(.7)
        m,t=load(a.donor,converted=a.converted,expanded=a.expanded)
        gates=[(module,module.depth_gate) for module in m.modules() if hasattr(module,'depth_gate')]
        if len(gates)!=6:raise ValueError('Expected Phase 7 depth gates')
        check_live(end,out);enabled=measure(m,t)
        for module,gate in gates:del module.depth_gate
        check_live(end,out);disabled=measure(m,t)
        for module,gate in gates:module.depth_gate=gate
        result={'complete':True,'checkpoint_manifest_sha256':digest(Path(a.expanded)/'manifest.json'),'capacity':1.0,'parity':compare(disabled,enabled)}
        (out/'depth-parity.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--donor',default='/donor');p.add_argument('--converted',required=True);p.add_argument('--expanded',required=True);p.add_argument('--output',default='/output');p.add_argument('--deadline',required=True);p.add_argument('--max-new-tokens',type=int,default=128);main(p.parse_args())
