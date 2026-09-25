"""Create random 64K-capacity Alpha weights in an isolated CUDA diagnostic."""
import json
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/baby_arcus/alpha_fresh64.json')
    args = parser.parse_args()
    import torch
    from arcus.tokenizer import get_tokenizer
    from baby_arcus.shared_factory import create,source_manifest
    from baby_arcus.shared_checkpoint import save,digest
    from baby_arcus.runtime_contract import require_gpu
    from baby_arcus.gpu_job_control import gpu_job
    require_gpu();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.75)
    cfg=json.loads(Path(args.config).read_text())
    root=Path('/evidence/fresh-model');root.mkdir(exist_ok=False)
    with gpu_job():
        model=create(cfg,get_tokenizer(cfg['encoding']).vocab_size,device='cuda')
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
        progress={'initialization':'random','sources':{},'updates':0,'trained_tokens':0,'receipts':[],
                  'seed':cfg['seed'],'source_manifest':source_manifest(),'parameters':sum(p.numel() for p in model.parameters())}
        candidate=save(root,model,optimizer,progress)
        for name,value in [('candidate.json',candidate),('initial.json',candidate),('experiment.json',cfg)]:
            (root/name).write_text(json.dumps(value,indent=2))
        (root/'pause-training').touch()
        report={'candidate':candidate,'parameters':progress['parameters'],'context_tokens':cfg['context_tokens'],
                'random_initialization':True,'updates':0,'training_enabled':False,
                'hash_verified':digest(root/(candidate['generation']+'.pt'))==candidate['sha256'],
                'full_64k_training_qualified':False}
        Path('/evidence/initialization.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
