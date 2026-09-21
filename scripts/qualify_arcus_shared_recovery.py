"""Full-size optimizer/RNG interruption replay on isolated checkpoint copies."""
import argparse,gc,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import load,save,restore_optimizer
from baby_arcus.shared_learning import update
from baby_arcus.shared_curriculum import example
from baby_arcus.language_stream import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--report',default='recovery.json');a=p.parse_args()
    if Path(a.report).name!=a.report or not a.report.endswith('.json'):raise ValueError('Report must be a JSON filename')
    cfg=json.loads(Path(a.config).read_text());root=Path(cfg['root']);manifest=json.loads((root/'candidate.json').read_text())
    torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu'
    from arcus.tokenizer import get_tokenizer
    tokenizer=get_tokenizer(cfg['encoding']);row,target,_=example(37,paired=True)
    model,data=load(root,manifest,device);model.requires_grad_(True)
    if model.version>=8:
        from scripts.initialize_arcus_shared_temporal import examples
        row,target,outcome,_=examples(9821800,1,'training')[0]
        target['prediction']=outcome['after']['internal']
    rows,targets=[row],[target]
    if model.version>=9:
        from baby_arcus.shared_causal_curriculum import transition
        samples=[transition(index,'training') for index in (0,15)]
        rows=[sample[0] for sample in samples];targets=[sample[1] for sample in samples]
    optimizer=restore_optimizer(model,data,cfg['learning_rate'])
    work=root/'recovery-work';before=save(work,model,optimizer,data['progress'])
    first=update(model,optimizer,rows,tokenizer,targets)
    expected={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    del model,optimizer,data;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    model,data=load(work,before,device);optimizer=restore_optimizer(model,data,cfg['learning_rate'])
    torch.set_rng_state(data['rng'])
    if device=='cuda':torch.cuda.set_rng_state_all(data['cuda_rng'])
    second=update(model,optimizer,rows,tokenizer,targets)
    mismatches=[name for name,value in model.state_dict().items() if not torch.equal(expected[name],value.cpu())]
    report={'candidate':manifest,'device':device,'recovery':first==second and not mismatches,'losses':[first,second],
        'mismatched_tensors':mismatches,'isolated_pre_update_checkpoint':before,'desktop_changed':False,'candidate_pointer_changed':False,
        'memory_and_sequence_exercised':model.version>=9}
    from baby_arcus.shared_qualification import source_snapshot
    report['runtime_sources']=source_snapshot()
    atomic_json(root/a.report,report);print(json.dumps(report))
    if not report['recovery']:raise SystemExit(1)

if __name__=='__main__':main()
