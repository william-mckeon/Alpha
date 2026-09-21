"""Calibrate new delayed-outcome heads without changing established behavior weights."""
import argparse,json,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_model import SharedModel
from baby_arcus.shared_checkpoint import load,save,restore_optimizer
from baby_arcus.shared_experience import capture
from baby_arcus.shared_temporal import targets,body_vector
from baby_arcus.language_stream import atomic_json
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.body_dynamics import pose
from baby_arcus.contracts import ContractError

def examples(seed,count,split):
    rng=random.Random(seed);rows=[]
    for index in range(count):
        app=PlayroomApplication()
        try:
            app.world.body.motor_mode='independent'
            app.world.body.joint_positions=pose(rng.random());app.world.body.previous_joints=dict(app.world.body.joint_positions)
            for _ in range(12):app.advance()
            before=capture(app);before['objects']=[];before['lesson_provenance']={'split':split,'family':'delayed_outcomes','seed':seed+index}
            before['eligibility']['training']=split=='training'
            action=({'kind':'joint','joint':'missing_joint','delta':.15} if index%2 else
                {'kind':'gaze','yaw':rng.choice((-.5,0,.5)),'pitch':0} if index%4==0 else
                {'kind':'joint','joint':'front_left.hip','delta':rng.choice((-.15,.15))})
            try:
                status,result=app('POST','/v1/action',{'request_id':str(index),'source':'human','action':action});executed=status==200
            except ContractError as exc:result={'error':str(exc)};executed=False
            outcome={'experience_id':before['id'],'action':action,'executed':executed,'result':result,'after':capture(app)}
            for _ in range(3):app.advance()
            after=capture(app);after['lesson_provenance']=dict(before['lesson_provenance'])
            labels=targets(before,outcome,after);before['executed_action']=action
            rows.append((before,labels,outcome,after))
        finally:app.close()
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text());root=Path(args.output);root.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(2);torch.manual_seed(98118)
    source=json.loads((Path(cfg['root'])/'candidate.json').read_text());device='cuda' if torch.cuda.is_available() else 'cpu'
    original,data=load(cfg['root'],source,device)
    model=SharedModel(original.body,original.language,version=8).to(device)
    missing,unexpected=model.load_state_dict(original.state_dict(),strict=False)
    if unexpected or set(missing)!={head+'.'+part for head in ('future_body','future_rgb','action_quality') for part in ('weight','bias')}:
        raise ValueError('Unexpected temporal migration')
    # Restore the existing optimizer, then add only the new parameter group.
    optimizer=restore_optimizer(model,data,cfg['learning_rate'])
    new_params=[p for name,p in model.named_parameters() if name.startswith(('future_body.','future_rgb.','action_quality.'))]
    optimizer.add_param_group({'params':new_params,'lr':.001})
    original_weights={name:value.detach().cpu().clone() for name,value in original.state_dict().items()}
    del original
    from arcus.tokenizer import get_tokenizer
    tokenizer=get_tokenizer(cfg['encoding']);model.eval().requires_grad_(False)
    training=examples(9811800,96,'training');held=examples(9911800,48,'confirmation')
    def encode(items):
        with torch.no_grad():hidden=torch.cat([model([row],tokenizer,requested=('hidden',))['hidden'] for row,_,_,_ in items])
        labels={key:torch.tensor([target[key] for _,target,_,_ in items],device=device,dtype=torch.long if key=='action_quality' else torch.float32)
            for key in ('future_body','future_rgb','action_quality')}
        return hidden,labels
    train_x,train_y=encode(training);test_x,test_y=encode(held)
    def measure():
        with torch.no_grad():return {'body_mse':float(torch.nn.functional.mse_loss(model.future_body(test_x),test_y['future_body'])),
            'rgb_mse':float(torch.nn.functional.mse_loss(model.future_rgb(test_x),test_y['future_rgb'])),
            'accepted_rejected_accuracy':float((model.action_quality(test_x).argmax(-1)==test_y['action_quality']).float().mean())}
    before=measure()
    for parameter in new_params:parameter.requires_grad_(True)
    for step in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss=sum(torch.nn.functional.mse_loss(getattr(model,key)(train_x),train_y[key]) for key in ('future_body','future_rgb'))
        loss=loss+torch.nn.functional.cross_entropy(model.action_quality(train_x),train_y['action_quality'])
        loss.backward();optimizer.step()
    after=measure()
    mismatches=[key for key,value in original_weights.items() if not torch.equal(value,model.state_dict()[key].cpu())]
    if mismatches:raise ValueError('Temporal calibration changed established behavior weights')
    model.requires_grad_(True);progress=data['progress'];progress['updates']+=300
    progress['receipts'].append({'curriculum':'delayed-head-calibration','updates':300,'training_examples':96,'held_out_examples':48,
        'existing_weights_unchanged':True,'training_seed':9811800,'evaluation_seed':9911800})
    manifest=save(root,model,optimizer,progress);atomic_json(root/'candidate.json',manifest)
    atomic_json(root/'config.json',dict(cfg,root=root.as_posix(),delayed_outcomes=True))
    from baby_arcus.shared_replay import Replay
    replay=Replay(root/'temporal-calibration.sqlite3')
    try:
        for row,_,outcome,later in training+held:replay.add_delayed(row,outcome,later)
    finally:replay.close()
    baseline=sum(sum((a-b)**2 for a,b in zip(body_vector(row),target['future_body']))/20 for row,target,_,_ in held)/len(held)
    report={'candidate':manifest,'source':source,'before':before,'after':after,'unchanged_body_baseline_mse':baseline,
        'training_examples':96,'held_out_examples':48,'existing_weights_unchanged':not mismatches,
        'scope':'Head calibration and delayed/rejected replay wiring. Does not establish causal prediction better than persistence or autonomous curiosity.'}
    from baby_arcus.shared_qualification import source_snapshot
    report['runtime_sources']=source_snapshot();atomic_json(root/'temporal-calibration.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
