"""Action-conditioned sensory features and bounded experiment candidates."""
from baby_arcus.body_dynamics import JOINTS
from baby_arcus.shared_memory import sensory_features


def action_features(row,action=None,horizon=None):
    action=action or row.get('executed_action') or {};values=[0.0]*19
    kind=action.get('kind');gaze=row.get('gaze',[0,0,0,0])
    if kind=='sequence':
        from copy import deepcopy
        working=deepcopy(row)
        if not 1<=len(action.get('actions',[]))<=8:raise ValueError('Invalid planned sequence')
        for item in action['actions']:
            encoded=action_features(working,item,1)
            values[:18]=[a+b for a,b in zip(values[:18],encoded[:18])]
            if item.get('kind') in ('head','gaze'):
                part=0 if item['kind']=='head' else 2
                working['gaze'][part:part+2]=[item['yaw'],item['pitch']]
    elif kind=='joint' and action.get('joint') in JOINTS:values[JOINTS.index(action['joint'])]=action['delta']
    elif kind in ('gaze','head'):
        offset=12 if kind=='gaze' else 14;part=2 if kind=='gaze' else 0
        values[offset]=action['yaw']-gaze[part];values[offset+1]=action['pitch']-gaze[part+1]
    elif kind=='eyelids':values[16]=action['openness']-row['senses']['eyelid_openness']
    elif kind:values[17]=1.0
    ticks=row.get('prediction_horizon',3) if horizon is None else horizon
    if type(ticks) is not int or not 1<=ticks<=90:raise ValueError('Prediction horizon must be 1–90 ticks')
    values[18]=ticks/90
    return values


def causal_features(row):
    physical=sensory_features(row);action=action_features(row);memories=row.get('memory',[])[-8:]
    desired=[physical[20+i]+action[14+i] for i in range(2)]+[physical[22+i]+action[12+i] for i in range(2)]
    nearest=min(memories,key=lambda m:sum((a-b)**2 for a,b in zip(m['features'][20:24],desired))) if memories else None
    remembered=nearest['features']+[1.0] if nearest else [0.0]*73
    return physical+action+remembered


def experiments(row):
    if row['senses']['sleep_state']!='awake' or row['senses']['held']:return []
    if row['senses']['eyelid_openness']<=0:return [{'kind':'eyelids','openness':1.0}]
    yaw,pitch=row.get('gaze',[0,0,0,0])[2:]
    return [{'kind':'gaze','yaw':x,'pitch':y} for x,y in ((-.75,0),(.75,0),(0,-.75),(0,.75),(0,0)) if abs(x-yaw)+abs(y-pitch)>.1]


def choose_experiment(model,tokenizer,row):
    """Use learned forecasts to seek unseen sensory outcomes within a small budget."""
    from copy import deepcopy
    import torch
    from baby_arcus.shared_depth import verify_depth
    verify_depth(model)
    actions=experiments(row)
    if not actions:return None,[]
    seen=[sensory_features(row)[24:72]]+[m['features'][24:72] for m in row.get('memory',[])[-8:]]
    scored=[]
    with torch.no_grad():
        for action in actions:
            conditioned=deepcopy(row);conditioned['executed_action']=action;conditioned['prediction_horizon']=3
            out=model([conditioned],tokenizer,requested=('future_body','future_rgb','uncertainty','curiosity'))
            rgb=out['future_rgb'][0];previous=torch.tensor(seen,device=rgb.device)
            novelty=float((previous-rgb).square().mean(-1).min())
            uncertainty=float(out['uncertainty'][0,0])
            progress=float(out['curiosity'][0,0])
            # Small uncertainty bonus; high error alone cannot dominate indefinitely.
            score=novelty*(.25+.75*progress)+min(uncertainty,.001)*.1
            scored.append({'action':action,'score':score,'predicted_novelty':novelty,'expected_learning_progress':progress,'uncertainty':uncertainty,
                'future_body':out['future_body'][0].cpu().tolist(),'future_rgb':rgb.cpu().tolist()})
    best=max(scored,key=lambda item:item['score'])
    return (best if best['score']>1e-5 else None),scored
