"""Symbolic toy observations: undiscovered effects are never policy inputs."""
import math
from baby_arcus.contracts import ContractError

def observe(world):
    if world.paused or world.view.held or world.view.region!='playpen' or world.body.sleep_state!='awake' or world.body.eyelid_openness<=0:
        raise ContractError('Exploration requires awake open eyes in the active playpen')
    p=world.environment.placements[world.body.entity_id]
    return [{'id':obj['id'],'features':[float(obj['discovered'] is None),
             math.hypot(obj['x']-p['x'],obj['y']-p['y'])/12,
             min(obj['visits'],5)/5,world.body.rest_need]}
            for obj in world.environment.objects.values()]

def utility(x):
    unknown,distance,visits,need=x
    return 2*unknown-.5*distance-.5*visits-1.5*need-.5

def score(artifact,x):
    w,b,v,c=artifact['weights']
    hidden=[math.tanh(sum(a*z for a,z in zip(row,x))+bias) for row,bias in zip(w,b)]
    return sum(a*z for a,z in zip(v[0],hidden))+c[0]

def choose(artifact,observations):
    ranked=[(score(artifact,row['features']),row['id']) for row in observations]
    if not ranked:return None
    value,key=max(ranked)
    return key if value>0 else None
