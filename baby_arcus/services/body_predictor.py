"""Private stdio predictor: no OS tools, credentials, or direct body mutation."""
import argparse
import json
import sys
import torch
from baby_arcus.body_policy import load
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);a=p.parse_args()
    torch.set_num_threads(2)
    model,_=load(a.checkpoint);model.eval()
    count=sum(p.numel() for p in model.parameters())
    device="cuda" if count>1000000 and torch.cuda.is_available() else "cpu"
    model.to(device)
    goals=["standing"]+(["lying"] if hasattr(model,"lying_actor") else [])+(["sitting"] if hasattr(model,"sitting_actor") else [])
    print(json.dumps({"ready":True,"parameters":count,"device":device,"goals":goals}),flush=True)
    for line in sys.stdin:
        data=json.loads(line)
        if 'observation' in data:
            obs=data['observation'];events=obs['events'];model.receive_events(events)
            reply={'received':[e['sequence'] for e in events],'body_action':None,'arrived':False}
            goal=data.get('goal');senses=obs['senses']
            if goal:
                from baby_arcus.body_vocabulary import ACTIONS
                if goal=='approach':
                    from baby_arcus.approach_vocabulary import ACTIONS as MOVES,ARRIVAL
                    import math
                    if math.hypot(*obs['relative_target'])<=ARRIVAL:reply['arrived']=True
                    elif senses['height']<.99 or not senses['stable']:reply['body_action']=ACTIONS[model.choose(senses,'standing')]
                    else:
                        with torch.no_grad():index=int(model.approach([senses],[obs['relative_target']]).argmax(-1)[0])
                        reply['body_action']=MOVES[index]
                else:reply['body_action']=ACTIONS[model.choose(senses,goal)]
            print(json.dumps(reply),flush=True);continue
        print(json.dumps({"action":model.choose(data["senses"],data.get("goal","standing"))}),flush=True)
if __name__=="__main__":main()
