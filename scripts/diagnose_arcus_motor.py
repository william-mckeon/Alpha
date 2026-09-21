"""Compare primitive and contextual motor paths on identical held-out poses."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.shared_checkpoint import load
from baby_arcus.body_policy import load as load_body
from baby_arcus.shared_experience import capture
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.standing_environment import StandingEnvironment
from baby_arcus.rest_environment import features

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--episodes',type=int,default=5);a=p.parse_args()
    torch.set_num_threads(2);c=json.loads(Path(a.config).read_text());root=Path(c['root'])
    manifest=json.loads((root/'candidate.json').read_text());device='cuda' if torch.cuda.is_available() else 'cpu'
    model,_=load(root,manifest,device);model.eval().requires_grad_(False)
    parent,_=load_body(c['body_checkpoint']);parent.to(device).eval().requires_grad_(False)
    from arcus.tokenizer import get_tokenizer
    tokenizer=get_tokenizer(c['encoding']);app=PlayroomApplication();row=capture(app);app.close()
    row['vision'].update(available=False,image_base64='',sha256=__import__('hashlib').sha256(b'').hexdigest());row['hearing']=[];row['objects']=[]
    results=[]
    with torch.no_grad():
        for seed in range(9711800,9711800+a.episodes):
            for mode in ('parent','primitive','contextual'):
                env=StandingEnvironment(seed);done=False
                while not done:
                    senses=env.observe()
                    if mode=='parent':action=parent.choose(senses,'standing')
                    elif mode=='primitive':action=model.body.choose(senses,'standing')
                    else:
                        row['senses']=senses;row['internal']=features(env.session.body)
                        action=int(model([row],tokenizer,requested=('body',))['body'][0].argmax())
                    _,_,done=env.step(action)
                result=dict(seed=seed,mode=mode,success=env.success,steps=env.steps);results.append(result);print(json.dumps(result),flush=True)
    (root/'motor-diagnostic.json').write_text(json.dumps(dict(candidate=manifest,episodes=results),indent=2))

if __name__=='__main__':main()
