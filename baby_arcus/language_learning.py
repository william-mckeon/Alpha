"""Bounded next-token learning from provenance-tagged passages and human speech."""
import argparse,hashlib,json,random,time
from pathlib import Path
import torch
import torch.nn.functional as F
from arcus.tokenizer import get_tokenizer
from baby_arcus.body_policy import load as load_body
from baby_arcus.language_model import LanguageAdapter
from baby_arcus.language_checkpoint import save
from baby_arcus.language_stream import inventory,documents,atomic_json
from baby_arcus.large_body_learning import file_hash

def loss(adapter,core,ids,device):
    data=torch.tensor([ids],device=device)
    logits=adapter(core,data[:,:-1])
    return F.cross_entropy(logits.flatten(0,1),data[:,1:].flatten())

def update(adapter,core,optimizer,ids,device):
    optimizer.zero_grad(set_to_none=True)
    value=loss(adapter,core,ids,device)
    if not torch.isfinite(value):raise RuntimeError('Nonfinite language loss')
    value.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1);optimizer.step()
    return float(value.detach())

def bootstrap(config_path,steps=256):
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8'));root=Path(cfg['language_root']);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(920);rng=random.Random(920)
    tokenizer=get_tokenizer(cfg['encoding']);manifest=inventory(cfg['dataset_root'],cfg['source_patterns'])
    atomic_json(root/'dataset-manifest.json',manifest)
    body,_=load_body(cfg['body_checkpoint']);parent=file_hash(cfg['body_checkpoint']);body.to('cuda').eval().requires_grad_(False)
    adapter=LanguageAdapter(body.cfg.dim,tokenizer.vocab_size,cfg['text_dim']).to('cuda')
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=.001)
    train=[];validation=[];evidence=[]
    for entry in manifest['files']:
        for number,text in documents(Path(manifest['root'])/entry['path']):
            ids=tokenizer.encode('Environment:\n'+text,add_eot=True)
            windows=[ids[i:i+65] for i in range(0,min(len(ids)-64,512),64)]
            (validation if number%10==0 else train).extend(windows)
            evidence.append({'file':entry['path'],'document':number,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                             'split':'validation' if number%10==0 else 'train','sampled_tokens':sum(len(x)-1 for x in windows)})
            if number>=24:break
    if not train or not validation:raise ValueError('Need disjoint train and held-out documents')
    validation=validation[:16]
    atomic_json(root/'sample-manifest.json',evidence)
    def evaluate():
        with torch.no_grad():return sum(float(loss(adapter,body.core,ids,'cuda')) for ids in validation)/len(validation)
    atomic_json(root/'validation.json',validation)
    initial=evaluate();updates=0;tokens=0;sampled=set()
    with (root/'training.jsonl').open('w') as log:
        for index in range(steps):
            window=rng.randrange(len(train));ids=train[window];sampled.add(window)
            value=update(adapter,body.core,optimizer,ids,'cuda');updates+=1;tokens+=len(ids)-1
            log.write(json.dumps({'update':updates,'loss':value,'training_tokens':tokens})+'\n')
            if updates%32==0:print(json.dumps({'update':updates,'loss':value,'tokens':tokens}),flush=True)
    final=evaluate()
    import importlib.metadata
    meta={'parent_sha256':parent,'encoding':cfg['encoding'],'tiktoken_version':importlib.metadata.version('tiktoken'),
          'vocab_size':tokenizer.vocab_size,'text_dim':cfg['text_dim'],'updates':updates,'training_tokens':tokens,
          'sampled_window_tokens_without_repeats':sum(len(train[i])-1 for i in sampled),
          'available_training_tokens':sum(len(x)-1 for x in train),'validation_loss_before':initial,'validation_loss_after':final,
          'motor_parameters':sum(p.numel() for p in body.parameters()),'language_parameters':sum(p.numel() for p in adapter.parameters())}
    save(root/'bootstrap.pt',adapter,optimizer,meta)
    # Source parent is never saved by this trainer; compare actual source tensors too.
    original,_=load_body(cfg['body_checkpoint'])
    preserved=all(torch.equal(v.cpu(),original.state_dict()[k]) for k,v in body.state_dict().items())
    valid=final<initial and preserved and file_hash(cfg['body_checkpoint'])==parent
    report={**meta,'body_weights_preserved':preserved,'gate_passed':valid,'checkpoint_sha256':file_hash(root/'bootstrap.pt'),
            'gate_scope':'pipeline, held-out loss improvement, frozen motor tensors; not language comprehension'}
    atomic_json(root/'qualification.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/baby_arcus/language.json');p.add_argument('--steps',type=int,default=256);a=p.parse_args()
    if not 1<=a.steps<=2000:p.error('steps must be 1..2000')
    bootstrap(a.config,a.steps)
