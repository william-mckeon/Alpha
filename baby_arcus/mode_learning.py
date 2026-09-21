"""Isolated shared-backbone MoDE retraining; never activates a desktop checkpoint."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from arcus.tokenizer import get_tokenizer
from baby_arcus.body_policy import load as load_body
from baby_arcus.depth_policy import capacity, routing
from baby_arcus.language_checkpoint import load as load_language
from baby_arcus.language_model import LanguageAdapter
from baby_arcus.language_stream import documents, inventory, atomic_json
from baby_arcus.large_body_learning import file_hash
from baby_arcus.standing_environment import StandingEnvironment
from baby_arcus.lying_environment import LyingEnvironment
from baby_arcus.sitting_environment import SittingEnvironment

ENVIRONMENTS={'standing':StandingEnvironment,'lying':LyingEnvironment,'sitting':SittingEnvironment}


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def language_loss(body,adapter,ids):
    tokens=torch.tensor([ids],device=next(body.parameters()).device)
    logits=adapter(body.core,tokens[:,:-1])
    return F.cross_entropy(logits.flatten(0,1),tokens[:,1:].flatten())


def language_evaluate(body,adapter,windows,cap):
    with torch.no_grad(),capacity(body.core,cap):
        values=[float(language_loss(body,adapter,row)) for row in windows]
    return sum(values)/len(values)


def load_sources(cfg,device='cuda'):
    body_path=Path(cfg['body_checkpoint']);language_root=Path(cfg['language_root'])
    body,body_data=load_body(body_path)
    source_hash=file_hash(body_path)
    active=read_json(language_root/'active.json')
    if active['slot'] not in (0,1):raise ValueError('Invalid language slot')
    language_path=language_root/f"live-{active['slot']}.pt"
    if file_hash(language_path)!=active['sha256']:raise ValueError('Language source changed')
    dataset_cfg=read_json(cfg['dataset_config']);tokenizer=get_tokenizer(dataset_cfg['encoding'])
    adapter,language_data=load_language(language_path,body.core,tokenizer,source_hash,device)
    body.to(device).eval()
    sources={'body':str(body_path),'body_sha256':source_hash,'language':str(language_path),
             'language_sha256':active['sha256'],'language_metadata':language_data['metadata']}
    return body,adapter,tokenizer,sources


def corpus(cfg,tokenizer,root):
    data_cfg=read_json(cfg['dataset_config'])
    manifest=inventory(data_cfg['dataset_root'],data_cfg['source_patterns'])
    original=read_json(Path(cfg['language_root'])/'dataset-manifest.json')
    if manifest['fingerprint']!=original['fingerprint']:raise ValueError('Dataset identity changed')
    training=[];calibration=[];samples=[]
    for entry in manifest['files']:
        for number,text in documents(Path(manifest['root'])/entry['path']):
            if number>=40:break
            if number%10==0:continue
            split='train' if number<25 else 'calibration' if number>=31 else None
            if split is None:continue
            ids=tokenizer.encode('Environment:\n'+text,add_eot=True)
            windows=[ids[i:i+65] for i in range(0,min(len(ids)-64,512),64)]
            (training if split=='train' else calibration).extend(windows)
            samples.append({'file':entry['path'],'document':number,'split':split,
                'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'windows':len(windows)})
    if not training or not calibration:raise ValueError('Need train and calibration documents')
    validation=read_json(Path(cfg['language_root'])/'validation.json')
    atomic_json(root/'dataset-manifest.json',manifest);atomic_json(root/'samples.json',samples)
    atomic_json(root/'validation.json',validation)
    atomic_json(root/'calibration-language.json',calibration[:24])
    return training,calibration[:24],validation


def gradient_evidence(body):
    values=[]
    for block in body.core.blocks:
        for parameter in (block.router.fc.weight,block.moe.router.weight):
            values.append(parameter.grad.abs().sum() if parameter.grad is not None else torch.zeros((),device=parameter.device))
        expert=block.moe.experts.gate_proj
        values.extend(expert.grad.abs().sum(dim=(1,2)).unbind() if expert.grad is not None else
                      torch.zeros(expert.shape[0],device=expert.device).unbind())
    return torch.stack(values).detach()


def train(config):
    cfg=read_json(config);root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=False)
    if not (0<cfg['capacity']<1) or not 1<=cfg['updates']<=2000:raise ValueError('Invalid bounded MoDE pilot')
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);rng=random.Random(cfg['seed'])
    body,adapter,tokenizer,sources=load_sources(cfg)
    baseline_loss=None;parent=None;parent_updates={};parent_targets=0
    if cfg.get('initial_candidate'):
        parent=Path(cfg['initial_candidate'])
        gate=read_json(parent/'stage-report.json')
        if not gate['passed']:raise ValueError('Cannot continue from an unqualified stage')
        baseline_loss=gate['source_language_loss']
        body,adapter,parent_data=load_candidate(parent)
        if any(parent_data['sources'][k]!=sources[k] for k in ('body_sha256','language_sha256')):
            raise ValueError('Original source lineage changed between stages')
        if round((body.cfg.capacity-cfg['capacity'])*100)!=5:
            raise ValueError('Continuation must decrease capacity by exactly 0.05')
        parent_updates=parent_data.get('cumulative_updates',parent_data['updates'])
        parent_targets=parent_data.get('cumulative_language_training_targets',parent_data['language_training_targets'])
        del parent_data
    teacher,_=load_body(cfg['body_checkpoint']);teacher.to('cuda').eval().requires_grad_(False)
    train_rows,_,validation=corpus(cfg,tokenizer,root)
    if baseline_loss is None:baseline_loss=language_evaluate(body,adapter,validation,1.0)
    before_loss=language_evaluate(body,adapter,validation,cfg['capacity'])
    body.cfg.capacity=cfg['capacity']
    for block in body.core.blocks:block.capacity=cfg['capacity']
    body.requires_grad_(True);adapter.requires_grad_(True)
    # The listening/expression bandit is a separate policy, retained without inventing new rewards.
    adapter.choice.requires_grad_(False)
    router_params=[];core_params=[];head_params=[]
    for name,parameter in body.named_parameters():
        (router_params if '.router.' in name else core_params if name.startswith('core.') else head_params).append(parameter)
    head_params.extend(p for p in adapter.parameters() if p.requires_grad)
    optimizer=torch.optim.AdamW([
        {'params':core_params,'lr':cfg['core_lr'],'name':'core_and_experts'},
        {'params':router_params,'lr':cfg['router_lr'],'name':'depth_and_expert_routers'},
        {'params':head_params,'lr':cfg['head_lr'],'name':'motor_and_language_heads'}],weight_decay=0)
    if parent:
        parent_report=read_json(parent/'training-report.json')
        if file_hash(parent/'optimizer.pt')!=parent_report['optimizer_sha256']:
            raise ValueError('Parent optimizer checksum mismatch')
        optimizer.load_state_dict(torch.load(parent/'optimizer.pt',map_location='cpu',weights_only=True))
    parameters=[p for group in optimizer.param_groups for p in group['params']]
    envs={goal:[factory(cfg['seed']+i,hold_ticks=10) for i in range(cfg['batch_size'])]
          for goal,factory in ENVIRONMENTS.items()}
    schedule=('standing','lying','sitting','approach','language')
    counters=defaultdict(int);gradient_max=None;train_targets=0;started=time.monotonic()
    atomic_json(root/'config.json',cfg)
    atomic_json(root/'initial.json',{'capacity':cfg['capacity'],'source_language_loss':baseline_loss,'before_training_loss':before_loss,
        'sources':sources,'parameters':sum(p.numel() for p in body.parameters())+sum(p.numel() for p in adapter.parameters())})
    print(json.dumps({'stage':'start','capacity':cfg['capacity'],'baseline_language_loss':baseline_loss,'before_training_loss':before_loss}),flush=True)
    with (root/'training.jsonl').open('w',encoding='utf-8') as log:
        for index in range(cfg['updates']):
            task=schedule[index%len(schedule)]
            optimizer.zero_grad(set_to_none=True)
            if task=='language':
                ids=rng.choice(train_rows);task_loss=language_loss(body,adapter,ids);train_targets+=len(ids)-1
                preservation=task_loss.new_zeros(())
            elif task=='approach':
                senses=[env.observe() for env in envs['standing']]
                relative=[[rng.uniform(-9,9),rng.uniform(-6,6)] for _ in senses]
                with torch.no_grad():old=teacher.approach(senses,relative).softmax(-1)
                logits=body.approach(senses,relative);prob=logits.softmax(-1)
                xy=torch.tensor(relative,device='cuda');directions=torch.tensor([[0,-.32],[0,.32],[-.32,0],[.32,0]],device='cuda')
                rewards=xy.norm(dim=-1)[:,None]-(xy[:,None,:]-directions).norm(dim=-1)
                task_loss=-(prob*rewards).sum(-1).mean()
                preservation=F.kl_div(logits.log_softmax(-1),old,reduction='batchmean')
            else:
                senses=[env.observe() for env in envs[task]]
                with torch.no_grad():old=teacher(senses,task).softmax(-1)
                logits=body(senses,task);distribution=torch.distributions.Categorical(logits=logits)
                actions=distribution.sample();rewards=[]
                for i,(env,action) in enumerate(zip(envs[task],actions.tolist())):
                    _,reward,done=env.step(action);rewards.append(reward)
                    if done:envs[task][i]=ENVIRONMENTS[task](cfg['seed']+100000+index*cfg['batch_size']+i,hold_ticks=10)
                reward=torch.tensor(rewards,device='cuda')
                task_loss=-(distribution.log_prob(actions)*(reward-reward.mean())).mean()-.01*distribution.entropy().mean()
                preservation=F.kl_div(logits.log_softmax(-1),old,reduction='batchmean')
            objective=task_loss+cfg['preservation_kl_weight']*preservation+body.core.last_aux_loss
            if not torch.isfinite(objective):raise RuntimeError('Nonfinite MoDE objective')
            objective.backward();evidence=gradient_evidence(body)
            if not torch.isfinite(evidence).all():raise RuntimeError('Nonfinite router/expert gradients')
            gradient_max=evidence if gradient_max is None else torch.maximum(gradient_max,evidence)
            torch.nn.utils.clip_grad_norm_(parameters,1,error_if_nonfinite=True);optimizer.step();counters[task]+=1
            row={'update':index+1,'task':task,'task_loss':float(task_loss.detach()),
                 'preservation_kl':float(preservation.detach()),'language_training_targets':train_targets,
                 'routing':routing(body.core),'elapsed_seconds':time.monotonic()-started}
            log.write(json.dumps(row)+'\n')
            if (index+1)%50==0:log.flush();print(json.dumps({k:row[k] for k in ('update','task','task_loss','language_training_targets','elapsed_seconds')}),flush=True)
    final_loss=language_evaluate(body,adapter,validation,cfg['capacity'])
    gradients=gradient_max.cpu().reshape(body.cfg.n_layers,2+body.cfg.n_experts).tolist()
    changed=[]
    for i,(new,old) in enumerate(zip(body.core.blocks,teacher.core.blocks)):
        changed.append({'layer':i,'depth_router':not torch.equal(new.router.fc.weight,old.router.fc.weight),
            'expert_router':not torch.equal(new.moe.router.weight,old.moe.router.weight),
            'experts':[not torch.equal(new.moe.experts.gate_proj[e],old.moe.experts.gate_proj[e]) for e in range(body.cfg.n_experts)]})
    source_unchanged=all(file_hash(sources[key])==sources[key+'_sha256'] for key in ('body','language'))
    payload={'schema':'arcus-mode-joint-v1','body_config':asdict(body.cfg),'body':body.state_dict(),
        'language':adapter.state_dict(),'text_dim':adapter.embedding.embedding_dim,'vocab_size':tokenizer.vocab_size,
        'encoding':tokenizer.encoding_name,'sources':sources,'updates':dict(counters),
        'parent_candidate':str(parent) if parent else None,
        'parent_model_sha256':file_hash(parent/'model.pt') if parent else None,
        'cumulative_updates':{task:parent_updates.get(task,0)+count for task,count in counters.items()},
        'cumulative_language_training_targets':parent_targets+train_targets,
        'language_training_targets':train_targets,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
    torch.save(payload,root/'model.pending');(root/'model.pending').replace(root/'model.pt')
    torch.save(optimizer.state_dict(),root/'optimizer.pending');(root/'optimizer.pending').replace(root/'optimizer.pt')
    report={'capacity':cfg['capacity'],'source_language_loss':baseline_loss,'before_training_loss':before_loss,'after_training_loss':final_loss,
        'parent_candidate':payload['parent_candidate'],'parent_model_sha256':payload['parent_model_sha256'],
        'optimizer_continued':parent is not None,'cumulative_updates':payload['cumulative_updates'],
        'cumulative_language_training_targets':payload['cumulative_language_training_targets'],
        'updates':dict(counters),'language_training_targets':train_targets,'source_unchanged':source_unchanged,
        'gradient_columns':['depth_router','expert_router']+[f'expert_{i}' for i in range(body.cfg.n_experts)],
        'max_gradient_by_layer':gradients,'weights_changed':changed,
        'all_routing_and_expert_groups_received_gradients':bool((gradient_max>0).all()),
        'model_sha256':file_hash(root/'model.pt'),'optimizer_sha256':file_hash(root/'optimizer.pt'),
        'seconds':time.monotonic()-started,'activated':False}
    atomic_json(root/'training-report.json',report);print(json.dumps(report),flush=True)


def load_candidate(root,device='cuda'):
    from arcus.model_config import ModelConfig
    from baby_arcus.body_policy import BodyPolicy
    root=Path(root);report=read_json(root/'training-report.json')
    if file_hash(root/'model.pt')!=report['model_sha256']:raise ValueError('Candidate checksum mismatch')
    data=torch.load(root/'model.pt',map_location='cpu',weights_only=True)
    if data['schema']!='arcus-mode-joint-v1':raise ValueError('Wrong joint model schema')
    body=BodyPolicy(ModelConfig(**data['body_config']),lying=True,sitting=True,approach=True)
    body.load_state_dict(data['body']);body.to(device).eval()
    adapter=LanguageAdapter(body.cfg.dim,data['vocab_size'],data['text_dim'])
    adapter.load_state_dict(data['language']);adapter.to(device).eval()
    return body,adapter,data


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/baby_arcus/mode_025.json')
    train(parser.parse_args().config)
