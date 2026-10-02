"""Read-only checkpoint routing diagnosis: no optimizer, no saved model changes."""
import json,os,sys,time,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    if not Path('/.dockerenv').exists() or os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1':raise RuntimeError('Docker CUDA required')
    import torch
    from safetensors.torch import load_file
    from arcus3.donor import load,verify
    from arcus3.checkpoint import digest
    from arcus3.distillation import loss as kl_loss
    from baby_arcus.language_stream import atomic_json
    from baby_arcus.gpu_job_control import gpu_job
    out=Path('/output');started=time.monotonic()
    report={'training_updates':0,'optimizer_steps':0,'parent_updates':11008,'complete':False}
    def write(stage):
        report.update(stage=stage,seconds=time.monotonic()-started);atomic_json(out/'report.json',report)
        print(json.dumps({'stage':stage,'seconds':report['seconds']}),flush=True)
    def check():
        if (out/'pause-inference').exists() or time.monotonic()-started>1400:raise RuntimeError('Diagnostic stopped')
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.7)
    with gpu_job():
        try:
            write('load-verified-paused-checkpoint');verify('/donor')
            report['parent_sha256']=digest(Path('/parent/manifest.json'))
            if report['parent_sha256']!='eea89b0d88738fd794170d37cc6e2bd9c1eea8617702af98f208dadeb9e3d688':raise ValueError('Unexpected checkpoint')
            model,tok=load('/donor',converted='/converted',expanded='/parent')
            model.requires_grad_(False).eval()
            layers=[3,7,11,15,19,23];last=model.model.layers[-1].mlp
            saved_router=last.router.weight.detach().clone()
            # Diagnostics cover the exact legacy held-out language cohort and
            # two bounded training records per source, not a new benchmark.
            language=json.loads(Path('/app/evaluation/arcus3/baseline-v1.json').read_text())['language']
            rows=[]
            for item in language:
                ids=tok(item['text'],add_special_tokens=False)['input_ids']
                rows.append({'id':item['id'],'source':'heldout-'+item['domain'],'input_ids':ids,'labels':ids})
            selected={}
            with Path('/data/train-00000.jsonl').open() as f:
                for line in f:
                    r=json.loads(line);source=r.get('source','unknown')
                    if selected.get(source,0)>=2 or len(r['input_ids'])>512 or not any(x!=-100 for x in r['labels'][1:]):continue
                    rows.append({**r,'id':r['sha256']});selected[source]=selected.get(source,0)+1
                    if len(selected)==5 and min(selected.values())==2:break
            report['training_sample_by_source']=selected
            report['rows']=[];report['route_probability_totals']={str(i):{'positions':0,'new_choices':0,'new_probability_sum':0.,'margin_sum':0.,'entropy_sum':0.} for i in layers}
            captured={};handles=[]
            def pre(index):
                def hook(module,args):
                    h=args[0].detach();flat=h.reshape(-1,h.shape[-1]);z=module.router(flat.float()).float();p=z.softmax(-1)
                    stat=report['route_probability_totals'][str(index)];stat['positions']+=len(flat)
                    stat['new_choices']+=int((z.argmax(-1)==1).sum());stat['new_probability_sum']+=float(p[:,1].sum())
                    stat['margin_sum']+=float((z[:,1]-z[:,0]).sum());stat['entropy_sum']+=float(-(p*p.clamp_min(1e-9).log()).sum())
                    if index==23:captured['hidden']=h
                return hook
            for i in layers:handles.append(model.model.layers[i].mlp.register_forward_pre_hook(pre(i)))
            def residual_hook(module,args):captured['residual']=args[0].detach()
            handles.append(model.model.layers[-1].post_attention_layernorm.register_forward_pre_hook(residual_hook))
            write('counterfactual-experts-and-gradients')
            for row in rows:
                check();ids=torch.tensor([row['input_ids']],device='cuda');labels=torch.tensor(row['labels'][1:],device='cuda');mask=labels!=-100
                with torch.no_grad():
                    normal=model(ids,use_cache=False).logits[0,:-1].float()
                    normal_loss=torch.nn.functional.cross_entropy(normal,labels,ignore_index=-100)
                    h=captured['hidden'];residual=captured['residual'];flat=h.reshape(-1,h.shape[-1])
                    probabilities=last.router(flat.float()).softmax(-1)
                    expert_losses=[];expert_outputs=[]
                    for expert in last.experts:
                        value=expert(h.to(expert.gate_proj.weight.dtype)).to(h.dtype);expert_outputs.append(value)
                        output=model.lm_head(model.model.norm(residual+value))[0,:-1].float()
                        expert_losses.append(float(torch.nn.functional.cross_entropy(output,labels,ignore_index=-100)))
                    relative_mse=float((expert_outputs[1].float()-expert_outputs[0].float()).square().mean()/expert_outputs[0].float().square().mean().clamp_min(1e-9))
                record={'id':row['id'],'source':row['source'],'input_tokens':len(row['input_ids']),'targets':int(mask.sum()),
                        'natural_nll':float(normal_loss),'force_original_nll':expert_losses[0],'force_new_nll':expert_losses[1],
                        'new_probability_mean':float(probabilities[:,1].mean()),'new_selection_fraction':float((probabilities.argmax(-1)==1).float().mean()),
                        'local_expert_relative_mse':relative_mse}
                # All prior layers and both experts remain frozen; gradients only
                # measure the router objective at the captured last-layer inputs.
                for handle in handles:handle.remove()
                last.router.weight.requires_grad_(True);last.collect_aux=True
                local=last(h);pred=model.lm_head(model.model.norm(residual+local))[0,:-1].float()
                if not torch.equal(pred.detach(),normal):raise ValueError('Local last-layer reconstruction mismatch')
                task=torch.nn.functional.cross_entropy(pred,labels,ignore_index=-100)
                # Production averages the auxiliary loss across all six blocks.
                parts={'task':task,'weighted_balance':.01*last.last_aux/len(layers)}
                if 'sha256' in row:
                    target=load_file(str(Path('/teacher')/(row['sha256']+'.safetensors')))
                    parts['weighted_teacher']=.5*kl_loss(pred,target,mask)
                gradients={name:torch.autograd.grad(loss,last.router.weight,retain_graph=True)[0] for name,loss in parts.items()}
                record['router_gradients']={}
                for name,g in gradients.items():
                    # Sign of an SGD descent direction, not an Adam update forecast.
                    direction=-(flat.float() @ (g[1]-g[0]))
                    record['router_gradients'][name]={'l2':float(g.norm()),'mean_margin_descent_direction':float(direction.mean())}
                record['balance_loss']=float(last.last_aux.detach())
                last.router.weight.requires_grad_(False);last.collect_aux=False;last.last_aux=None
                del pred,local,parts,gradients,normal,expert_outputs
                handles=[model.model.layers[i].mlp.register_forward_pre_hook(pre(i)) for i in layers]
                handles.append(model.model.layers[-1].post_attention_layernorm.register_forward_pre_hook(residual_hook))
                report['rows'].append(record);write('row-'+str(len(report['rows'])))
            for handle in handles:handle.remove()
            report['router_unchanged']=bool(torch.equal(saved_router,last.router.weight))
            for stat in report['route_probability_totals'].values():
                n=stat['positions'];stat.update(new_selection_fraction=stat['new_choices']/n,new_probability_mean=stat['new_probability_sum']/n,
                                                mean_new_minus_original_logit=stat['margin_sum']/n,mean_entropy=stat['entropy_sum']/n)
            held=[r for r in report['rows'] if r['source'].startswith('heldout')]
            report['heldout_aggregate']={key:sum(r[key]*r['targets'] for r in held)/sum(r['targets'] for r in held)
                                         for key in ('natural_nll','force_original_nll','force_new_nll')}
            report['complete']=True;write('complete')
        except Exception as exc:
            report['error']={'type':type(exc).__name__,'message':str(exc)};write('failed');raise


if __name__=='__main__':main()
