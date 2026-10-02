"""Sequential disposable Alpha 3.2.2 warmup arms; no production checkpoints."""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(args):
    if not Path('/.dockerenv').exists() or os.environ.get('ARCUS3_CONTROLLED_DOCKER') != '1':
        raise RuntimeError('Controlled Docker CUDA required')
    import torch
    from safetensors.torch import load_file
    from arcus3.adapters import train_added_experts
    from arcus3.backbone_adaptation import update
    from arcus3.campaign import validate
    from arcus3.checkpoint import digest
    from arcus3.config import check_live, read, safe_child
    from arcus3.corpus_stream import CorpusStream
    from arcus3.donor import load, verify
    from arcus3.evaluation import aggregate, load_suite, masked_nll
    from arcus3.learning_rate import apply_for_update, build_optimizer, initial_state
    from arcus3.routing_objectives import configure
    from scripts.calibrate_arcus3_alpha322_schedule import validate_protocol
    from scripts.qualify_arcus3_training import frozen_digest
    from baby_arcus.gpu_job_control import gpu_job
    from baby_arcus.language_stream import atomic_json

    end=datetime.fromisoformat(args.deadline.replace('Z','+00:00'))
    if end.tzinfo is None or not 0<(end-datetime.now(timezone.utc)).total_seconds()<=86400:
        raise ValueError('Calibration deadline must be within the next 24 hours')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    base=validate(read(args.config));calibration=validate_protocol(read(args.calibration_config))
    if base.get('model_label')!='alpha3.2.2' or base.get('campaign_enabled'):
        raise ValueError('Calibration requires the disabled fresh Alpha 3.2.2 configuration')
    if calibration.get('lineage_id')!=base['lineage']['id']:
        raise ValueError('Calibration/config lineage mismatch')
    if calibration.get('peak_learning_rate')!=base['learning_rate']:
        raise ValueError('Calibration/config peak learning rate mismatch')
    qualification=read(args.qualification_report)
    if not qualification.get('qualified') or not qualification.get('exact_replay') or not qualification.get('frozen_unchanged'):
        raise ValueError('Live scheduler checkpoint replay qualification required')
    qualification_config_sha=hashlib.sha256(json.dumps({**base,'campaign_enabled':False},sort_keys=True).encode()).hexdigest()
    if (qualification.get('config_sha256')!=qualification_config_sha
            or qualification.get('max_input_tokens_tested')!=base['max_length']):
        raise ValueError('Replay qualification does not cover this exact schedule configuration and context')
    data=Path(args.data);teacher=Path(args.teacher);stream=CorpusStream(data,repeat=False)
    teacher_manifest=read(teacher/'manifest.json')
    from arcus3.tokenizer_contract import validate_data
    token_contract=validate_data(args.donor,stream.manifest)
    if base['max_length']!=token_contract['context_tokens']:
        raise ValueError('Calibration context must exactly match donor')
    if stream.manifest.get('qualification_only'):
        raise ValueError('Calibration requires the sealed production stage')
    if digest(Path(args.converted)/'manifest.json')!=base['parent_sha256']:
        raise ValueError('Calibration parent mismatch')
    verify(args.donor)
    if (teacher_manifest.get('data_sha256')!=stream.sha
            or teacher_manifest.get('donor_manifest_sha256')!=digest(Path(args.donor)/'manifest.json')
            or teacher_manifest.get('top_k')!=base['teacher_top_k']
            or teacher_manifest.get('teacher_input_tokens')!=stream.manifest['input_tokens_per_pass']):
        raise ValueError('Teacher/data lineage or coverage mismatch')
    rows=[];tokens=0
    while True:
        before=stream.snapshot()
        try:row=stream.next()
        except StopIteration:break
        if tokens+len(row['input_ids'])>calibration['tokens_per_arm']:
            stream=CorpusStream(data,before,repeat=False);break
        if len(row['input_ids'])>base['max_length'] or len(row['input_ids'])>qualification['max_input_tokens_tested']:
            raise ValueError('Calibration row exceeds qualified context')
        rows.append(row);tokens+=len(row['input_ids'])
    if not rows or tokens < calibration['tokens_per_arm']-base['max_length']:
        raise ValueError('Calibration prefix does not fill the bounded token budget')
    order_sha=hashlib.sha256(json.dumps([row['sha256'] for row in rows]).encode()).hexdigest()
    recipe={key:value for key,value in base.items() if key!='learning_rate_schedule'}
    initialization_recipe_sha=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()
    suite,_=load_suite('/app')
    report={'schema':'arcus3-alpha322-warmup-calibration-v1','complete':False,'campaign_updates':0,
            'requested_input_tokens_per_arm':calibration['tokens_per_arm'],'input_tokens_per_arm':tokens,
            'records_per_arm':len(rows),'record_order_sha256':order_sha,'arms':{},
            'qualification_report_sha256':digest(args.qualification_report),
            'note':'Every optimizer update is disposable; no arm writes a production checkpoint.'}
    atomic_json(output/'report.json',report)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_math_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    with gpu_job():
        torch.cuda.set_per_process_memory_fraction(.7)
        for arm in calibration['arms']:
            check_live(end,output);started=time.monotonic();torch.manual_seed(base['seed']);torch.cuda.reset_peak_memory_stats()
            cfg=json.loads(json.dumps(base));cfg['learning_rate_schedule']['warmup_input_tokens']=arm['warmup_input_tokens']
            cfg['learning_rate_schedule']['selection']['selected_warmup_input_tokens']=arm['warmup_input_tokens']
            validate(cfg);model,tokenizer=load(args.donor,converted=args.converted);train_added_experts(model)
            configure(model,cfg,initialize=True);frozen=frozen_digest(model);optimizer=build_optimizer(model,cfg);scheduler=initial_state(cfg)
            def language():
                model.eval();values=[]
                with torch.no_grad():
                    for item in suite['language']:
                        check_live(end,output)
                        ids=tokenizer(item['text'],return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
                        values.append(masked_nll(model(input_ids=ids,use_cache=False).logits,ids,1))
                return aggregate(values)
            before=language();gradient={'expert':0.0,'router':0.0,'gate':0.0};routing=None;position=0
            for row in rows:
                check_live(end,output);name=row['sha256']+'.safetensors';path=safe_child(teacher,name)
                if teacher_manifest['files'].get(name)!=digest(path):raise ValueError('Teacher shard changed')
                next_state=apply_for_update(optimizer,cfg,scheduler,position,len(row['input_ids']))
                metrics=update(model,optimizer,row,load_file(str(path)),cfg);position+=metrics['input_tokens'];scheduler=next_state
                for key in gradient:gradient[key]+=metrics['gradient_groups'][key]
                routing=metrics['routing_layers']
            after=language();arm_result={'schema':'arcus3-alpha322-warmup-arm-v1','disposable':True,
                'model_label':'alpha3.2.2','lineage_id':base['lineage']['id'],'arm_id':arm['id'],
                'warmup_input_tokens':arm['warmup_input_tokens'],
                'requested_input_tokens':calibration['tokens_per_arm'],'input_tokens':position,'updates':len(rows),
                'initialization_manifest_sha256':digest(Path(args.converted)/'manifest.json'),'data_sha256':stream.sha,
                'initialization_recipe_sha256':initialization_recipe_sha,
                'record_order_sha256':order_sha,'qualification_exact_replay':True,
                'scheduler_qualification_sha256':digest(args.qualification_report),'complete':True,
                'frozen_unchanged':frozen_digest(model)==frozen,'nll_before':before['nll'],'nll_after':after['nll'],
                'gradient_groups':gradient,'last_routing_layers':routing,'scheduler':scheduler,
                'seconds':time.monotonic()-started,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
                'campaign_updates':0,'production_checkpoint':None}
            arm_path=output/(arm['id']+'.json');atomic_json(arm_path,arm_result);report['arms'][arm['id']]=str(arm_path)
            atomic_json(output/'report.json',report);print(json.dumps({'arm':arm['id'],'before':before,'after':after}),flush=True)
            del optimizer,model,tokenizer;gc.collect();torch.cuda.empty_cache()
    report['complete']=True;atomic_json(output/'report.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name,default in [('donor','/donor'),('converted','/converted'),('teacher','/teacher'),('data','/data'),
                         ('output','/output'),('config','/app/configs/arcus3/backbone_adaptation_alpha322.json'),
                         ('calibration-config','/app/configs/arcus3/alpha322_schedule_calibration.json'),
                         ('qualification-report','/qualification.json')]:
        parser.add_argument('--'+name,default=default)
    parser.add_argument('--deadline',required=True);parser.add_argument('--max-new-tokens',type=int,default=128)
    main(parser.parse_args())
