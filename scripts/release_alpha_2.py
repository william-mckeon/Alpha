"""Verified private Alpha releases: completed 60k or explicitly selected research snapshot."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_snapshot(spec):
    root=Path(spec['run_root']);pointer=read(root/'candidate.json')
    if spec.get('private') is not True or spec.get('repo_id')!='Islanderintel/Alpha-2.0':
        raise ValueError('Snapshot requires the selected private repository')
    if spec.get('authorization')!='user-requested-current-checkpoint-publication':
        raise ValueError('Explicit snapshot authorization required')
    if pointer!=spec.get('expected_candidate') or pointer['updates']!=spec['required_updates']:
        raise ValueError('Snapshot checkpoint differs from explicitly selected identity')
    if not (root/'pause-training').exists():raise ValueError('Snapshot training must remain paused')
    if digest(root/(pointer['generation']+'.pt'))!=pointer['sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    reports={key:read(spec[key]) for key in ('developmental_report','mapping_report','agent_report')}
    for report in reports.values():
        if report.get('candidate')!=pointer or not report.get('complete') or not report.get('checkpoint_unchanged'):
            raise ValueError('Incomplete or mismatched snapshot diagnostics')
    development=reports['developmental_report'];mapping=reports['mapping_report']
    if (len(development.get('records',[]))!=36 or len(mapping.get('traces',[]))!=36
            or development.get('executor_controls')!={'correct_passed':True,'incorrect_rejected':True}
            or mapping['inventory']['unique_parameters']!=128353994):
        raise ValueError('Incomplete snapshot diagnostic cohorts or controls')
    return pointer, {'release_kind':'research-snapshot','completed_60000':False,
        'checkpoint_updates':pointer['updates'],'scores':reports['agent_report']['scores'],
        'developmental_summary':development['summary'],'developmental_identity':development['identity'],
        'unique_parameters':mapping['inventory']['unique_parameters'],
        'evidence_hashes':{key:digest(spec[key]) for key in reports},
        'limitations':'Single seed; small repeated cohorts; human qualitative ratings pending; 16k configured, not demonstrated. This is not a completed 60k run.'}


def validate_evidence(spec):
    if spec.get('release_kind')=='research-snapshot':return validate_snapshot(spec)
    root=Path(spec['run_root']);pointer=read(root/'candidate.json')
    if spec.get('private') is not True or spec['repo_id']!='Islanderintel/Alpha-2.0':
        raise ValueError('Alpha 2.0 requires the selected private repository')
    if spec['required_updates']!=60000 or pointer['updates']!=60000:
        raise ValueError('Release requires exactly 60000 durable updates')
    if spec['milestones']!=list(range(42000,60001,1000)):
        raise ValueError('All 19 milestones required')
    if digest(root/(pointer['generation']+'.pt'))!=pointer['sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    milestones=[]
    for step in spec['milestones']:
        comparison=read(root/'pilot'/f'comparison-{step}.json')
        evaluation=read(root/'pilot'/f'evaluation-{step}'/'evaluation.json')
        cp=comparison['candidate']
        if (cp['updates']!=step or evaluation['candidate']!=cp or not evaluation.get('complete')
                or not evaluation.get('checkpoint_unchanged') or not evaluation.get('coding_execution_evaluated')):
            raise ValueError('Incomplete milestone evidence')
        if digest(root/(cp['generation']+'.pt'))!=cp['sha256']:
            raise ValueError('Milestone checkpoint checksum mismatch')
        from scripts.report_alpha_tool_correction import scores
        if scores(evaluation)!=comparison['after']:
            raise ValueError('Milestone scores differ from evaluation')
        if step==60000 and (cp!=pointer or not comparison['gates'].get('language_retention')):
            raise ValueError('Final checkpoint or retention review is unresolved')
        milestones.append({'updates':step,'sha256':cp['sha256'],'scores':comparison['after'],
                           'gates':comparison['gates'],
                           'baseline_sha256':comparison['baseline']['sha256'],
                           'evaluation_identity_sha256':hashlib.sha256(json.dumps(evaluation['evaluation_identity'],sort_keys=True).encode()).hexdigest()})
    development=read(root/spec['developmental_report']);mapping=read(root/spec['mapping_report'])
    for report in (development,mapping):
        if report.get('candidate')!=pointer or not report.get('complete') or not report.get('checkpoint_unchanged'):
            raise ValueError('Missing matched final diagnostics')
    if len(development.get('records',[]))!=36 or not mapping.get('traces') or not development.get('executor_controls',{}).get('correct_passed'):
        raise ValueError('Incomplete final diagnostic cohorts')
    if mapping['inventory']['unique_parameters']!=128353994:
        raise ValueError('Unexpected model parameter count')
    return pointer, {'milestones':milestones,'developmental_summary':development['summary'],
                     'developmental_identity':development['identity'],
                     'unique_parameters':mapping['inventory']['unique_parameters'],
                     'limitations':'Single seed, small repeated cohorts, subjective review pending; configured 16k context is not proven competence.'}


def verify_package(directory):
    directory=Path(directory);manifest=read(directory/'manifest.json')
    paths=set(manifest['files'])
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    if actual != paths|{'manifest.json'}:raise ValueError('Unexpected or missing package files')
    for name, expected in manifest['files'].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path)!=expected:
            raise ValueError('Package hash/path mismatch: '+name)
    return manifest


def copy_inference_sources(destination):
    """Copy runtime code and only the explicitly needed observation assets."""
    destination=Path(destination)
    for folder in ('arcus','baby_arcus'):
        for source in Path(folder).rglob('*.py'):
            target=destination/source;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    for name in ('arcus-lying.png','arcus-body.png','arcus-sitting.png'):
        source=Path('baby_arcus/web')/name;target=destination/source
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    shutil.copy2('LICENSE',destination/'LICENSE')
    if Path('NOTICE').exists():shutil.copy2('NOTICE',destination/'NOTICE')


def package(spec):
    pointer,evidence=validate_evidence(spec)
    from baby_arcus.runtime_contract import require_gpu
    from baby_arcus.gpu_job_control import gpu_job
    from baby_arcus.shared_checkpoint import load
    from safetensors.torch import save_model
    from scripts.package_alpha_hf import LOADER
    from baby_arcus.conversation_probe import respond
    from arcus.tokenizer import get_tokenizer
    import torch
    require_gpu();torch.cuda.set_per_process_memory_fraction(.70);torch.set_num_threads(2)
    destination=Path(spec['destination'])
    if destination.exists():raise ValueError('Refusing to replace an existing package')
    destination.mkdir(parents=True)
    with gpu_job():
        model,data=load(spec['run_root'],pointer,'cuda');model.eval().requires_grad_(False)
        if (data['progress']['updates']!=spec['required_updates'] or data['progress'].get('initialization')!='random'
                or model.body.cfg.capacity!=1.0 or model.body.cfg.max_seq_len!=16384
                or sum(p.numel() for p in model.parameters())!=128353994):
            raise ValueError('Checkpoint content does not match the Alpha 2.0 release identity')
        metadata={'family':'Alpha','model_name':'Alpha-2.0','body_config':data['body_config'],
            'schema_version':int(data['schema'].rsplit('v',1)[1]),'vocab_size':data['vocab_size'],
            'text_dim':data['text_dim'],'integrated_motor':data.get('integrated_motor',False),
            'experiment_depth_capacity':data.get('experiment_depth_capacity'),
            'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_updates':spec['required_updates'],
            'release_kind':spec.get('release_kind','completed-60000'),
            'encoding':'o200k_base','trained_tokens':data['progress'].get('trained_tokens')}
        del data
        tokenizer=get_tokenizer('o200k_base')
        before=respond(model,tokenizer,'Hi, how are you?')['token_ids']
        tensor_hashes={k:hashlib.sha256(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}
        save_model(model,destination/'model.safetensors')
        (destination/'alpha_config.json').write_text(json.dumps(metadata,indent=2))
        (destination/'load_alpha.py').write_text(LOADER)
        del model;torch.cuda.empty_cache()
        import importlib.util
        sys.dont_write_bytecode=True
        module_spec=importlib.util.spec_from_file_location('alpha_export_loader',destination/'load_alpha.py')
        loader=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(loader)
        reloaded,_=loader.load_alpha(destination,device='cuda')
        actual={k:hashlib.sha256(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in reloaded.state_dict().items()}
        if actual!=tensor_hashes or respond(reloaded,tokenizer,'Hi, how are you?')['token_ids']!=before:
            raise ValueError('Export tensor or deterministic inference mismatch')
        del reloaded
    # Only source code and explicitly sanitized evidence; never copy a run directory.
    copy_inference_sources(destination)
    (destination/'requirements.txt').write_text('torch==2.11.0\nsafetensors>=0.4\ntiktoken==0.14.0\nPillow>=12\nnumpy>=2\n')
    (destination/'README.md').write_text(Path(spec.get('model_card','docs/ALPHA_2_MODEL_CARD.md')).read_text()+'\n\nActual architecture and parameter count: see alpha_config.json.\n')
    (destination/'evaluation-summary.json').write_text(json.dumps(evidence,indent=2))
    manifest={'model_name':'Alpha-2.0','source_checkpoint':pointer,'inference_only':True,
              'release_kind':spec.get('release_kind','completed-60000'),
              'validated_tensor_identity':True,'validated_greedy_parity':True,
              'files':{p.relative_to(destination).as_posix():digest(p) for p in destination.rglob('*') if p.is_file()}}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if digest(Path(spec['run_root'])/(pointer['generation']+'.pt'))!=pointer['sha256']:
        raise ValueError('Source checkpoint changed during export')
    verify_package(destination)
    return destination


def publish(spec):
    # Revalidate final evidence and package on every explicit publication attempt.
    pointer,_=validate_evidence(spec);directory=Path(spec['destination']);manifest=verify_package(directory)
    if manifest['source_checkpoint']!=pointer or not manifest.get('validated_tensor_identity') or not manifest.get('validated_greedy_parity'):
        raise ValueError('Unverified package')
    from huggingface_hub import HfApi, hf_hub_download
    import os
    api=HfApi(token=os.environ['HF_TOKEN']);repo=spec['repo_id']
    api.create_repo(repo_id=repo,repo_type='model',private=True,exist_ok=True)
    if api.model_info(repo).private is not True:raise ValueError('Refusing to upload to a public repository')
    commit=api.upload_folder(repo_id=repo,repo_type='model',folder_path=directory,
                             commit_message=f"Verified Alpha 2.0 {spec.get('release_kind','completed run')} at {pointer['updates']} updates")
    revision=commit.oid
    if api.model_info(repo,revision=revision).private is not True:raise ValueError('Repository privacy changed')
    for name,expected in {**manifest['files'],'manifest.json':digest(directory/'manifest.json')}.items():
        path=hf_hub_download(repo,name,revision=revision,token=os.environ['HF_TOKEN'])
        if digest(path)!=expected:raise ValueError('Remote file mismatch: '+name)
    record={'repo_id':repo,'revision':revision,'private':True,'all_files_verified':True,'candidate':pointer}
    # Per-commit record preserves all previous publications.
    path=directory.parent/('Alpha-2.0-publication-'+revision+'.json')
    path.write_text(json.dumps(record,indent=2));return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',default='configs/baby_arcus/alpha_2_release.json')
    p.add_argument('action',choices=('validate','package','publish'));a=p.parse_args();spec=read(a.spec)
    result={'validate':validate_evidence,'package':package,'publish':publish}[a.action](spec)
    print(json.dumps(result,default=str,indent=2))
