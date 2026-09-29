"""CPU-only report generation; reject incomplete or incompatible comparisons."""
import argparse,json
from pathlib import Path

def comparison(report):
    if not report.get('complete') or not report.get('matched_exposure'):raise ValueError('Incomplete matched run')
    arms=report['arms']
    states=[arms[k]['milestones'][-1]['state'] for k in ('dense','expanded')]
    for key in ('updates','cursor','target_tokens','input_tokens','data_sha256','config_sha256'):
        if states[0][key]!=states[1][key]:raise ValueError('Incompatible exposure/identity: '+key)
    for arm in arms.values():
        if [m['state']['updates'] for m in arm['milestones']]!=[16,32,64] or not arm['frozen_unchanged']:raise ValueError('Missing milestones or changed base')
    lines=['# Phase 7 matched experiment','',
           'Single seed, small donor-related dataset. Equal target exposure does not imply equal compute or trainable parameters. No general specialization or depth-skipping capability is established.','',
           '| Arm | Updates | Target tokens | Input tokens | Trainable parameters | Initial NLL | Final NLL | Final PPL | Peak CUDA bytes |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,a in arms.items():
        m=a['milestones'][-1];s=m['state'];h=m['heldout']
        lines.append(f"| {name} | {s['updates']} | {s['target_tokens']} | {s['input_tokens']} | {a['trainable_parameters']} | {a['baseline']['nll']:.9f} | {h['nll']:.9f} | {h['perplexity']:.9f} | {a['peak_cuda_bytes']} |")
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    (root/'comparison.md').write_text(comparison(json.loads((root/'campaign-report.json').read_text())),encoding='utf-8')
