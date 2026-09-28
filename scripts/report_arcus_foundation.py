"""Render separate capability tracks without inventing an intelligence score."""
import argparse
import html
import json
from pathlib import Path


def report(records):
    if not records or any(not r.get('checkpoint_unchanged') for r in records):raise ValueError('Verified read-only evidence required')
    lineages={r['lineage'] for r in records}
    if len(lineages)!=1:raise ValueError('Separate reports for different training lineages')
    identities={r.get('evaluation_identity') for r in records}
    if None in identities or len(identities)!=1:raise ValueError('Matching evaluation settings, corpus and tokenizer required')
    return '<!doctype html><meta charset="utf-8"><title>Arcus foundation evaluation</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto}pre{white-space:pre-wrap}</style><h1>Arcus developmental evidence</h1><p>Separate language, completion, chat, choice and context diagnostics. Missing benchmark scores are unmeasured.</p>'+''.join('<h2>'+str(r['candidate']['updates'])+' updates</h2><pre>'+html.escape(json.dumps(r,indent=2))+'</pre>' for r in records)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reports',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.output).exists():raise ValueError('Report already exists')
    Path(a.output).write_text(report([json.loads(Path(p).read_text()) for p in a.reports]),encoding='utf-8')
