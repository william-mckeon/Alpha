"""Frozen diagnostic metrics. No training or model imports at module import time."""
import hashlib
import json
import math
import re
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_suite(root):
    root = Path(root)
    manifest = json.loads((root/'evaluation/arcus3/baseline-v1.json').read_text())
    for name, expected in manifest['files'].items():
        if sha(root/name) != expected:
            raise ValueError('Frozen suite hash mismatch: '+name)
    prompts = [json.loads(line) for line in (root/manifest['prompts']).read_text().splitlines() if line.strip()]
    if len(prompts) != 36 or len({p['id'] for p in prompts}) != 36:
        raise ValueError('Expected 36 unique preserved prompts')
    return manifest, prompts


def aggregate(rows):
    count = sum(r['target_tokens'] for r in rows)
    if not count or any(r['target_tokens'] <= 0 or not math.isfinite(r['nll_sum']) for r in rows):
        raise ValueError('Invalid scored tokens')
    nll = sum(r['nll_sum'] for r in rows)/count
    return {'target_tokens':count, 'nll':nll, 'perplexity':math.exp(nll)}


def masked_nll(logits, ids, start):
    """Score targets at [start, len(ids)); context prefix is never a target."""
    import torch
    if not 1 <= start < ids.shape[1] or logits.shape[:2] != ids.shape:
        raise ValueError('Invalid target span')
    loss = torch.nn.functional.cross_entropy(logits[:, start-1:-1].float().reshape(-1, logits.shape[-1]),
                                            ids[:, start:].reshape(-1), reduction='sum')
    return {'target_tokens':ids[:, start:].numel(), 'nll_sum':float(loss)}


def messages_for(item, settings):
    system = settings['system']
    if item.get('check') == 'tool':
        system += '\nReturn only a JSON object with keys name and arguments for one call. Available tool: '+json.dumps(item['tool_schema'], sort_keys=True)
    return [{'role':'system','content':system}, {'role':'user','content':item['prompt']}]


def score(item, response):
    text = response.strip()
    words = text.split()
    grams = [tuple(words[i:i+4]) for i in range(max(0,len(words)-3))]
    result = {'repeated_fourgram_fraction':1-len(set(grams))/len(grams) if grams else 0,
              'human_review':{'relevance':None,'fluency':None,'coherence':None}, 'task_success':None}
    if item.get('check') == 'exact':
        result['strict_exact'] = text in item['answers']
        result['task_success'] = result['strict_exact']
    if item.get('check') == 'python':
        code = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.S)
        result.update(source=code.group(1).strip() if code else text, execution='pending')
    if item.get('check') == 'tool':
        try:
            call = json.loads(text)
            parseable = isinstance(call,dict)
        except ValueError:
            call, parseable = None, False
        schema = item['tool_schema']
        args = call.get('arguments') if parseable else None
        valid = (parseable and set(call)=={'name','arguments'} and call['name']==schema['name']
                 and isinstance(args,dict) and set(args)==set(schema['parameters']['required'])
                 and all(isinstance(v,str) for v in args.values()))
        semantic = valid and call == item['expected_call']
        # A fixed in-memory fixture, never host files, shell, or the live web.
        fixtures = {'echo':'hello','lookup':'blue','read_file':'Remember to test changes.',
                    'search':['Python range produces an integer sequence.'], 'calculate':5,
                    'list_files':['notes.txt']}
        executed = bool(semantic)
        result.update(parseable=parseable,schema_valid=bool(valid),semantic_correct=bool(semantic),
                      executed=executed,tool_result=fixtures[schema['name']] if executed else None,
                      execution='frozen-in-memory-fixture' if executed else 'rejected',task_success=executed)
    return result


def summarize(records):
    groups = {}
    for row in records:
        group = groups.setdefault(row['category'], {'count':0,'passed':0,'measured':0,'truncated':0})
        group['count'] += 1
        group['truncated'] += int(row['truncated'])
        value = row['metrics']['task_success']
        if value is not None:
            group['measured'] += 1
            group['passed'] += int(value)
    return groups


def compatible(a,b):
    return all(a.get(k) is not None and a[k]==b.get(k) for k in
               ('suite_sha256','tokenizer_revision','settings_sha256','precision','orchestration'))
