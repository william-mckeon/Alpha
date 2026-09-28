"""Separate deterministic checks and explicit human review; never train on this suite."""
import ast
import hashlib
import json
import re
from pathlib import Path


def suite_identity(paths):
    return {str(Path(p).name): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def score(item, response):
    text = response['response'].strip()
    words = text.lower().split()
    grams = [tuple(words[i:i+4]) for i in range(max(0, len(words)-3))]
    repetition = 1 - len(set(grams))/len(grams) if grams else 0.
    result = {'relevance': None, 'fluency': None, 'coherence': None, 'correctness': None, 'instruction_following': None,
              'task_success': None, 'repeated_4gram_fraction': repetition,
              'human_review_required': True, 'execution': 'not_requested'}
    check = item.get('check')
    if check == 'exact':
        success = text in item['answers']
        result.update(correctness=success, instruction_following=success, task_success=success)
    elif check == 'python':
        code = extract_code(text)
        try:
            tree = ast.parse(code)
            result['python_parseable'] = True
            result['contains_for_loop'] = any(isinstance(n, ast.For) for n in ast.walk(tree))
        except SyntaxError:
            result['python_parseable'] = False
            result['contains_for_loop'] = False
        result['execution'] = 'pending_restricted_executor'
        # Parsing alone is deliberately not marked as correctness or success.
    elif check == 'tool':
        try:
            call = json.loads(text)
        except (ValueError, TypeError):
            call = None
        valid = call == item['expected_call']
        result.update(parseable_call=isinstance(call, dict), correct_call=valid,
                      correctness=valid, instruction_following=valid, task_success=valid,
                      execution='schema_check_only')
    return result


def extract_code(text):
    match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.S)
    return match.group(1) if match else text


def summarize(records):
    output = {}
    for category in sorted({r['category'] for r in records}):
        selected = [r for r in records if r['category'] == category]
        rows = [r['scores'] for r in selected]
        output[category] = {'count': len(rows), 'review_pending': sum(r['human_review_required'] for r in rows)}
        repetitions=[r['repeated_4gram_fraction'] for r in rows if r.get('repeated_4gram_fraction') is not None]
        output[category]['repetition']={'mean_repeated_4gram_fraction':sum(repetitions)/len(repetitions) if repetitions else None,'measured':len(repetitions)}
        lengths=[r['generated_tokens'] for r in selected if r.get('generated_tokens') is not None]
        output[category]['generation']={'truncated':sum(r.get('truncated') is True for r in selected),
            'truncation_measured':sum(type(r.get('truncated')) is bool for r in selected),
            'mean_generated_tokens':sum(lengths)/len(lengths) if lengths else None,
            'empty_responses':sum(not r.get('response','').strip() for r in selected)}
        for metric in ('fluency','coherence','relevance'):
            measured=[r[metric] for r in rows if r.get(metric) is not None]
            output[category][metric]={'mean':sum(measured)/len(measured) if measured else None,'measured':len(measured),'pending':len(rows)-len(measured)}
        for metric in ('correctness', 'instruction_following', 'task_success'):
            measured = [r[metric] for r in rows if r[metric] is not None]
            output[category][metric] = {'passed': sum(measured), 'measured': len(measured)}
    return output
