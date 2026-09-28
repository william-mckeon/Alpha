"""Token-weighted language metrics; no averaging perplexities across windows."""
import math


def domain_language_metrics(windows, expected_domains):
    groups = {name: [] for name in expected_domains}
    for row in windows:
        if row['domain'] not in groups:
            raise ValueError('Unexpected language domain')
        groups[row['domain']].append(row)
    return {'domains': {name: language_metrics(rows) if rows else None for name, rows in groups.items()},
            'overall': language_metrics(windows) if windows else None,
            'missing_domains': [name for name, rows in groups.items() if not rows],
            'weighting': 'scored target tokens; never mean perplexity',
            'tokenizer_comparison': 'Compare only identical tokenizer and held-out corpus identities.'}


def language_metrics(windows):
    total = 0
    nll_sum = 0.0
    for row in windows:
        count, nll = row['target_tokens'], row['nll']
        if type(count) is not int or count <= 0 or not math.isfinite(nll) or nll < 0:
            raise ValueError('Invalid scored window')
        total += count
        nll_sum += count * nll
    if not total:
        raise ValueError('No scored target tokens')
    mean = nll_sum / total
    return {'target_tokens': total, 'nll_sum': nll_sum, 'nll': mean,
            'perplexity': math.exp(mean) if mean < 709 else None,
            'perplexity_overflow': mean >= 709}
