"""Token-weighted language metrics; no averaging perplexities across windows."""
import math


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
