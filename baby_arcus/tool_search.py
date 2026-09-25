"""Deterministic lexical discovery with bounded complete schemas, no LLM."""
import re
from baby_arcus.contracts import canonical


def search(catalog, query, limit=3, max_bytes=12000):
    if not isinstance(query, str) or not 1<=len(query)<=300 or type(limit) is not int or not 1<=limit<=5:
        raise ValueError('Invalid tool search query or limit')
    words = set(re.findall(r'[a-z0-9]+', query.lower().replace('_',' ')))
    ranked = []
    for name, tool in catalog.tools.items():
        if name == 'tool_search':
            continue
        terms = set(re.findall(r'[a-z0-9]+', (name+' '+tool['description']).lower().replace('_',' ')))
        score = len(words & terms)
        if score:
            ranked.append((-score, name, tool))
    results = []
    for _, _, tool in sorted(ranked)[:limit]:
        if len(canonical(results+[tool]))>max_bytes:
            break
        results.append(tool)
    return {'catalog_version': catalog.version, 'query': query, 'tools': results,
            'status': 'matches' if results else 'no_matches'}
