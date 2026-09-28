"""Separate parseable proposals from successful executor outcomes."""
import json


def summarize(events):
    intents=[e['decision'] for e in events if e.get('kind')=='intent']
    outcomes=[e['result'] for e in events if e.get('kind')=='outcome']
    fingerprints=[json.dumps(d.get('call') or d.get('text'),sort_keys=True) for d in intents]
    return {'parseable_calls':sum(d.get('status')=='call' for d in intents),
            'executed_calls':sum(d.get('status')=='call' and r.get('status') not in
                ('tool_error','cancelled','invalid_call') for d,r in zip(intents,outcomes)),
            'tool_errors':sum(r.get('status')=='tool_error' for r in outcomes),
            'repeated_decisions':sum(a==b for a,b in zip(fingerprints,fingerprints[1:])),
            'error_messages':[r.get('error','') for r in outcomes if r.get('status')=='tool_error']}
