"""Matched pilot gates; a complete evaluation can still fail capability gates."""
from baby_arcus.evaluation_metrics import language_metrics


def scores(report):
    coding=report['coding'];tasks=coding['tasks']
    decisions=[d for t in tasks for d in t['decisions']]
    return {'language':language_metrics(report['sft']),
            'coding_solved':coding['solved'],'coding_total':coding['total'],
            'discovery_solved':sum(t['solved'] for t in report['unseen_tools']),
            'valid_call_rate':sum(d['status']=='call' for d in decisions)/max(1,len(decisions)),
            'executed_call_rate':sum(t.get('executed_calls',0) for t in tasks)/max(1,len(decisions)),
            'tool_errors':sum(t.get('tool_errors',0) for t in tasks),
            'repeated_decisions':sum(t.get('repeated_decisions',0) for t in tasks),
            'external_transcript_rate':sum(t['external_transcript_decisions'] for t in tasks)/max(1,len(decisions))}


def build(baseline, candidate, gates):
    for report in (baseline,candidate):
        if not report.get('complete') or not report.get('checkpoint_unchanged') or not report.get('coding_execution_evaluated'):
            raise ValueError('Incomplete evaluation')
        if (report['coding'].get('candidate')!=report['candidate'] or not report['coding'].get('checkpoint_unchanged')
                or not report['coding'].get('complete')):
            raise ValueError('Coding checkpoint evidence mismatch')
    start=baseline['candidate']['updates']
    ceiling={40000:41000,41000:60000}.get(start,0)
    if not start<candidate['candidate']['updates']<=ceiling:
        raise ValueError('Comparison outside correction lineage budget')
    if baseline['evaluation_identity']!=candidate['evaluation_identity']:raise ValueError('Unmatched evaluation cohorts')
    before,after=scores(baseline),scores(candidate);limits=gates['tool_correction']
    passed={'valid_calls':after['valid_call_rate']>=limits['min_valid_call_rate'],
            'discovery':after['discovery_solved']>=limits['min_discovery_solved'],
            'no_external_transcripts':after['external_transcript_rate']<=limits['max_external_transcript_rate'],
            'language_retention':after['language']['nll']-before['language']['nll']<=limits['max_nll_increase'],
            'coding_gain':after['coding_solved']-before['coding_solved']>=gates['capability_thresholds']['min_coding_gain'],
            'coding_minimum':after['coding_solved']>=gates['capability_thresholds']['min_coding_solved']}
    return {'baseline':baseline['candidate'],'candidate':candidate['candidate'],'before':before,'after':after,
            'gates':passed,'accepted':all(passed.values()),'automatic_promotion':False,
            'resources':{'baseline_seconds':baseline.get('seconds'),'candidate_seconds':candidate.get('seconds'),
                         'candidate_peak_cuda_allocated_bytes':candidate.get('peak_cuda_allocated_bytes')},
            'limitations':'Small single-seed repeated cohorts; does not establish general coding mastery or 16k competence.'}


def attach_developmental(report, developmental):
    """Link a separate diagnostic without changing the frozen task gates."""
    if developmental.get('candidate') != report['candidate'] or not developmental.get('checkpoint_unchanged'):
        raise ValueError('Developmental checkpoint mismatch')
    return {**report, 'developmental': {'identity':developmental['identity'], 'summary':developmental['summary'],
                                      'separate_from_retention_gate':True}}
