"""Assistant-only supervision, with explicit role markers and bounded windows."""
from baby_arcus.sft_validation import validate


def validate_window(sequence, vocabulary, context=512):
    if not isinstance(sequence, dict) or set(sequence) != {'ids','mask'}:
        raise ValueError('Invalid SFT window')
    ids, mask = sequence['ids'], sequence['mask']
    if not isinstance(ids,list) or not 2 <= len(ids) <= context + 1:
        raise ValueError('SFT window exceeds context')
    if any(type(i) is not int or not 0 <= i < vocabulary for i in ids):
        raise ValueError('Invalid vocabulary token')
    if not isinstance(mask,list) or len(mask) != len(ids) or any(type(v) is not bool for v in mask) or not any(mask[1:]):
        raise ValueError('SFT target mask must contain assistant targets')


def windows(record, tokenizer, context=512):
    validate(record)
    if type(context) is not int or not 8 <= context <= 8192:
        raise ValueError('Context exceeds the experimental 8192-token packing limit')
    from baby_arcus.conversation_format import pack, content
    history = []
    for item in record['messages']:
        if item['role'] == 'assistant' and item.get('train', True):
            target = tokenizer.encode(content(item) + '\n</assistant>\n')
            if len(target) >= context:
                raise ValueError('Quarantine oversize assistant action; never split a tool call')
            _, prefix = pack(history, tokenizer, context + 1 - len(target))
            sequence = {'ids':prefix + target, 'mask':[False]*len(prefix) + [True]*len(target)}
            validate_window(sequence, tokenizer.vocab_size if hasattr(tokenizer,'vocab_size') else max(sequence['ids'])+1, context)
            yield sequence
        history.append(item)


def packing_report(records, tokenizer, context=512):
    """Preflight full examples before training; never partially admit an example."""
    from baby_arcus.contracts import digest
    result={'context_tokens':context,'accepted':[],'quarantined':[],'target_tokens':0}
    for record in records:
        identity=digest(record)
        try:
            count=0; length=0; window_count=0
            for window in windows(record,tokenizer,context):
                window_count+=1
                count+=sum(window['mask'][1:]); length=max(length,len(window['ids'])-1)
            if count==0: raise ValueError('No assistant targets')
            result['accepted'].append({'sha256':identity,'target_tokens':count,'max_input_tokens':length,'windows':window_count})
            result['target_tokens']+=count
        except ValueError as exc:
            result['quarantined'].append({'sha256':identity,'reason':str(exc)})
    return result
