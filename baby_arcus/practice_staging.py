"""Review targets use the actual packed model context, never reconstructed teacher text."""
import json
from baby_arcus.contracts import digest


def examples(episode, events):
    records=[]
    for index,event in enumerate(events[:-1]):
        if event.get('kind')!='intent': continue
        decision=event['decision']; outcome=events[index+1]
        if (not decision.get('call') or not decision.get('input_messages') or outcome.get('kind')!='outcome'
                or outcome['result'].get('status') in ('tool_error','cancelled')):
            continue
        messages=[dict(item,train=False) if item['role']=='assistant' else dict(item)
                  for item in decision['input_messages']]
        messages.append({'role':'assistant','content':json.dumps(decision['call'],sort_keys=True)})
        records.append({'version':1,'source':'arcus:practice:'+episode+':'+digest([event,outcome]),
                        'group':'arcus-practice:'+episode,'split':'training','messages':messages})
    return records
