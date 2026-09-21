"""Cardinal heading semantics for bounded 2D translation, independent of joint gait."""
from baby_arcus.contracts import ContractError,fields

HEADINGS=('up','right','down','left')

def resolve(action,facing):
    if facing not in HEADINGS:raise ContractError('Unknown body heading')
    if action.get('kind')=='turn':
        fields(action,('kind','direction'))
        if action['direction'] not in ('left','right'):raise ContractError('Unknown turn')
        return None,HEADINGS[(HEADINGS.index(facing)+(1 if action['direction']=='right' else -1))%4]
    fields(action,('kind','direction'))
    if action.get('kind')!='step' or action['direction'] not in ('forward','backward'):raise ContractError('Unknown relative step')
    direction=facing if action['direction']=='forward' else HEADINGS[(HEADINGS.index(facing)+2)%4]
    return {'kind':'move','direction':direction},facing
