"""Small real local tools; no shell, network, filesystem or Python evaluation."""
import math

SCHEMAS = [
    {'type':'function','function':{'name':'echo','description':'Return the supplied text unchanged.',
     'parameters':{'type':'object','properties':{'text':{'type':'string'}},'required':['text'],'additionalProperties':False}}},
    {'type':'function','function':{'name':'calculate','description':'Calculate one arithmetic operation.',
     'parameters':{'type':'object','properties':{'operation':{'type':'string','enum':['add','subtract','multiply','divide']},
        'a':{'type':'number'},'b':{'type':'number'}},'required':['operation','a','b'],'additionalProperties':False}}}
]


def dispatch(name, args):
    try:
        if not isinstance(args,dict): raise ValueError('Arguments must be an object')
        if name=='echo':
            if set(args)!={'text'} or not isinstance(args['text'],str) or len(args['text'])>1000:
                raise ValueError('echo requires text of at most 1000 characters')
            value=args['text']
        elif name=='calculate':
            if set(args)!={'operation','a','b'} or args['operation'] not in ('add','subtract','multiply','divide'):
                raise ValueError('Invalid calculate arguments')
            a,b=args['a'],args['b']
            if any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>1e6 for v in (a,b)):
                raise ValueError('Numbers must be finite and bounded by 1000000')
            op=args['operation']
            if op=='divide' and b==0: raise ValueError('Division by zero')
            value={'add':lambda:a+b,'subtract':lambda:a-b,'multiply':lambda:a*b,'divide':lambda:a/b}[op]()
        else:
            raise ValueError('Tool is not allowlisted')
        return {'ok':True,'result':value}
    except (ValueError,TypeError,OverflowError) as error:
        return {'ok':False,'error':str(error)}
