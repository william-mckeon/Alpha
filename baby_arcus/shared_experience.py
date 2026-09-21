"""One permitted, time-aligned sensory record. No Torch in the host."""
import base64,hashlib,time,uuid,math
from copy import deepcopy
from baby_arcus.body_senses import observe_body_senses
from baby_arcus.rest_environment import features
from baby_arcus.contracts import ContractError

def capture(app):
    from baby_arcus.playpen_capture import capture_playpen
    from baby_arcus.gaze import crop_frame
    with app.lock:
        w=app.world;state=deepcopy(w.snapshot())
        if w.paused or w.view.held or w.view.region!='playpen':raise ContractError('Shared observation requires active unheld playpen')
        senses=observe_body_senses(w.body,w.view.held)
        app.conversation.release(w.body.sleep_state=='awake')
        messages=deepcopy([m for m in app.conversation.snapshot(model=True) if not m.get('model_read')][-4:])
        events=app.interactions.pending(limit=4)
        for event in events:
            if event['kind']=='hearing_task':messages.append({'id':'interaction:'+event['request_id'],'text':event['payload']['text'],'source':'simulated_hearing_task','created_at':event['created_at']})
            if event['kind']=='call':messages.append({'id':'interaction:'+event['request_id'],'text':event['payload'].get('hearing',{}).get('utterance','Come here, Arcus'),'source':'simulated_hearing','created_at':event['created_at']})
            if event['kind']=='task':
                text={'standing':'stand up','lying':'lie down','sitting':'sit down','approach':'come here'}.get(event['payload'].get('goal'))
                if text:messages.append({'id':'interaction:'+event['request_id'],'text':text,'source':'simulated_hearing_task','created_at':event['created_at']})
        objects=[]
        if w.body.sleep_state=='awake' and w.body.eyelid_openness>0:
            from baby_arcus.curiosity_environment import observe
            objects=observe(w)
            position=w.environment.placements[w.body.entity_id]
            for obj in objects:
                target=w.environment.objects[obj['id']]
                obj['relative']=[target['x']-position['x'],target['y']-position['y']]
        position=w.environment.placements[w.body.entity_id]
        hearing_relative=[w.environment.human['x']-position['x'],w.environment.human['y']-position['y']]
        session=app.session;internal=features(w.body)
    visible=state['arcus']['sleep_state']=='awake' and state['arcus']['eyelid_openness']>0
    raw=crop_frame(capture_playpen(state),state['arcus'])['bytes'] if visible else b''
    return {'schema':'arcus-shared-experience-v1','id':uuid.uuid4().hex,'session':session,
        'entity_id':state['arcus']['entity_id'],'scope_id':state['view']['scope_id'],'environment_id':state['environment']['environment_id'],
        'epoch':state['view']['epoch'],'tick':state['tick'],'captured_at':time.time(),
        'senses':senses,'internal':internal,'hearing':messages,'events':events,
        'gaze':[state['arcus'][key] for key in ('head_yaw','head_pitch','eye_yaw','eye_pitch')],
        'hearing_relative':hearing_relative,'hearing_location_source':'simulated caregiver location; not microphone audio',
        'objects':objects,'object_source':'legacy_symbolic_simulator_observation','text_source':'caregiver',
        'vision':{'available':visible,'source':'playpen','sha256':hashlib.sha256(raw).hexdigest(),
                  'image_base64':base64.b64encode(raw).decode()},
        'eligibility':{'observed':True,'executed':False,'training':False}}

def validate(row):
    if row.get('schema')!='arcus-shared-experience-v1':raise ContractError('Wrong shared experience schema')
    def finite(value):
        if isinstance(value,float) and not math.isfinite(value):raise ContractError('Nonfinite shared observation')
        if isinstance(value,dict):
            for item in value.values():finite(item)
        elif isinstance(value,list):
            for item in value:finite(item)
    finite(row)
    def vector(values,length):
        if not isinstance(values,list) or len(values)!=length or any(type(v) not in (int,float) for v in values):
            raise ContractError('Invalid shared identity features')
    for key,length in (('search_query',13),('identity_context',7)):
        if key in row:vector(row[key],length)
    if 'identity_pair' in row:
        if not isinstance(row['identity_pair'],list) or len(row['identity_pair'])!=2:
            raise ContractError('Invalid shared identity pair')
        for values in row['identity_pair']:vector(values,11)
    memories=row.get('memory',[])
    if not isinstance(memories,list) or len(memories)>8:raise ContractError('Invalid shared memory budget')
    for memory in memories:
        values=memory.get('features') if isinstance(memory,dict) else None
        if not isinstance(values,list) or len(values)!=72 or any(type(value) not in (int,float) for value in values):
            raise ContractError('Invalid shared memory features')
    if len(row.get('gaze',[0,0,0,0]))!=4:raise ContractError('Invalid gaze senses')
    history=row.get('history',[])
    if len(history)>2:raise ContractError('Shared history limit exceeded')
    last_tick=-1
    for prior in history:
        if any(prior[key]!=row[key] for key in ('session','entity_id','scope_id','epoch')):raise ContractError('History crossed sensory scope')
        if not last_tick<=prior['tick']<=row['tick']:raise ContractError('Unordered sensory history')
        last_tick=prior['tick']
    if row['vision']['source']!='playpen':raise ContractError('Invalid shared visual source')
    raw=base64.b64decode(row['vision']['image_base64'],validate=True)
    if len(raw)>500000 or hashlib.sha256(raw).hexdigest()!=row['vision']['sha256']:raise ContractError('Invalid shared frame')
    if bool(raw)!=row['vision']['available']:raise ContractError('Missing vision mask mismatch')
    if raw and (row['senses']['sleep_state']!='awake' or row['senses']['eyelid_openness']<=0):raise ContractError('Vision unavailable')
    if len(row['internal'])!=5:raise ContractError('Invalid internal senses')
    return raw

def current(app,row):
    w=app.world
    return (not w.paused and not w.view.held and w.view.region=='playpen'
        and app.session==row['session'] and w.body.entity_id==row['entity_id']
        and w.view.scope_id==row['scope_id'] and w.view.epoch==row['epoch']
        and w.body.joint_positions==row['senses']['joint_positions']
        and 0<=w.tick-row['tick']<=30 and 0<=time.time()-row['captured_at']<=3)
