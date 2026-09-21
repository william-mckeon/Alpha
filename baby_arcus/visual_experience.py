"""Time-aligned, playpen-only visual experiences; no Torch in the native host."""
import base64
from copy import deepcopy
import hashlib
import time
import uuid
from baby_arcus.body_senses import observe_body_senses
from baby_arcus.contracts import ContractError
from baby_arcus.gaze import crop_frame
from baby_arcus.playpen_capture import capture_playpen

def available(world):
    return not world.paused and not world.view.held and world.view.region=='playpen' and world.body.sleep_state=='awake'

def observation(app,remaining=1.,navigation=False,perception=False):
    with app.lock:
        if not available(app.world):raise ContractError('Visual lesson requires an awake, unheld body in the active playpen')
        if perception and app.world.body.eyelid_openness<=0:raise ContractError('Perception requires open eyes')
        state=deepcopy(app.world.snapshot())
        senses=observe_body_senses(app.world.body,app.world.view.held)
        messages=deepcopy(app.conversation.snapshot(model=True)[-4:])
        session=app.session
    started=time.perf_counter();frame=None
    if state['arcus']['eyelid_openness']>0:
        if navigation:
            from baby_arcus.visual_navigation_environment import capture_navigation
            frame=capture_navigation(state)
        else:frame=crop_frame(capture_playpen(state),state['arcus'])
    raw=frame['bytes'] if frame else b''
    return {'schema':'arcus-visual-experience-v1','lesson':'perception' if perception else 'navigation' if navigation else 'gaze','id':uuid.uuid4().hex,'session':session,
        'tick':state['tick'],'entity_id':state['arcus']['entity_id'],
        'scope_id':state['view']['scope_id'],'epoch':state['view']['epoch'],
        'captured_at':time.time(),'senses':senses,
        'gaze':{k:state['arcus'][k] for k in ('head_yaw','head_pitch','eye_yaw','eye_pitch','eyelid_openness')},
        'messages':messages,'message_use':'recorded_context_only',
        'frame':{'source':'playpen','sha256':hashlib.sha256(raw).hexdigest(),
                 'image_base64':base64.b64encode(raw).decode(),'camera':frame['camera'] if frame else None},
        'resources':{'remaining_fraction':max(0,min(1,remaining)),
                     'capture_ms':1000*(time.perf_counter()-started),'frame_bytes':len(raw)},
        'model_inputs':['pixels','eyelid_openness','height','remaining_fraction'] if navigation else ['pixels','gaze','height','remaining_fraction']}

def still_current(app,record):
    w=app.world
    return (available(w) and app.session==record['session'] and w.body.entity_id==record['entity_id']
        and w.view.scope_id==record['scope_id'] and w.view.epoch==record['epoch']
        and w.body.joint_positions==record['senses']['joint_positions']
        and 0<=w.tick-record['tick']<=30 and 0<=time.time()-record['captured_at']<=3)

NAMES=('hold','left','right','up','down','open')

def motor_action(index,gaze):
    if type(index) is not int or not 0<=index<len(NAMES):raise ContractError('Invalid visual action')
    if index==0:return None
    if index==5:return {'kind':'eyelids','openness':1.}
    yaw=gaze['eye_yaw'];pitch=gaze['eye_pitch']
    yaw+=(-.25 if index==1 else .25 if index==2 else 0)
    pitch+=(-.25 if index==3 else .25 if index==4 else 0)
    return {'kind':'gaze','yaw':max(-1,min(1,yaw)),'pitch':max(-1,min(1,pitch))}
