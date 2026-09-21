"""Local, image-driven translation lessons; this is not a joint-powered gait simulator."""
import base64
import hashlib
from io import BytesIO
import math
import random
from baby_arcus.contracts import ContractError
from baby_arcus.play_session import PlaySession
from baby_arcus.playpen_capture import capture_playpen

NAMES = ('arrived', 'left', 'right', 'up', 'down', 'open', 'target_missing')
ARRIVAL_RADIUS = 1.25


def capture_navigation(snapshot):
    """Body-centred camera with the same room contents; no target coordinates encoded."""
    from PIL import Image
    # A body-mounted sensor does not view the external observer's dragon sprite.
    frame = capture_playpen(snapshot,include_body=False)
    env = snapshot['environment']; body = snapshot['arcus']
    p = env['placements'][body['entity_id']]
    cx = p['x']/env['width']*770+15
    cy = p['y']/env['height']*530+15
    # A stable local navigation camera; gaze camera remains the separate looking lesson.
    box = (round(cx-200), round(cy-140), round(cx+200), round(cy+140))
    with Image.open(BytesIO(frame['bytes'])) as source:
        image = source.crop(box); output = BytesIO(); image.save(output, 'PNG')
    return {'bytes': output.getvalue(), 'camera': {'mapping': 'body-centred-local-v1',
            'source_size': [800,560]}, 'mime_type': 'image/png'}


def record(world, remaining=1.):
    snapshot = world.snapshot()
    raw = capture_navigation(snapshot)['bytes'] if world.body.eyelid_openness else b''
    return {'lesson':'navigation','frame': {'source': 'playpen', 'image_base64': base64.b64encode(raw).decode(),
                     'sha256': hashlib.sha256(raw).hexdigest()},
            'gaze': {k: snapshot['arcus'][k] for k in ('head_yaw','head_pitch','eye_yaw','eye_pitch','eyelid_openness')},
            'senses': {'height': world.body.height}, 'resources': {'remaining_fraction': remaining}}


def target_visible(item):
    """Teacher/evaluator only; never called by the model or runtime action selector."""
    from PIL import Image
    import numpy as np
    if not item['frame']['image_base64']: return False
    with Image.open(BytesIO(base64.b64decode(item['frame']['image_base64']))) as image:
        return int((np.asarray(image.convert('RGB'))==(227,108,60)).all(axis=2).sum()) >= 30


def distance(world):
    p=world.environment.placements[world.body.entity_id]; h=world.environment.human
    return math.hypot(h['x']-p['x'],h['y']-p['y'])


def label(world, item):
    if not world.body.eyelid_openness: return 5
    if not target_visible(item): return 6
    p=world.environment.placements[world.body.entity_id]; h=world.environment.human
    dx,dy=h['x']-p['x'],h['y']-p['y']
    if math.hypot(dx,dy)<=ARRIVAL_RADIUS:return 0
    if abs(dx)>abs(dy):return 1 if dx<0 else 2
    return 3 if dy<0 else 4


def action(index):
    if type(index) is not int or not 0<=index<len(NAMES):raise ContractError('Invalid navigation action')
    if index in (0,6):return None
    if index==5:return {'kind':'eyelids','openness':1.}
    return {'kind':'move','direction':NAMES[index]}


def scenes(seed,count):
    rng=random.Random(seed);counts=[0]*7;items=[];limit=math.ceil(count/7)
    attempts=0
    while len(items)<count:
        attempts+=1
        if attempts>count*100:raise RuntimeError(f'Unbalanced navigation scene generator: {counts}')
        world=PlaySession();p=world.environment.placements[world.body.entity_id]
        p.update(x=rng.uniform(.5,9.5),y=rng.uniform(.5,6.5))
        world.body.facing=rng.choice(('up','down','left','right'))
        angle=rng.uniform(-math.pi,math.pi);radius=rng.uniform(.3,2.7)
        h=world.environment.human
        h.update(x=max(.5,min(9.5,p['x']+math.cos(angle)*radius)),
                 y=max(.5,min(6.5,p['y']+math.sin(angle)*radius)),present=rng.random()>.12)
        world.body.eyelid_openness=0 if rng.random()<.15 else 1
        item=record(world,rng.uniform(.1,1));target=label(world,item)
        if counts[target]>=limit:continue
        counts[target]+=1;items.append((item,target))
    return items
