"""Seeded RGB lessons. Oracle masks are labels only, never inference inputs."""
from copy import deepcopy
from io import BytesIO
import colorsys,random
import numpy as np
from PIL import Image
from baby_arcus.play_session import PlaySession
from baby_arcus.playpen_capture import capture_playpen
from baby_arcus.gaze import crop_frame

def arrays(raw):
    with Image.open(BytesIO(raw)) as image:
        return np.asarray(image.convert('RGB').resize((96,96)),dtype=np.float32).transpose(2,0,1)/255

def color(rng):
    return '#'+''.join(f'{round(v*255):02x}' for v in colorsys.hsv_to_rgb(rng.random(),rng.uniform(.35,.9),rng.uniform(.45,.95)))

def example(seed,index):
    rng=random.Random(seed+index*1009);w=PlaySession()
    colors={key:color(rng) for key in w.environment.colors}
    ball=color(rng);w.environment.color_lesson(colors,[ball,ball if index%4==0 else color(rng)])
    # Balance 0/1/2 objects. Include negative frames and same-color pairs.
    count=index%3
    for key in list(w.environment.objects)[count:]:
        del w.environment.objects[key];del w.environment._effects[key]
    placed=[]
    for obj in w.environment.objects.values():
        for _ in range(100):
            x,y=rng.uniform(1,9),rng.uniform(1,6)
            if all((x-a)**2+(y-b)**2>1 for a,b in placed):break
        obj.update(x=x,y=y);placed.append((x,y))
    w.body.eye_yaw=rng.uniform(-1,1);w.body.eye_pitch=rng.uniform(-1,1)
    w.environment.human.update(x=rng.uniform(1,9),y=rng.uniform(1,6),present=index%4==1)
    state=w.snapshot();frame=crop_frame(capture_playpen(state),state['arcus'])
    oracle=deepcopy(state)
    oracle['environment']['colors']={'floor':'#0000ff','rug':'#0000ff','wall':'#00ff00'}
    for obj in oracle['environment']['objects'].values():obj['color']='#ff00ff'
    mask=crop_frame(capture_playpen(oracle),oracle['arcus'])
    with Image.open(BytesIO(mask['bytes'])) as im:
        rgb=np.asarray(im.resize((96,96),Image.Resampling.NEAREST))
    labels=np.zeros((96,96),dtype=np.int64)
    labels[np.all(rgb==[0,0,255],axis=-1)]=1
    labels[np.all(rgb==[0,255,0],axis=-1)]=2
    labels[np.all(rgb==[255,0,255],axis=-1)]=3
    return arrays(frame['bytes']),labels,{'seed':seed,'index':index,'same_color':index%4==0}
