"""Pixel-only palette baseline. Not a learned detector or object identity tracker."""
from io import BytesIO
import hashlib

PALETTE = {'blue': (85,141,198), 'yellow': (214,184,68), 'purple': (153,107,180)}

def regions(mask,minimum=8):
    """Spatial components of a predicted mask; never simulator identities."""
    import numpy as np
    ys,xs=np.nonzero(mask);remaining=set(zip(xs.tolist(),ys.tolist()));result=[]
    while remaining:
        start=min(remaining);remaining.remove(start);stack=[start];points=[]
        while stack:
            x,y=stack.pop();points.append((x,y))
            for p in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if p in remaining:remaining.remove(p);stack.append(p)
        if len(points)>=minimum:result.append(points)
    return result

def learned_observation(pixels,labels):
    import numpy as np
    objects=[]
    for points in regions(labels==3):
        xs,ys=zip(*points)
        objects.append({'bbox':[min(xs),min(ys),max(xs)+1,max(ys)+1],
            'visible_pixels':len(points),'mean_rgb':pixels[:,ys,xs].mean(axis=1).tolist()})
    surfaces={name:(pixels[:,labels==i].mean(axis=1).tolist() if np.any(labels==i) else None)
              for name,i in (('floor',1),('wall',2))}
    return {'schema':'arcus-learned-object-observation-v1','size':[96,96],
            'objects':objects,'surfaces':surfaces,'method':'learned segmentation; RGB pooled over predicted regions',
            'identity_tracking':False}


def detect(frame):
    """Return visible connected regions in crop coordinates, never world metadata."""
    from PIL import Image
    with Image.open(BytesIO(frame)) as source:
        image = source.convert('RGB')
    width,height=image.size
    if width*height>800*560:
        raise ValueError('Observation exceeds playpen camera size')
    pixels=image.load(); detections=[]
    for label,color in PALETTE.items():
        remaining={(x,y) for y in range(height) for x in range(width) if pixels[x,y]==color}
        while remaining:
            start=min(remaining); remaining.remove(start); stack=[start]; component=[]
            while stack:
                x,y=stack.pop();component.append((x,y))
                for neighbor in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                    if neighbor in remaining:
                        remaining.remove(neighbor);stack.append(neighbor)
            if len(component)<12:continue
            xs,ys=zip(*component);box=[min(xs),min(ys),max(xs)+1,max(ys)+1]
            detections.append({'appearance':label,'bbox':box,'visible_pixels':len(component),
                'center':[sum(xs)/len(xs),sum(ys)/len(ys)],
                'touches_crop_edge':box[0]==0 or box[1]==0 or box[2]==width or box[3]==height})
    return {'schema':'arcus-object-pixels-v1','frame_sha256':hashlib.sha256(frame).hexdigest(),
            'size':[width,height],'detections':detections,
            'method':'exact palette connected components; confidence and identity not established'}


def observe(world):
    from baby_arcus.curiosity_environment import observe as require_available
    from baby_arcus.gaze import crop_frame
    from baby_arcus.playpen_capture import capture_playpen
    require_available(world)
    state=world.snapshot()
    frame=crop_frame(capture_playpen(state),state['arcus'])
    return frame,detect(frame['bytes'])
