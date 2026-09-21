"""Pixels and proprioception into the frozen Arcus MoDE trunk, then gaze actions."""
import base64
from io import BytesIO
import hashlib
import torch
from torch import nn
from baby_arcus.depth_policy import capacity
from baby_arcus.visual_experience import NAMES

def tensors(record):
    import numpy as np
    from PIL import Image
    raw=base64.b64decode(record['frame']['image_base64'],validate=True)
    if len(raw)>500000 or hashlib.sha256(raw).hexdigest()!=record['frame']['sha256']:
        raise ValueError('Invalid frame identity or size')
    if record['frame']['source']!='playpen':raise ValueError('Visual lesson accepts playpen frames only')
    if raw:
        with Image.open(BytesIO(raw)) as image:
            if image.width>800 or image.height>560:raise ValueError('Unexpected visual dimensions')
            array=np.array(image.convert('RGB').resize((96,96)),dtype='float32')/255
        pixels=torch.from_numpy(array).permute(2,0,1)
    else:pixels=torch.zeros(3,96,96)
    g=record['gaze']
    state=torch.tensor([g[k] for k in ('head_yaw','head_pitch','eye_yaw','eye_pitch','eyelid_openness')]+
        [record['senses']['height'],record['resources']['remaining_fraction']],dtype=torch.float32)
    if not torch.isfinite(state).all():raise ValueError('Nonfinite visual state')
    # The local navigation camera does not move with the independent gaze camera.
    if record.get('lesson')=='navigation':state[:4]=0
    return pixels,state

class VisualAdapter(nn.Module):
    def __init__(self,dim,actions=len(NAMES)):
        super().__init__()
        self.encoder=nn.Unfold(kernel_size=24,stride=24)
        self.project=nn.Linear(3*24*24,dim,bias=False)
        self.state=nn.Linear(7,dim)
        self.head=nn.Linear(17*dim,actions)
    def forward(self,core,pixels,state,budget=.5):
        patches=self.encoder((pixels-.5)*2).transpose(1,2)
        tokens=torch.cat((self.project(patches),self.state(state)[:,None]),dim=1)*.05
        with capacity(core,min(budget,core.cfg.capacity)):hidden=core.trunk_embedded(tokens)
        return self.head(hidden.flatten(1))


class ObjectPerceptionAdapter(nn.Module):
    """RGB patches through Arcus's core; spatial visible-surface output, no actor."""
    def __init__(self,dim):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.GELU(),
            nn.Conv2d(32,32,3,padding=1),nn.GELU(),nn.Unfold(kernel_size=16,stride=16))
        self.project=nn.Linear(32*16*16,dim)
        self.head=nn.Linear(dim,4*16*16)
    def forward(self,core,pixels):
        tokens=self.project(self.encoder(pixels*2-1).transpose(1,2))*.05
        with capacity(core,core.cfg.capacity):hidden=core.trunk_embedded(tokens)
        patches=self.head(hidden).transpose(1,2)
        return nn.functional.fold(patches,(96,96),kernel_size=16,stride=16)


class SpatialObjectPerceptionAdapter(nn.Module):
    """Local RGB detail with MoDE contextual features, jointly trained as one path."""
    def __init__(self,dim):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.GELU(),nn.Conv2d(32,32,3,padding=1),nn.GELU())
        self.project=nn.Linear(32,dim)
        self.context=nn.Linear(dim,16)
        self.head=nn.Sequential(nn.Conv2d(48,32,3,padding=1),nn.GELU(),nn.Conv2d(32,4,1))
    def forward(self,core,pixels):
        local=self.encoder(pixels*2-1)
        tokens=self.project(nn.functional.adaptive_avg_pool2d(local,(6,6)).flatten(2).transpose(1,2))*.05
        with capacity(core,core.cfg.capacity):hidden=core.trunk_embedded(tokens)
        context=self.context(hidden).transpose(1,2).reshape(-1,16,6,6)
        context=nn.functional.interpolate(context,(96,96),mode='bilinear',align_corners=False)
        return self.head(torch.cat((local,context),dim=1))


class MultiscaleObjectPerceptionAdapter(nn.Module):
    """Spatial decoder retains edges and shape; the same MoDE supplies context."""
    def __init__(self,dim):
        super().__init__()
        def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GELU(),nn.Conv2d(b,b,3,padding=1),nn.GELU())
        self.fine=block(3,24);self.mid=block(24,48);self.coarse=block(48,64)
        self.project=nn.Linear(64,dim);self.context=nn.Linear(dim,16)
        self.up_mid=block(128,48);self.up_fine=block(72,24);self.head=nn.Conv2d(24,4,1)
        nn.init.zeros_(self.context.weight);nn.init.zeros_(self.context.bias)
    def forward(self,core,pixels):
        fine=self.fine(pixels*2-1);mid=self.mid(nn.functional.avg_pool2d(fine,2))
        coarse=self.coarse(nn.functional.avg_pool2d(mid,2))
        from baby_arcus.shared_pooling import adaptive_pool
        tokens=self.project(adaptive_pool(coarse,(6,6)).flatten(2).transpose(1,2))*.05
        with capacity(core,core.cfg.capacity):hidden=core.trunk_embedded(tokens)
        context=self.context(hidden).transpose(1,2).reshape(-1,16,6,6)*.01
        context=nn.functional.interpolate(context,coarse.shape[-2:],mode='bilinear',align_corners=False)
        up=nn.functional.interpolate(torch.cat((coarse,context),1),mid.shape[-2:],mode='bilinear',align_corners=False)
        up=self.up_mid(torch.cat((up,mid),1))
        up=nn.functional.interpolate(up,fine.shape[-2:],mode='bilinear',align_corners=False)
        return self.head(self.up_fine(torch.cat((up,fine),1)))


class NavigationAdapter(VisualAdapter):
    """Learned colour features with spatial bins retained through the Arcus core."""
    def __init__(self,dim,actions=7):
        super().__init__(dim,actions)
        self.encoder=nn.Sequential(nn.Conv2d(3,16,1),nn.ReLU(),nn.Conv2d(16,8,1),nn.ReLU(),
                                  nn.AdaptiveAvgPool2d((8,8)),nn.Unfold(kernel_size=2,stride=2))
        self.project=nn.Linear(32,dim,bias=False)
        self.head=nn.Sequential(nn.Linear(17*dim,128),nn.ReLU(),nn.Linear(128,actions))


class GroundedNavigationAdapter(VisualAdapter):
    """A learned pixel detector preserves marker location before Arcus action learning."""
    def __init__(self,dim,actions=7):
        super().__init__(dim,actions)
        self.detector=nn.Sequential(nn.Conv2d(3,16,1),nn.ReLU(),nn.Conv2d(16,1,1))
        self.encoder=nn.Sequential(nn.AdaptiveAvgPool2d((8,8)),nn.Unfold(kernel_size=2,stride=2))
        self.project=nn.Linear(4,dim,bias=False)
        self.head=nn.Sequential(nn.Linear(17*dim,128),nn.ReLU(),nn.Linear(128,actions))

    def forward(self,core,pixels,state,budget=.5):
        mask=self.detector((pixels-.5)*2).sigmoid()
        patches=self.encoder(mask).transpose(1,2)
        tokens=torch.cat((self.project(patches),self.state(state)[:,None]),dim=1)*.05
        with capacity(core,min(budget,core.cfg.capacity)):hidden=core.trunk_embedded(tokens)
        return self.head(hidden.flatten(1))

