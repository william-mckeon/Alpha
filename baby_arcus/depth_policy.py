"""Experimental interaction-conditioned compute allocator; never changes model size.

Learns quality regret from measured counterfactual capacities. Calibration chooses
a common compute price, rather than assigning a fixed depth to any task by name.
"""
from contextlib import contextmanager
import math
import torch
from torch import nn

TASKS=('standing','lying','sitting','approach','language')
CAPACITIES=(.125,.25,.5,1.0)


@contextmanager
def capacity(core,value):
    if not 0<value<=1:raise ValueError('Capacity must be in (0,1]')
    previous=[block.capacity for block in core.blocks]
    try:
        for block in core.blocks:block.capacity=float(value)
        yield
    finally:
        for block,old in zip(core.blocks,previous):block.capacity=old


def features(task,senses=None,relative=None,ids=None):
    if task not in TASKS:raise ValueError('Unknown interaction task')
    values=[float(task==name) for name in TASKS]
    joints=list(senses['joint_positions'].values()) if senses else [0.0]
    relative=relative or [0.0,0.0];ids=ids or []
    return values+[float(senses.get('height',0)) if senses else 0.0,
        max(joints)-min(joints),min(1,math.hypot(*relative)/12),
        min(1,len(ids)/128),len(set(ids))/max(1,len(ids))]


def upper_cost(length):
    if length<1:raise ValueError('Positive context length required')
    return [math.ceil(value*length)/length for value in CAPACITIES]


class DepthPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(10,32),nn.Tanh(),nn.Linear(32,len(CAPACITIES)))
        self.register_buffer('compute_price',torch.tensor(0.0))

    def forward(self,observation):return self.net(observation)

    def choose(self,observation,length):
        device=self.compute_price.device
        with torch.no_grad():
            scores=self(torch.tensor(observation,device=device,dtype=torch.float32))
            scores=scores+self.compute_price*torch.tensor(upper_cost(length),device=device)
            return CAPACITIES[int(scores.argmin())]


def fit(policy,rows,target=.25,steps=600):
    if not rows:raise ValueError('No capacity calibration records')
    device=policy.compute_price.device
    x=torch.tensor([r['features'] for r in rows],device=device)
    losses=torch.tensor([r['losses'] for r in rows],device=device)
    regret=losses-losses.min(dim=1,keepdim=True).values
    if not torch.isfinite(regret).all():raise ValueError('Invalid capacity measurements')
    optimizer=torch.optim.AdamW(policy.parameters(),lr=.005,weight_decay=0)
    for _ in range(steps):
        loss=(policy(x)-regret).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    lengths=torch.tensor([r['length'] for r in rows],device=device)
    costs=torch.tensor([upper_cost(r['length']) for r in rows],device=device)
    with torch.no_grad():predicted=policy(x)
    def usage(price):
        index=(predicted+price*costs).argmin(dim=1)
        return float((costs.gather(1,index[:,None]).squeeze(1)*lengths).sum()/lengths.sum())
    low,high=0.0,1.0
    while usage(high)>target and high<1e6:high*=2
    if usage(high)>target:raise ValueError('Requested budget below minimum available capacity')
    if usage(0)<=target:high=0.0
    else:
        for _ in range(40):
            mid=(low+high)/2
            if usage(mid)>target:low=mid
            else:high=mid
    policy.compute_price.fill_(high)
    return {'target_token_weighted_fraction':target,'calibration_upper_fraction':usage(high),
            'compute_price':high,'regret_fit_mse':float(loss.detach()),'examples':len(rows)}


def routing(core):
    return [{'layer':index,'depth_fraction':float(block.last_compute_fraction),
             'expert_fraction':block.last_expert_fraction.detach().cpu().tolist(),
             'expert_overflow':float(block.last_expert_overflow)}
            for index,block in enumerate(core.blocks)]


class MeasuredBudgetPolicy(nn.Module):
    """Separate experimental visual allocator, trained on matched outcome measurements.

    Scores are learned eligibility estimates, not calibrated probabilities of success.
    Deployment requires independent quality and measured-latency qualification.
    """
    def __init__(self,inputs=115):
        super().__init__()
        self.capacities=tuple(round(.95-.05*i,2) for i in range(15))
        self.net=nn.Sequential(nn.Linear(inputs,64),nn.Tanh(),nn.Linear(64,len(self.capacities)))
        self.register_buffer('milliseconds',torch.ones(len(self.capacities)))

    def forward(self,x):return self.net(x)

    def choose_index(self,x):
        eligible=self(x).sigmoid()>=.9
        costs=self.milliseconds.expand_as(eligible).masked_fill(~eligible,float('inf'))
        chosen=costs.argmin(-1)
        return torch.where(eligible.any(-1),chosen,torch.zeros_like(chosen))


def visual_budget_features(pixels,state):
    from torch.nn import functional as F
    return torch.cat((F.adaptive_avg_pool2d(pixels,(6,6)).flatten(1),state),dim=1)
