"""Sitting lesson: extend the front legs, tuck the rear legs, hold support."""
import random
from baby_arcus.standing_environment import StandingEnvironment
from baby_arcus.body_dynamics import JOINTS
from baby_arcus.body_vocabulary import ACTIONS
from baby_arcus.posture_goals import achieved

def error(senses):
    return sum(abs(v-(1 if k.startswith("front") else 0)) for k,v in senses["joint_positions"].items())/12

class SittingEnvironment(StandingEnvironment):
    def __init__(self,seed=0,limit=240,initial_joints=None,hold_ticks=50):
        rng=random.Random(seed)
        mode=seed%3
        joints=initial_joints if initial_joints is not None else {
            k:rng.uniform(*((.88,1) if mode==0 else (0,.12) if mode==1 else (.25,.75))) for k in JOINTS}
        super().__init__(seed,limit,joints,hold_ticks)
        for _ in range(12):self.session.step()
    def step(self,index):
        if type(index) is not int or not 0<=index<len(ACTIONS):raise ValueError("Invalid posture action")
        if self.steps>=self.limit or self.success:raise ValueError("Episode finished")
        before=self.observe()
        if ACTIONS[index]:self.session.action(ACTIONS[index])
        self.session.step();self.steps+=1
        after=self.observe()
        self.hold=self.hold+1 if achieved(after,"sitting") else 0
        self.success=self.hold>=self.hold_ticks
        reward=4*(error(before)-error(after))-.01
        if not after["stable"]:reward-=.01
        if self.success:reward+=1
        return after,reward,self.success or self.steps>=self.limit
