"""Resettable closed-eye standing lesson using the same body dynamics as the UI."""
import random
from baby_arcus.embodiment import Embodiment
from baby_arcus.play_session import PlaySession
from baby_arcus.body_dynamics import JOINTS
from baby_arcus.body_senses import observe_body_senses
from baby_arcus.body_vocabulary import ACTIONS
class StandingEnvironment:
    def __init__(self,seed=0,limit=180,initial_joints=None,hold_ticks=10):
        rng=random.Random(seed)
        self.session=PlaySession(Embodiment(height=.25,target_posture="lying",motor_mode="independent",eyelid_openness=0))
        self.session.body.joint_positions={key:rng.uniform(0,.12) for key in JOINTS}
        if initial_joints is not None:
            from baby_arcus.contracts import fields
            from baby_arcus.body_dynamics import number
            fields(initial_joints,JOINTS)
            self.session.body.joint_positions={key:number(initial_joints[key],0,1) for key in JOINTS}
        self.session.body.previous_joints=dict(self.session.body.joint_positions)
        if type(hold_ticks) is not int or not 1<=hold_ticks<=600:
            raise ValueError("Invalid standing hold duration")
        self.hold_ticks=hold_ticks
        self.limit=limit;self.steps=0;self.hold=0;self.success=False
    def observe(self):
        return observe_body_senses(self.session.body)
    def step(self,index):
        if type(index) is not int or not 0<=index<len(ACTIONS): raise ValueError("Invalid standing action")
        if self.steps>=self.limit or self.success: raise ValueError("Episode finished")
        before=self.observe();q0=sum(before["joint_positions"].values())/12
        if ACTIONS[index]:self.session.action(ACTIONS[index])
        self.session.step();self.steps+=1
        after=self.observe();q1=sum(after["joint_positions"].values())/12
        self.hold=self.hold+1 if after["height"]>=.92 and after["stable"] else 0
        self.success=self.hold>=self.hold_ticks
        reward=4*(q1-q0)+2*(after["height"]-before["height"])-.01
        if not after["stable"]:reward-=.05
        if self.success:reward+=1
        return after,reward,self.success or self.steps>=self.limit
