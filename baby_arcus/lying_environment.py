"""Reward-only controlled lowering lesson; collapse cannot satisfy the goal."""
import random
from baby_arcus.standing_environment import StandingEnvironment
from baby_arcus.body_dynamics import JOINTS
from baby_arcus.body_vocabulary import ACTIONS
from baby_arcus.posture_goals import achieved

class LyingEnvironment(StandingEnvironment):
    def __init__(self,seed=0,limit=240,initial_joints=None,hold_ticks=50):
        rng=random.Random(seed)
        joints=initial_joints if initial_joints is not None else {key:rng.uniform(.88,1) for key in JOINTS}
        super().__init__(seed,limit,joints,hold_ticks)
        self.session.body.target_posture="standing"
        for _ in range(12):self.session.step()
    def step(self,index):
        if type(index) is not int or not 0<=index<len(ACTIONS):raise ValueError("Invalid posture action")
        if self.steps>=self.limit or self.success:raise ValueError("Episode finished")
        before=self.observe();q0=sum(before["joint_positions"].values())/12
        if ACTIONS[index]:self.session.action(ACTIONS[index])
        self.session.step();self.steps+=1
        after=self.observe();q1=sum(after["joint_positions"].values())/12
        self.hold=self.hold+1 if achieved(after,"lying") else 0
        self.success=self.hold>=self.hold_ticks
        # Height can lag a joint action: rewarding its later drop can accidentally
        # credit an extension and teach oscillation. Use joint progress directly.
        def spread(senses):
            q=senses["joint_positions"]
            lengths=[sum(q[f"{leg}.{j}"] for j in ("hip","knee","ankle"))/3
                     for leg in ("front_left","front_right","rear_left","rear_right")]
            return max(lengths)-min(lengths)
        reward=4*(q0-q1)+.25*(spread(before)-spread(after))-.01
        if not after["stable"]:reward-=.01
        if self.success:reward+=1
        return after,reward,self.success or self.steps>=self.limit
