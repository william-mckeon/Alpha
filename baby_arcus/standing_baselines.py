"""Constructive and negative controls, never used as imitation targets."""
import random
from baby_arcus.body_vocabulary import JOINTS,mask
def scripted(senses):
    positions=senses["joint_positions"]
    if min(positions.values())>=1:return 0
    key=min(JOINTS,key=lambda k:positions[k])
    return 2+2*JOINTS.index(key)
def random_action(senses,rng):
    return rng.choice([i for i,allowed in enumerate(mask(senses)) if allowed])

