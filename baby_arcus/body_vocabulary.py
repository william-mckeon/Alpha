"""Separate embodied token/action schema; incompatible with grid checkpoints."""
from baby_arcus.body_dynamics import JOINTS
SCHEMA="arcus-standing-v1"
ACTIONS=[None]+[{"kind":"joint","joint":joint,"delta":delta} for joint in JOINTS for delta in (-.15,.15)]
def encode(senses):
    return [1]+[2+i*21+round(senses["joint_positions"][key]*20) for i,key in enumerate(JOINTS)]+[254+round(senses["height"]*20)]
def mask(senses):
    q=senses["joint_positions"]
    return [True]+[q[a["joint"]]>0 if a["delta"]<0 else q[a["joint"]]<1 for a in ACTIONS[1:]]

