"""Deterministic normalized-joint support model, not robotics-grade rigid-body physics."""
import math
from baby_arcus.contracts import ContractError, fields
LEGS = ("front_left", "front_right", "rear_left", "rear_right")
JOINTS = tuple(f"{leg}.{joint}" for leg in LEGS for joint in ("hip", "knee", "ankle"))
AXES = ("head_yaw", "head_pitch", "eye_yaw", "eye_pitch")
def number(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ContractError("Motor value outside finite bounds")
    return float(value)
def pose(extension):
    return {key: extension for key in JOINTS}
def validate_motor(action):
    kind = action.get("kind")
    if kind == "joint":
        fields(action, ("kind", "joint", "delta"))
        if action["joint"] not in JOINTS: raise ContractError("Unknown joint")
        number(action["delta"], -.15, .15)
    elif kind in ("head", "gaze"):
        fields(action, ("kind", "yaw", "pitch"))
        number(action["yaw"], -1, 1); number(action["pitch"], -1, 1)
    elif kind == "eyelids":
        fields(action, ("kind", "openness")); number(action["openness"], 0, 1)
    else: raise ContractError("Unknown motor action")
def sensations(body, held=False):
    lengths = {leg: sum(body.joint_positions[f"{leg}.{j}"] for j in ("hip","knee","ankle"))/3 for leg in LEGS}
    roll = (lengths["front_left"]+lengths["rear_left"]-lengths["front_right"]-lengths["rear_right"])/2
    pitch = (lengths["front_left"]+lengths["front_right"]-lengths["rear_left"]-lengths["rear_right"])/2
    spread = max(lengths.values())-min(lengths.values())
    contacts = {leg: not held and value >= max(lengths.values())-.12 for leg,value in lengths.items()}
    # In a seated pose the hindquarters support the tucked rear legs. This is
    # derived from joint geometry, never the requested task or an animation.
    haunch_contact=not held and all(body.joint_positions[k]<=.15 for k in JOINTS if k.startswith("rear"))
    seated_support=(haunch_contact and min(lengths["front_left"],lengths["front_right"])>=.7
                    and abs(lengths["front_left"]-lengths["front_right"])<=.12)
    stable = not held and (spread <= .22 or seated_support)
    return {"height":body.height,"tilt":{"roll":roll,"pitch":pitch}, "contacts":contacts,
            "stable":stable,"held":held,"supported":not held,"haunch_contact":haunch_contact,
            "joint_positions":dict(body.joint_positions),"joint_velocities":dict(body.joint_velocities),
            "fallen":not held and spread > .45 and not seated_support}
def step(body, held=False):
    if body.motor_mode == "assisted" and not held:
        target = 1 if body.target_posture == "standing" else 0
        for key,value in dict(body.joint_positions).items():
            body.joint_positions[key] = round(value+max(-.1,min(.1,target-value)), 6)
    body.joint_velocities = {key:round((body.joint_positions[key]-body.previous_joints[key])/.1,6) for key in JOINTS}
    body.previous_joints = dict(body.joint_positions)
    senses = sensations(body,held)
    if not held:
        extension = sum(body.joint_positions.values())/12
        target = .25+.75*extension if senses["stable"] else .25
        body.height = round(body.height+max(-.075,min(.075,target-body.height)),6)
    if body.sleep_state == "waking_up":
        body.sleep_state = "awake"
