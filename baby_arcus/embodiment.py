"""Versioned identity and independent normalized joints, separate from placement."""
from dataclasses import dataclass, asdict, field
import uuid
from baby_arcus.contracts import ContractError, fields, identifier
from baby_arcus.body_dynamics import JOINTS, AXES, pose, number, step, LEGS

@dataclass
class Embodiment:
    entity_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "Arcus Alpha"
    facing: str = "right"
    height: float = 1.0
    target_posture: str = "standing"
    radius: float = .42
    sleep_state: str = "awake"
    joint_positions: dict = field(default_factory=lambda: pose(1))
    joint_velocities: dict = field(default_factory=lambda: pose(0))
    previous_joints: dict = field(default_factory=lambda: pose(1))
    motor_mode: str = "assisted"
    head_yaw: float = 0
    head_pitch: float = 0
    eye_yaw: float = 0
    eye_pitch: float = 0
    eyelid_openness: float = 1
    rest_need: float = .2
    stimulation: float = 0
    alertness: float = .8
    rest_mode: str = 'active'

    def record(self):
        return {"version": 3, **asdict(self)}

    @classmethod
    def restore(cls, record):
        fields(record, ("version", "entity_id", "name", "facing", "height", "target_posture", "radius"),
               ("sleep_state","joint_positions","joint_velocities","previous_joints","motor_mode",*AXES,"eyelid_openness",
                'rest_need','stimulation','alertness','rest_mode'))
        identifier(record["entity_id"])
        if (type(record["version"]) is not int or record["version"] not in (1,2,3)
                or record["name"] != "Arcus Alpha"
                or record.get("sleep_state", "awake") not in ("awake", "sleeping", "waking_up")
                or (record.get("sleep_state") == "sleeping" and record["target_posture"] != "lying")
                or record["facing"] not in ("up", "down", "left", "right")
                or record["target_posture"] not in ("standing", "lying")
                or type(record["height"]) not in (float, int) or not .25 <= record["height"] <= 1
                or type(record["radius"]) not in (float, int) or record["radius"] != .42):
            raise ContractError("Invalid embodiment record")
        values={k:v for k,v in record.items() if k!="version"}
        for key in ('rest_need','stimulation','alertness'):number(values.get(key,{'rest_need':.2,'stimulation':0,'alertness':.8}[key]),0,1)
        if values.get('rest_mode','active') not in ('active','resting'):raise ContractError('Invalid rest mode')
        if record["version"]==1:
            fields(record,("version","entity_id","name","facing","height","target_posture","radius"),("sleep_state",))
            values.update(joint_positions=pose((record["height"]-.25)/.75),
                          previous_joints=pose((record["height"]-.25)/.75))
        else:
            for key in ("joint_positions","joint_velocities","previous_joints"):
                fields(record.get(key),JOINTS)
                for value in record[key].values(): number(value,-10 if key=="joint_velocities" else 0,10 if key=="joint_velocities" else 1)
            if record.get("motor_mode") not in ("assisted","independent"): raise ContractError("Invalid motor mode")
            for key in AXES: number(record.get(key),-1,1)
            number(record.get("eyelid_openness"),0,1)
        return cls(**values)

    def step(self, held=False):
        step(self,held)
        from baby_arcus.rest_environment import advance_signals
        advance_signals(self,held)

    def snapshot(self):
        from baby_arcus.body_visual import visual_pose
        target = 1.0 if self.target_posture == "standing" else .25
        posture = self.target_posture if abs(self.height-target) < .001 else (
            "getting_up" if target > self.height else "lying_down")
        visual=visual_pose(self)
        if self.motor_mode=="independent":posture=visual['kind']
        if visual["kind"]=="sitting":posture="sitting"
        return {**self.record(), "posture": posture, "visual_pose":visual, "joints": {
            leg: {joint:self.joint_positions[f"{leg}.{joint}"] for joint in ("hip","knee","ankle")}
            for leg in LEGS}}

