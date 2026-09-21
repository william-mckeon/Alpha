"""Single action and clock coordinator for independent body and environment."""
from copy import deepcopy
from baby_arcus.contracts import ContractError, fields
from baby_arcus.embodiment import Embodiment
from baby_arcus.playroom import Playroom, DT, DIRECTIONS, coordinate
from baby_arcus.view_policy import ViewPolicy
from baby_arcus.body_dynamics import validate_motor
from baby_arcus.body_senses import observe_body_senses

class PlaySession:
    def __init__(self, body=None, environment=None):
        self.body = body if body is not None else Embodiment()
        self.environment = environment if environment is not None else Playroom()
        self.environment.attach(self.body.entity_id, self.body.radius)
        self.tick, self.paused, self.cue = 0, False, None
        self.last_result = "Ready for interaction"
        self.view = ViewPolicy()

    def replace_environment(self, environment):
        self.view.reset()
        self.environment.placements.pop(self.body.entity_id, None)
        self.environment = environment
        self.environment.attach(self.body.entity_id, self.body.radius)
        self.cue = None

    def snapshot(self):
        return {"version": "arcus-playroom-v2", "tick": self.tick, "dt": DT,
                "paused": self.paused, "arcus": self.body.snapshot(), "view": self.view.snapshot(),
                "environment": self.environment.snapshot(), "cue": deepcopy(self.cue),
                "last_result": self.last_result, "controller": "human demonstration; no model connected",
                "senses": observe_body_senses(self.body,self.view.held),
                "body_model": "simplified normalized-joint support model; not rigid-body physics"}

    def step(self):
        if not self.paused:
            self.tick += 1
            self.body.step(self.view.held)

    def action(self, action):
        if not isinstance(action, dict) or not isinstance(action.get("kind"), str):
            raise ContractError("Expected an action kind")
        kind = action["kind"]
        if kind=='color_lesson':
            fields(action,('kind','colors','balls'))
            if self.paused or self.view.held or self.view.region!='playpen':raise ContractError('Lesson requires active playpen')
            self.environment.color_lesson(action['colors'],action['balls'])
            self.view.epoch+=1
            return 'Color lesson ready'
        if kind=='inspect_object':
            fields(action,('kind','object_id'))
            if self.paused or self.view.held or self.view.region!='playpen' or self.body.sleep_state!='awake' or self.body.eyelid_openness<=0:
                raise ContractError('Object interaction requires awake open eyes in the active playpen')
            result=self.environment.interact(self.body.entity_id,action['object_id'])
            self.last_result='Object response: '+result['response'];self.view.epoch+=1
            return result
        if kind in ('rest','alert','sleep_when_ready','wake_voluntarily'):
            fields(action,('kind',))
            if self.paused or self.view.held or self.view.region!='playpen':
                raise ContractError('Rest action requires an unheld active playpen body')
            if kind=='sleep_when_ready':
                from baby_arcus.posture_goals import achieved
                if not achieved(observe_body_senses(self.body),'lying'):
                    raise ContractError('Voluntary sleep requires supported lying; no assisted posture change')
                self.body.sleep_state='sleeping';self.body.target_posture='lying'
                self.body.motor_mode='independent';self.body.eyelid_openness=0;self.body.rest_mode='resting'
                self.view.epoch+=1
            elif kind=='wake_voluntarily':
                self.body.sleep_state='awake';self.view.epoch+=1
            elif self.body.sleep_state!='awake':raise ContractError('Wake before changing rest mode')
            else:self.body.rest_mode='resting' if kind=='rest' else 'active'
            self.last_result='Voluntary rest action: '+kind
            return self.last_result
        if kind in ('step','turn'):
            from baby_arcus.navigation_actions import resolve
            move,facing=resolve(action,self.body.facing)
            if (self.body.sleep_state!='awake' or self.paused or self.view.held or self.view.region!='playpen'
                    or self.body.height<.99 or not observe_body_senses(self.body)['stable']):
                self.last_result='Relative movement needs a stable awake standing body in the playpen'
                return self.last_result
            if move:self.action(move)
            else:self.last_result='Turned '+action['direction']
            self.body.facing=facing
            self.view.epoch+=1
            return self.last_result
        if kind in ("joint","head","gaze","eyelids"):
            validate_motor(action)
            if self.body.sleep_state != "awake" or (kind!="eyelids" and (self.paused or self.view.held)):
                self.last_result="Motor action unavailable while asleep, paused or held"
                return self.last_result
            if kind=="joint":
                self.body.motor_mode="independent"
                key=action["joint"]
                self.body.joint_positions[key]=round(max(0,min(1,self.body.joint_positions[key]+action["delta"])),6)
            elif kind=="eyelids":
                self.body.eyelid_openness=action["openness"]
                self.view.epoch+=1
            else:
                prefix="head" if kind=="head" else "eye"
                setattr(self.body,prefix+"_yaw",action["yaw"])
                setattr(self.body,prefix+"_pitch",action["pitch"])
                self.view.epoch+=1
            self.last_result="Body control applied: "+kind
            return self.last_result
        schemas = {"move": ("direction",), "stand": (), "lie": (), "sleep": (), "wake_up": (), "human": ("x", "y", "name"),
                   "call": (), "feedback": ("value",), "pause": ("value",), "reset": (), "return": ()}
        if kind not in schemas:
            raise ContractError("Unknown playroom action")
        fields(action, ("kind", *schemas[kind]))
        if kind == "move" and action["direction"] not in tuple(DIRECTIONS):
            raise ContractError("Unknown direction")
        if kind == "human":
            x = coordinate(action["x"], .5, self.environment.width-.5)
            y = coordinate(action["y"], .5, self.environment.height-.5)
            if action["name"] not in ("You", "Your wife"):
                raise ContractError("Unknown participant")
        if kind == "feedback" and action["value"] not in ("encourage", "try_again"):
            raise ContractError("Unknown feedback")
        if kind == "pause" and type(action["value"]) is not bool:
            raise ContractError("Pause must be boolean")
        result = ""
        if kind == "sleep":
            if self.body.sleep_state != "sleeping":
                self.body.sleep_state = "sleeping"
                self.body.target_posture = "lying"
                self.body.motor_mode = "assisted"
                self.body.eyelid_openness = 0
                self.view.epoch += 1
            result = "Sleeping; visual observations disabled"
        elif kind == "wake_up":
            if self.body.sleep_state == "sleeping":
                self.body.sleep_state = "waking_up"
                # Wake is awareness, not an automatic stand or eye-opening command.
                self.view.epoch += 1
            result = "Wake-up requested; current viewing boundary preserved"
        elif self.body.sleep_state != "awake" and kind in ("move", "stand", "lie"):
            result = "Wake up fully before moving"
        elif self.view.held and kind in ("move", "stand", "lie"):
            result = "Being held; body actions are unavailable"
        elif kind == "move" and self.view.region == "desktop":
            result = "Desktop walking is not enabled; human placement only"
        elif self.paused and kind in ("move", "stand", "lie"):
            result = "Resume before moving"
        elif kind == "reset":
            self.view.reset()
            self.environment.reset()
            self.cue = None
            result = "Play area reset; Arcus identity and posture preserved"
        elif kind == "return":
            self.view.reset()
            result = "Returned to the playpen; desktop view revoked"
        elif kind == "pause":
            self.paused = action["value"]
            result = "Session paused" if self.paused else "Session resumed"
        elif kind in ("stand", "lie"):
            self.body.motor_mode = "assisted"
            self.body.target_posture = "standing" if kind == "stand" else "lying"
            result = "Posture requested: " + self.body.target_posture
        elif kind == "move":
            if (self.body.height < .99 or not observe_body_senses(self.body)['stable']
                or (self.body.motor_mode=='assisted' and self.body.target_posture!='standing')):
                result = "Stand up before moving"
            else:
                dx, dy = DIRECTIONS[action["direction"]]
                wall = self.environment.move(self.body.entity_id, dx*.32, dy*.32, self.body.radius)
                self.body.facing = action["direction"]
                self.view.epoch += 1
                result = "Wall reached" if wall else "Moved " + self.body.facing
        elif kind == "human":
            self.environment.human.update(x=x, y=y, name=action["name"])
            result = action["name"] + " moved in the play area"
        elif kind in ("call", "feedback"):
            human = self.environment.human
            self.cue = {"kind": "come_here" if kind == "call" else action["value"],
                        "from": human["name"], "tick": self.tick,
                        "entity_id": self.body.entity_id, "environment_id": self.environment.environment_id}
            if kind == "call":
                self.cue.update(x=human["x"], y=human["y"])
            result = "Human cue recorded"
        self.last_result = result
        return result

