"""Model-facing tools. Human controls are never part of this catalog."""
from copy import deepcopy
from baby_arcus.contracts import ContractError, fields, identifier, digest
from baby_arcus.body_dynamics import validate_motor, JOINTS

BODY_ACTIONS = ("stand", "lie", "move", "step", "turn", "sleep", "wake_up", "joint", "head", "gaze", "eyelids",
                'rest','alert','sleep_when_ready','wake_voluntarily','inspect_object')
TOOLS = [
    {"name": "observe_view", "description": "Observe the currently permitted visual source. Cannot choose or broaden access.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "observe_body", "description": "Observe your body, placement, area and human cues.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "body_action", "description": "Use bounded joints, head, gaze and eyelids. Rest/alert change rest mode only; sleep_when_ready requires supported lying; wake_voluntarily preserves posture and eyes. Legacy stand/lie/sleep are assisted demonstrations and excluded by the rest policy.",
     "parameters": {"oneOf": [
         {"type": "object", "properties": {"kind": {"enum": ["stand", "lie", "sleep", "wake_up"]}},
          "required": ["kind"], "additionalProperties": False},
         {"type": "object", "properties": {"kind": {"const": "move"},
          "direction": {"enum": ["up", "down", "left", "right"]}},
          "required": ["kind", "direction"], "additionalProperties": False}]}}
]
TOOLS.extend({"name":name,"description":description,
              "parameters":{"type":"object","properties":{},"additionalProperties":False}}
             for name,description in (("observe_interactions","Read pending human interaction events; receipt does not establish understanding."),("observe_senses","Read body sensations without vision."),
                                      ("observe_messages","Read available human messages; queued sleeping messages are withheld.")))
TOOLS[2]['parameters']['oneOf'].append({'type':'object','properties':{'kind':{'enum':
    ['rest','alert','sleep_when_ready','wake_voluntarily']}},'required':['kind'],'additionalProperties':False})
TOOLS[2]['parameters']['oneOf'].append({'type':'object','properties':{'kind':{'const':'inspect_object'},
    'object_id':{'type':'string'}},'required':['kind','object_id'],'additionalProperties':False})
for kind,directions in (('step',['forward','backward']),('turn',['left','right'])):
    TOOLS[2]['parameters']['oneOf'].append({'type':'object','properties':{'kind':{'const':kind},
        'direction':{'enum':directions}},'required':['kind','direction'],'additionalProperties':False})
for kind,properties in (
    ("joint",{"joint":{"enum":list(JOINTS)},"delta":{"type":"number","minimum":-.15,"maximum":.15}}),
    ("head",{"yaw":{"type":"number","minimum":-1,"maximum":1},"pitch":{"type":"number","minimum":-1,"maximum":1}}),
    ("gaze",{"yaw":{"type":"number","minimum":-1,"maximum":1},"pitch":{"type":"number","minimum":-1,"maximum":1}}),
    ("eyelids",{"openness":{"type":"number","minimum":0,"maximum":1}})):
    TOOLS[2]["parameters"]["oneOf"].append({"type":"object","properties":{"kind":{"const":kind},**properties},
        "required":["kind",*properties],"additionalProperties":False})


def validate_body_action(action):
    if not isinstance(action, dict) or action.get("kind") not in BODY_ACTIONS:
        raise ContractError("Body tools cannot change human or environment controls")
    if action['kind']=='inspect_object':
        fields(action,('kind','object_id'));identifier(action['object_id']);return
    if action['kind'] in ('step','turn'):
        from baby_arcus.navigation_actions import resolve
        resolve(action,'up');return
    if action["kind"] in ("joint","head","gaze","eyelids"):
        validate_motor(action)
        return
    fields(action, ("kind", "direction") if action["kind"] == "move" else ("kind",))
    if action["kind"] == "move" and action["direction"] not in ("up", "down", "left", "right"):
        raise ContractError("Unknown body direction")


class BodyToolsApplication:
    """Restricted HTTP facade; deploy with a credential different from the human backend."""
    def __init__(self, dispatch, observe_view=None):
        self.dispatch = dispatch
        self.observe_view = observe_view

    def __call__(self, method, path, body):
        if method == "GET" and path == "/v1/tools":
            return 200, {"tools": deepcopy(TOOLS), "model_connected": False}
        if method != "POST" or path != "/v1/tools/call":
            raise KeyError(path)
        fields(body, ("request_id", "name", "arguments"))
        identifier(body["request_id"])
        if body["name"] == "observe_view":
            fields(body["arguments"], ())
            if self.observe_view is None:
                return 503, {"error": "Vision service is not connected"}
            return self.observe_view()
        if body["name"] == "observe_body":
            fields(body["arguments"], ())
            return self.dispatch("GET", "/v1/state", None)
        if body['name']=='observe_interactions':
            fields(body['arguments'],())
            return self.dispatch('GET','/v1/interactions',None)
        if body["name"] in ("observe_senses","observe_messages"):
            fields(body["arguments"],())
            return self.dispatch("GET","/v1/"+("senses" if body["name"]=="observe_senses" else "messages/available"),None)
        if body["name"] != "body_action":
            raise ContractError("Unknown body tool")
        validate_body_action(body["arguments"])
        return self.dispatch("POST", "/v1/action", {"request_id": "tool-"+digest(body["request_id"]),
            "source": "policy", "action": body["arguments"]})
