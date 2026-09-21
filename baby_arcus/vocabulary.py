"""Versioned structured tokens and prediction targets; never encode hidden state."""
from baby_arcus.actions import ACTIONS
from baby_arcus.messages import SIGNALS
from baby_arcus.contracts import ContractError, fields, integer, digest

VERSION = "baby-vocab-v1"
SIZE = 512
FEATURES = ("wall","door_open","door_closed","plate","delivery","parcel",
            "marker_red","marker_blue","a","b","clue:marker_red","clue:marker_blue")
RESULTS = (None,"ok","blocked","collision","nothing_to_pick_up","nothing_to_drop","nothing_to_interact")
ROLES = ("holder","carrier","scout","seeker")
GOALS = ("deliver_parcel","select_indicated_marker")
BOS, OBS, END, CELL, MESSAGE, ACTION = 1,2,3,4,5,6
VOCABULARY_HASH = digest({"version":VERSION,"actions":list(ACTIONS),"signals":list(SIGNALS),
                          "features":list(FEATURES),"results":list(RESULTS)})

def prefix(observation):
    return [BOS, 10+GOALS.index(observation["goal"]), 16+ROLES.index(observation["role"])]

def encode(observation):
    fields(observation,("agent_id","role","step","position","inventory","cells","goal",
                        "messages","last_result","action_mask"))
    if observation["agent_id"] not in ("a","b") or observation["role"] not in ROLES or observation["goal"] not in GOALS:
        raise ContractError("Unknown agent, role, or goal")
    integer(observation["step"],0,256)
    if observation["inventory"] not in (None,"parcel") or observation["last_result"] not in RESULTS:
        raise ContractError("Unknown inventory or result")
    frame = [OBS, position_token(observation["position"]), 32+int(observation["inventory"] is not None),
             40+RESULTS.index(observation["last_result"])]
    seen = set()
    for cell in sorted(observation["cells"],key=lambda c:tuple(c["position"])):
        fields(cell,("position","features"))
        pos = position_token(cell["position"])
        if pos in seen:
            raise ContractError("Duplicate visible cell")
        seen.add(pos)
        frame += [CELL,pos]
        for feature in sorted(set(cell["features"])):
            if feature not in FEATURES:
                raise ContractError("Unknown visible feature")
            frame.append(100+FEATURES.index(feature))
    for message in observation["messages"]:
        fields(message,("sender","signal","step"))
        if message["sender"] not in ("a","b") or message["signal"] not in SIGNALS:
            raise ContractError("Invalid visible message")
        frame += [MESSAGE,60+("a","b").index(message["sender"]),70+SIGNALS.index(message["signal"])]
    if set(observation["action_mask"]) != set(ACTIONS) or not all(type(v) is bool for v in observation["action_mask"].values()):
        raise ContractError("Invalid action mask")
    return frame+[END]

def position_token(position):
    if not isinstance(position,list) or len(position) != 2:
        raise ContractError("Invalid position")
    x,y = (integer(n,0,6) for n in position)
    return 200+y*7+x

def action_tokens(action,signal):
    return [ACTION,300+action,320+signal]

def targets(observation):
    encode(observation)
    cells = [[0.0]*len(FEATURES) for _ in range(49)]
    visible = [0.0]*49
    for cell in observation["cells"]:
        index = position_token(cell["position"])-200
        visible[index] = 1.0
        for feature in cell["features"]:
            cells[index][FEATURES.index(feature)] = 1.0
    return {"cells":cells,"visible":visible,
            "inventory":int(observation["inventory"] is not None),
            "result":RESULTS.index(observation["last_result"])}
