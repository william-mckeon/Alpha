"""Scripted controls diagnose task solvability, never supply imitation targets."""
from collections import deque
from random import Random
from baby_arcus.actions import Action, MOVES, ACTIONS
from baby_arcus.messages import SIGNALS

def route(world, agent_id, target):
    start = tuple(world.agents[agent_id]["position"])
    occupied = {tuple(a["position"]) for k,a in world.agents.items() if k != agent_id}
    todo = deque([(start, [])])
    seen = {start}
    while todo:
        pos, path = todo.popleft()
        if list(pos) == target:
            return Action(path[0] if path else "wait")
        for name,(dx,dy) in MOVES.items():
            nxt = (pos[0]+dx,pos[1]+dy)
            if nxt not in seen and nxt not in occupied and world.passable(nxt):
                seen.add(nxt)
                todo.append((nxt,path+[name]))
    return Action()

def scripted(world, use_messages=True):
    actions = {}
    for key,agent in world.agents.items():
        if world.family == "switch_delivery":
            if agent["role"] == "holder":
                actions[key] = route(world,key,world.plate)
            elif agent["inventory"]:
                actions[key] = (Action("drop") if agent["position"] == world.delivery
                                else route(world,key,world.delivery))
            else:
                actions[key] = (Action("pickup") if agent["position"] == world.object_position
                                else route(world,key,world.object_position))
        elif agent["role"] == "scout":
            # Read the clue through the actual observation interface.
            from baby_arcus.observations import observe
            clues = [f[5:] for c in observe(world,key)["cells"] for f in c["features"] if f.startswith("clue:")]
            actions[key] = Action(signal=clues[0] if clues and use_messages else "none")
        else:
            signals = [m["signal"] for m in world.messages
                       if m["sender"] != key and m["signal"].startswith("marker_")]
            if use_messages and not signals:
                actions[key] = Action()
            else:
                marker = signals[-1] if signals else "marker_red"
                target = next(c["position"] for c in world.containers if c["marker"] == marker)
                actions[key] = Action("interact") if agent["position"] == target else route(world,key,target)
    return actions

def random_actions(world, rng=None):
    rng = rng or Random()
    return {key:Action(rng.choice(ACTIONS),rng.choice(SIGNALS)) for key in world.agents}
