"""Actor observations never include full state, seeds, or hidden answer metadata."""
from baby_arcus.actions import ACTIONS

def visible(world, origin, target):
    ox, oy = origin
    tx, ty = target
    if max(abs(tx-ox), abs(ty-oy)) > 2:
        return False
    # Conservative ray sampling includes orthogonal cells at diagonal corners.
    dx, dy = tx-ox, ty-oy
    steps = max(abs(dx), abs(dy))*4
    for n in range(1, steps):
        x = ox + dx*n/steps
        y = oy + dy*n/steps
        cells = {(ix, iy)
                 for ix in (int(x+0.499999), int(x+0.500001))
                 for iy in (int(y+0.499999), int(y+0.500001))}
        for cell in cells:
            if cell in (tuple(origin), tuple(target)):
                continue
            if list(cell) in world.walls or (list(cell) == world.door and not world.door_open()):
                return False
    return True

def observe(world, agent_id):
    agent = world.agents[agent_id]
    origin = agent["position"]
    cells = []
    for y in range(world.size):
        for x in range(world.size):
            pos = [x, y]
            if not visible(world, origin, pos):
                continue
            features = []
            if pos in world.walls:
                features.append("wall")
            if pos == world.door:
                features.append("door_open" if world.door_open() else "door_closed")
            if pos == world.plate:
                features.append("plate")
            if pos == world.delivery:
                features.append("delivery")
            if pos == world.object_position:
                features.append("parcel")
            for container in world.containers:
                if pos == container["position"]:
                    features.append(container["marker"])
            for key, other in world.agents.items():
                if key != agent_id and pos == other["position"]:
                    features.append(key)
            if world.family == "clue_search" and pos == world.clue_position and agent["role"] == "scout":
                features.append("clue:" + world.correct_marker)
            cells.append({"position": pos, "features": features})
    # The mask describes action syntax, not secret-dependent physical feasibility.
    return {"agent_id": agent_id, "role": agent["role"], "step": world.step,
            "position": list(origin), "inventory": agent["inventory"], "cells": cells,
            "goal": "deliver_parcel" if world.family == "switch_delivery" else "select_indicated_marker",
            "messages": [dict(m) for m in world.messages if m["sender"] != agent_id],
            "last_result": world.results.get(agent_id),
            "action_mask": {action: True for action in ACTIONS}}

def observations(world):
    return {key: observe(world, key) for key in sorted(world.agents)}
