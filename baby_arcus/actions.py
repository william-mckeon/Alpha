"""Movement proposals are simultaneous; conflicts have no ID priority."""
from dataclasses import dataclass
from baby_arcus.contracts import ContractError, fields
from baby_arcus.messages import validate_signal

MOVES = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}
ACTIONS = ("wait", *MOVES, "pickup", "drop", "interact")

@dataclass(frozen=True)
class Action:
    action: str = "wait"
    signal: str = "none"

    @classmethod
    def parse(cls, value):
        fields(value, ("action",), ("signal",))
        if value["action"] not in ACTIONS:
            raise ContractError("Unknown action")
        return cls(value["action"], validate_signal(value.get("signal", "none")))

    def wire(self):
        return {"action": self.action, "signal": self.signal}

def resolve_moves(world, actions):
    original = {key: tuple(agent["position"]) for key, agent in world.agents.items()}
    proposed = dict(original)
    failures = {}
    for key, action in actions.items():
        if action.action in MOVES:
            dx, dy = MOVES[action.action]
            target = (original[key][0] + dx, original[key][1] + dy)
            if world.passable(target):
                proposed[key] = target
            else:
                failures[key] = "blocked"
    # Resolve all collisions repeatedly: a blocked mover may also block another mover.
    while True:
        blocked = {key for key in proposed
                   if any(key != other and (proposed[key] == proposed[other] or
                          (proposed[key] == original[other] and proposed[other] == original[key]))
                          for other in proposed)}
        changed = False
        for key in blocked:
            if proposed[key] != original[key]:
                proposed[key] = original[key]
                failures[key] = "collision"
                changed = True
        if not changed:
            break
    return proposed, failures
