"""Authoritative replayable environment. Only observations go to actors."""
from copy import deepcopy
from dataclasses import dataclass, field
from baby_arcus.actions import Action, MOVES, resolve_moves
from baby_arcus.contracts import ContractError, WORLD_VERSION
from baby_arcus.messages import deliver
from baby_arcus.rewards import award

@dataclass
class World:
    family: str
    seed: int
    agents: dict
    walls: list
    door: list | None = None
    plate: list | None = None
    object_position: list | None = None
    delivery: list | None = None
    containers: list = field(default_factory=list)
    clue_position: list | None = None
    correct_marker: str | None = None
    size: int = 7
    max_steps: int = 64
    step: int = 0
    terminated: bool = False
    truncated: bool = False
    success: bool = False
    rewarded: list = field(default_factory=list)
    shaping_total: float = 0.0
    messages: list = field(default_factory=list)
    results: dict = field(default_factory=dict)
    world_version: str = WORLD_VERSION
    split: str = "train"
    layout_id: int | None = None
    difficulty: int = 0

    def snapshot(self):
        return deepcopy(vars(self))

    @classmethod
    def restore(cls, state):
        if state.get("world_version") not in ("baby-grid-v1",WORLD_VERSION):
            raise ContractError("Incompatible world version")
        return cls(**deepcopy(state))

    def door_open(self):
        return self.plate is not None and any(a["position"] == self.plate for a in self.agents.values())

    def passable(self, position):
        x, y = position
        if not (0 <= x < self.size and 0 <= y < self.size) or list(position) in self.walls:
            return False
        return list(position) != self.door or self.door_open()

    def advance(self, actions):
        if self.terminated or self.truncated:
            raise ContractError("Episode already finished")
        if set(actions) != set(self.agents) or not all(isinstance(a, Action) for a in actions.values()):
            raise ContractError("Exactly one valid action per agent is required")
        positions, failures = resolve_moves(self, actions)
        for key, position in positions.items():
            self.agents[key]["position"] = list(position)
        self.results = {key: failures.get(key, "ok") for key in self.agents}
        events = []
        if self.door_open():
            events.append("plate_held")
        selected = False
        for key, action in sorted(actions.items()):
            agent = self.agents[key]
            pos = agent["position"]
            if action.action == "pickup":
                if self.object_position == pos and agent["inventory"] is None:
                    agent["inventory"] = "parcel"
                    self.object_position = None
                    events.append("object_collected")
                else:
                    self.results[key] = "nothing_to_pick_up"
            elif action.action == "drop":
                if agent["inventory"] == "parcel" and self.object_position is None:
                    agent["inventory"] = None
                    self.object_position = list(pos)
                    if pos == self.delivery:
                        self.success = True
                        self.terminated = True
                else:
                    self.results[key] = "nothing_to_drop"
            elif action.action == "interact":
                choices = [c for c in self.containers if c["position"] == pos]
                if self.family == "clue_search" and agent["role"] == "seeker" and choices:
                    selected = True
                    self.success = choices[0]["marker"] == self.correct_marker
                    self.terminated = True
                else:
                    self.results[key] = "nothing_to_interact"
        self.step += 1
        self.messages = deliver(actions, self.step)
        self.truncated = self.step >= self.max_steps and not self.terminated
        reward = award(self, events, self.success)
        return {"step": self.step, "reward": reward, "terminated": self.terminated,
                "truncated": self.truncated, "success": self.success,
                "selection_made": selected}
