"""Independent, bounded episode histories with complete-frame truncation."""
from copy import deepcopy
from baby_arcus.contracts import ContractError, digest
from baby_arcus.vocabulary import encode, prefix, action_tokens

class AgentMemory:
    def __init__(self, agent_id, limit=512):
        self.agent_id = agent_id
        self.limit = limit
        self.records = []
        self.header = []

    def observe(self, observation):
        if observation["agent_id"] != self.agent_id:
            raise ContractError("Another agent's observation cannot enter this history")
        frame = encode(observation)
        self.header = prefix(observation)
        fingerprint = digest(observation)
        step = observation["step"]
        if self.records and self.records[-1]["step"] == step:
            if self.records[-1]["digest"] != fingerprint:
                raise ContractError("Conflicting observation at same step")
        else:
            if self.records and step != self.records[-1]["step"]+1:
                raise ContractError("History step skipped or moved backward")
            if len(frame)+len(self.header)+3 > self.limit:
                raise ContractError("Observation exceeds context capacity")
            self.records.append({"step":step,"digest":fingerprint,"tokens":frame,"action":None})
        self._trim()
        return self.context()

    def choose(self, action, signal):
        if not self.records:
            raise ContractError("No observation to attach action to")
        entry = self.records[-1]
        choice = action_tokens(action,signal)
        if entry["action"] is not None and entry["action"] != choice:
            raise ContractError("Conflicting action for observed step")
        entry["action"] = choice
        self._trim()

    def _trim(self):
        while len(self.records)>1 and len(self.context())>self.limit:
            self.records.pop(0)

    def context(self):
        result = list(self.header)
        for entry in self.records:
            result += entry["tokens"]
            if entry["action"] is not None:
                result += entry["action"]
        return result

    def preview(self, observation):
        clone = deepcopy(self)
        return clone.observe(observation)
