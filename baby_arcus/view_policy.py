"""Ephemeral viewing permission. Only trusted desktop events broaden the view."""
from dataclasses import dataclass, field
import time
import uuid
from baby_arcus.contracts import ContractError


@dataclass
class ViewPolicy:
    scope_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    epoch: int = 0
    region: str = "playpen"
    held: bool = False
    lease_until: float = 0.0
    sequence: int = 0
    event: str = "inside_playpen"

    def snapshot(self):
        return {"scope_id": self.scope_id, "epoch": self.epoch, "region": self.region,
                "held": self.held, "source": "desktop" if self.region == "desktop" else "playpen",
                "sequence": self.sequence, "event": self.event, "cursor_included": False}

    def stamp(self):
        return self.scope_id, self.epoch

    def reset(self, event="returned_to_playpen"):
        self.region, self.held, self.lease_until = "playpen", False, 0.0
        self.epoch += 1
        self.sequence += 1
        self.event = event

    def expire(self, now=None):
        now = time.monotonic() if now is None else now
        if self.lease_until and now >= self.lease_until:
            self.reset("desktop_connection_lost")
            return True
        return False

    def apply(self, kind, inside=None, now=None):
        now = time.monotonic() if now is None else now
        self.expire(now)
        if kind == "heartbeat":
            if self.held or self.region == "desktop":
                self.lease_until = now + 3
            return
        if kind == "return":
            self.reset()
            return
        if kind == "pickup":
            if self.held:
                raise ContractError("Arcus is already being held")
            self.held = True
            self.event = "picked_up"
        elif kind in ("carry", "drop"):
            if not self.held or type(inside) is not bool:
                raise ContractError("Carry/drop requires a held body and an inside flag")
            region = "playpen" if inside else "desktop"
            if region != self.region:
                self.region = region
                self.epoch += 1
            self.held = kind == "carry"
            self.event = "being_held" if self.held else "put_down"
        else:
            raise ContractError("Unknown desktop event")
        self.sequence += 1
        self.lease_until = now + 3 if self.held or self.region == "desktop" else 0.0
