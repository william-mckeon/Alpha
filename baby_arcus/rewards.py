"""Once-only shaping is accounted separately from verified completion."""
EVENT_VALUES = {
    "switch_delivery": {"plate_held": 0.05, "object_collected": 0.15},
    "clue_search": {},  # Do not leak correctness through intermediate rewards.
}

def award(world, events, success):
    shaping = 0.0
    for event in sorted(set(events)):
        if event not in world.rewarded and event in EVENT_VALUES[world.family]:
            value = min(EVENT_VALUES[world.family][event], max(0.0, 0.2-world.shaping_total))
            shaping += value
            world.shaping_total += value
            world.rewarded.append(event)
    return {"objective": 1.0 if success else 0.0, "intermediate": shaping, "teaching": 0.0}
