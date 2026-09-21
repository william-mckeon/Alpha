"""Prior visual-survey features; no simulator identities or future frames."""
import math
import json


def position(value):
    # Invert the documented half-frame 2D gaze crop. This is camera geometry,
    # not access to world objects, and is valid only for a stationary survey.
    return [(value[3+i]+value[5+i])/4+(max(-1, min(1, value[7+i]+value[9+i]))+1)/4 for i in (0, 1)]


def pair_features(first, second, query):
    for value in (first, second, query):
        if len(value) != 11 or any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
            raise ValueError('Invalid visual inventory feature')
    a, b = position(first), position(second)
    return ([abs(first[i]-second[i]) for i in range(3)]+[abs(a[i]-b[i]) for i in (0, 1)]+
            [sum((query[i]-value[i])**2 for i in range(3))/3 for value in (first, second)])


def uncertainty(model, hidden, inventory, query):
    """A learned risk estimate from a bounded prior survey."""
    import torch
    if not 1 <= len(inventory) <= 32:
        return 1.0
    pairs = [pair_features(a, b, query) for i, a in enumerate(inventory) for b in inventory[i+1:]]
    # A single apparent position provides no evidence of scene-wide uniqueness.
    if not pairs or max(sum(v*v for v in pair[3:5]) for pair in pairs) < .04**2:
        return 1.0
    features = torch.tensor(pairs, device=hidden.device)
    with torch.no_grad():
        return float(model.uncertainty_logits(hidden.expand(len(pairs), -1), features).sigmoid().max())


def identity_confidence(association, risk):
    """Separate the binary ambiguity decision from pairwise identity confidence.

    These independently trained classifiers are not calibrated independent
    probabilities of the same event. Multiplication silently imposed a 0.1
    ambiguity cutoff at the 0.9 association threshold. Use the balanced BCE
    classifier's decision boundary, retaining the original association gate.
    """
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (association, risk)):
        raise ValueError('Invalid identity confidence')
    return 0.0 if risk >= .5 else association


class VisualSurvey:
    """Durable bounded priming observations, isolated exactly like object memory."""
    def __init__(self, memory, generation='test'):
        self.memory = memory
        self.generation = generation
        memory.db.execute('CREATE TABLE IF NOT EXISTS object_surveys (namespace TEXT PRIMARY KEY, value TEXT NOT NULL)')

    def observe(self, row, descriptors):
        from baby_arcus.shared_continuity_curriculum import VIEWS
        namespace = self.memory.namespace(row)
        stored = self.memory.db.execute('SELECT value FROM object_surveys WHERE namespace=?', (namespace,)).fetchone()
        if len(descriptors) > 32:
            raise ValueError('Survey observation budget exceeded')
        for value in descriptors:
            pair_features(value, value, value)
        boundary = [self.generation, row['session'], row['scope_id'], row['gaze'][:2], row.get('hearing_relative')]
        state = json.loads(stored[0]) if stored else None
        if state is None or state['boundary'] != boundary:
            state = {'boundary': boundary, 'views': {}, 'complete': False, 'inventory': []}
        if not state['complete']:
            gaze = tuple(row['gaze'][2:])
            if gaze in VIEWS:
                state['views'][str(VIEWS.index(gaze))] = descriptors[:32]
            state['complete'] = len(state['views']) == len(VIEWS)
            state['inventory'] = [d for key in sorted(state['views'], key=int) for d in state['views'][key]][:32]
            with self.memory.db:
                self.memory.db.execute('INSERT OR REPLACE INTO object_surveys VALUES (?,?)', (namespace, json.dumps(state)))
        return state
