"""Interruptible gaze-plan bookkeeping; scores come from the shared learner.

Each call authorizes one step only. A newer observation is required before a
second step, even when the caller supplies the same predicted plan again.
"""
import math


class VisualPlan:
    def __init__(self, budget=3, lifetime=10):
        if type(budget) is not int or not 1 <= budget <= 3:
            raise ValueError('Invalid visual plan budget')
        if type(lifetime) not in (int, float) or not math.isfinite(lifetime) or not 0 < lifetime <= 30:
            raise ValueError('Invalid visual plan lifetime')
        self.budget, self.lifetime = budget, lifetime
        self.state = None

    @staticmethod
    def boundary(row):
        return tuple(row[key] for key in ('session', 'entity_id', 'environment_id', 'scope_id', 'epoch'))

    def cancel(self, reason):
        self.state = None
        return {'action': None, 'status': 'cancelled', 'reason': reason}

    def acknowledge(self, before, outcome):
        """Advance the scope only for this plan's verified, applied gaze step."""
        state = self.state
        if state is None or state.get('pending') is None:
            raise ValueError('No pending visual action')
        after = outcome['after']
        if (outcome.get('experience_id') != state['pending_id'] or before['id'] != state['pending_id']
                or outcome.get('action') != state['pending'] or outcome.get('executed') is not True
                or self.boundary(before) != state['boundary']
                or self.boundary(after)[:-1] != state['boundary'][:-1]
                or after['epoch'] not in (before['epoch'], before['epoch']+1)
                or after['tick'] < before['tick']
                or after.get('gaze', [0]*4)[2:] != [state['pending']['yaw'], state['pending']['pitch']]):
            return self.cancel('Unverified or interrupted visual action')
        state['boundary'] = self.boundary(after)
        state['pending'] = None
        return {'status': 'awaiting_observation'}

    def step(self, row, candidates, target, now):
        if type(now) not in (int, float) or not math.isfinite(now):
            raise ValueError('Invalid planning time')
        if row.get('hearing') or row.get('events'):
            return self.cancel('Caregiver or interaction input')
        if row['senses']['held'] or row['senses']['sleep_state'] != 'awake' or not row['vision']['available']:
            return self.cancel('Sensory availability changed')
        boundary = self.boundary(row)
        if self.state and self.state['boundary'] != boundary:
            return self.cancel('Sensory scope changed')
        if self.state and now >= self.state['expires']:
            return self.cancel('Plan expired')
        if not isinstance(target, str) or not target:
            return self.cancel('No object target')
        if self.state and self.state['target'] != target:
            return self.cancel('Target changed')
        if self.state and self.state.get('pending') is not None:
            return {'action': None, 'status': 'awaiting_execution'}
        if len(candidates) > 9:
            raise ValueError('Visual planning candidate budget exceeded')
        for candidate in candidates:
            action, score = candidate['action'], candidate['score']
            if action.get('kind') != 'gaze' or set(action) != {'kind', 'yaw', 'pitch'}:
                raise ValueError('Visual plan permits gaze actions only')
            if any(type(action[k]) not in (int, float) or not math.isfinite(action[k]) or abs(action[k]) > 1 for k in ('yaw', 'pitch')):
                raise ValueError('Invalid planned gaze')
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid learned search score')
        if not candidates:
            return self.cancel('No candidate views')
        if self.state is None:
            self.state = {'boundary': boundary, 'target': target, 'expires': now+self.lifetime,
                          'steps': 0, 'last_tick': -1, 'last_time': -1, 'views': []}
        state = self.state
        if state['steps'] >= self.budget:
            return self.cancel('Action budget exhausted')
        if row['tick'] <= state['last_tick'] or row['captured_at'] <= state['last_time']:
            return {'action': None, 'status': 'awaiting_observation'}
        remaining = [c for c in candidates if [c['action']['yaw'], c['action']['pitch']] not in state['views']]
        if not remaining:
            return self.cancel('Candidate views exhausted')
        selected = max(remaining, key=lambda c: c['score'])
        state['steps'] += 1
        state['last_tick'], state['last_time'] = row['tick'], row['captured_at']
        state['views'].append([selected['action']['yaw'], selected['action']['pitch']])
        state['pending'], state['pending_id'] = dict(selected['action']), row['id']
        return {'action': dict(selected['action']), 'score': selected['score'], 'target': target,
                'status': 'reobserve_after_step', 'steps': state['steps'], 'remaining': self.budget-state['steps']}
