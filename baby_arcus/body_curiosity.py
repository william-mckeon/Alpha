"""Bounded body discovery experiment; does not update the Arcus neural checkpoint."""
import argparse
import json
import math
import random
from pathlib import Path

from baby_arcus.body_dynamics import JOINTS, pose
from baby_arcus.body_senses import observe_body_senses
from baby_arcus.body_vocabulary import ACTIONS
from baby_arcus.embodiment import Embodiment
from baby_arcus.play_session import PlaySession


class EffectMemory:
    """Learn action effects and reduce novelty after repeated observations."""
    def __init__(self):
        self.effects = {}
        self.counts = {}

    def observe(self, action, before, after):
        effect = [after['joint_positions'][k] - before['joint_positions'][k] for k in JOINTS]
        prediction = self.effects.get(action, [0.0] * len(JOINTS))
        error = sum(abs(a-b) for a,b in zip(effect, prediction)) / len(JOINTS)
        count = self.counts.get(action, 0)
        # Uncertainty falls with experience; raw surprise alone earns no bonus.
        novelty = 1 / math.sqrt(count + 1)
        self.effects[action] = [p + (e-p)/(count+1) for p,e in zip(prediction,effect)]
        self.counts[action] = count + 1
        return {'prediction_error': error, 'novelty_reward': novelty,
                'visits_before': count, 'predicted_joint_delta': prediction,
                'observed_joint_delta': effect}

    def choose(self, rng):
        least = min(self.counts.get(i,0) for i in range(1,len(ACTIONS)))
        return rng.choice([i for i in range(1,len(ACTIONS)) if self.counts.get(i,0)==least])


def trial(session, action, memory):
    if session.paused or session.view.held or session.body.sleep_state != 'awake':
        raise ValueError('Discovery requires an awake, unheld, unpaused body')
    if type(action) is not int or not 1 <= action < len(ACTIONS):
        raise ValueError('Discovery requires a bounded joint action')
    before = observe_body_senses(session.body)
    session.action(ACTIONS[action])
    session.step()
    after = observe_body_senses(session.body)
    return {'action_index':action, 'action':ACTIONS[action], 'before':before,
            'after':after, 'visual_pose':session.body.snapshot()['visual_pose'],
            **memory.observe(action,before,after)}


def experiment(output, trials=240, seed=917):
    if type(trials) is not int or not 24 <= trials <= 2400:
        raise ValueError('Use 24 to 2400 bounded trials')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    memory = EffectMemory()
    rng = random.Random(seed)
    records = []
    with (output/'transitions.jsonl').open('w',encoding='utf-8') as stream:
        for index in range(trials):
            # An explicit experiment reset isolates one cause from accumulated
            # imbalance. No saved/live body or neural weights are touched.
            body = Embodiment(height=.625, motor_mode='independent', eyelid_openness=0,
                              joint_positions=pose(.5), previous_joints=pose(.5))
            record = trial(PlaySession(body), memory.choose(rng), memory)
            record.update(trial=index, reset='uniform_joint_extension_0.5')
            stream.write(json.dumps(record)+'\n')
            records.append(record)
    first = [r['prediction_error'] for r in records if r['visits_before']==0]
    repeated = [r['prediction_error'] for r in records if r['visits_before']>0]
    report = {'schema':'arcus-body-discovery-v1', 'trials':trials, 'seed':seed,
              'unique_joint_actions':len(memory.counts),
              'first_observation_error':sum(first)/len(first),
              'repeated_observation_error':sum(repeated)/len(repeated) if repeated else None,
              'unstable_trials':sum(not r['after']['stable'] for r in records),
              'neural_checkpoint_updated':False, 'live_body_controlled':False,
              'controller':'least-observed-action exploration with learned effect memory',
              'counts':memory.counts, 'learned_effects':memory.effects}
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--trials',type=int,default=240)
    args = parser.parse_args()
    print(json.dumps(experiment(args.output,args.trials),indent=2))
