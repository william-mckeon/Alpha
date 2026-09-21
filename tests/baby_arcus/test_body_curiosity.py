import tempfile
import unittest
from pathlib import Path
from baby_arcus.body_curiosity import EffectMemory, experiment, trial
from baby_arcus.body_dynamics import pose
from baby_arcus.embodiment import Embodiment
from baby_arcus.play_session import PlaySession


class CuriosityTests(unittest.TestCase):
    def session(self):
        return PlaySession(Embodiment(height=.625,motor_mode='independent',
                                     joint_positions=pose(.5),previous_joints=pose(.5)))

    def test_repetition_reduces_uncertainty_and_prediction_error(self):
        memory=EffectMemory()
        first=trial(self.session(),1,memory)
        second=trial(self.session(),1,memory)
        self.assertGreater(first['prediction_error'],0)
        self.assertAlmostEqual(second['prediction_error'],0)
        self.assertLess(second['novelty_reward'],first['novelty_reward'])
        self.assertEqual(sum(abs(x)>0 for x in second['observed_joint_delta']),1)

    def test_inactive_body_cannot_explore(self):
        for mode in ('paused','held','sleeping'):
            session=self.session()
            if mode=='paused':session.paused=True
            elif mode=='held':session.view.held=True
            else:session.body.sleep_state='sleeping'
            with self.assertRaises(ValueError):trial(session,1,EffectMemory())
            self.assertEqual(session.body.joint_positions,pose(.5))

    def test_all_actions_and_persisted_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            report=experiment(Path(root)/'discovery',48)
            self.assertEqual(report['unique_joint_actions'],24)
            self.assertEqual(report['unstable_trials'],0)
            self.assertAlmostEqual(report['repeated_observation_error'],0)
            self.assertFalse(report['neural_checkpoint_updated'])
            self.assertFalse(report['live_body_controlled'])
            self.assertEqual(len((Path(root)/'discovery/transitions.jsonl').read_text().splitlines()),48)
            with self.assertRaises(FileExistsError):experiment(Path(root)/'discovery',48)
