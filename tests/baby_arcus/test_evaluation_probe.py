import unittest
from baby_arcus.evaluation_probe import LocalSimulation
from baby_arcus.baselines import scripted
from baby_arcus.lessons import generate
from baby_arcus.evaluation import seed_for,layout_for

class EvaluationProbeTests(unittest.TestCase):
    def test_collection_diagnostics_match_scripted_completion(self):
        import time
        from baby_arcus.collection import episode
        from baby_arcus.actions import ACTIONS
        from baby_arcus.messages import SIGNALS
        for family in ('switch_delivery','clue_search'):
            simulator=LocalSimulation()
            class ScriptedPolicy:
                def request(self,method,path,body):
                    return {'checkpoint_id':'diagnostic','agents':{
                        agent:{'action':ACTIONS.index(action.action),'signal':SIGNALS.index(action.signal),'value':0.}
                        for agent,action in scripted(simulator.world).items()}}
            # Use the known-solvable control layouts from the adapter test below.
            # These scripted records are never submitted to a learner.
            _,record=episode(simulator,ScriptedPolicy(),'diagnostic','test',time.time()+10,family,0,split='evaluation',difficulty=2)
            diagnostic=record['diagnostics']
            self.assertTrue(record['success'])
            self.assertTrue(diagnostic['objective_completed'])
            self.assertFalse(diagnostic['timed_out'])
            self.assertEqual(diagnostic['agent_actions'],2*record['steps'])
            self.assertEqual(sum(diagnostic['results'].values()),2*record['steps'])
            if family=='switch_delivery':
                self.assertEqual(diagnostic['subgoals'],{'plate_held':True,'object_collected':True})
            else:
                self.assertTrue(diagnostic['selection_made'])
                self.assertEqual(diagnostic['subgoals'],{})

    def test_local_adapter_matches_authoritative_world_through_completion(self):
        for family in ("switch_delivery","clue_search"):
            simulator=LocalSimulation()
            seed=seed_for("evaluation",0)
            layout=layout_for("evaluation",0)
            expected=generate(family,seed,64,layout,"evaluation",2)
            simulator.request("POST","/v1/episodes",{"schema_version":1,"request_id":"probe",
                "family":family,"seed":seed,"split":"evaluation","layout_id":layout,"difficulty":2,"max_steps":64})
            while not (expected.terminated or expected.truncated):
                actions=scripted(expected)
                transition=expected.advance(actions)
                result=simulator.request("POST","/v1/episodes/probe/step",{"actions":{k:v.wire() for k,v in actions.items()}})
                self.assertEqual(result["transition"],transition)
                self.assertEqual(simulator.world.snapshot(),expected.snapshot())
            self.assertTrue(expected.success)
