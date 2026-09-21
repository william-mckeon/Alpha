import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from baby_arcus.actions import Action
from baby_arcus.baselines import scripted
from baby_arcus.contracts import canonical, decode
from baby_arcus.services.simulation import SimulationApplication
from baby_arcus.transport import RemoteError
from baby_arcus.world import World
from tests.baby_arcus.conftest import running_service,LocalArtifacts

def start_request(request_id="create",family="switch_delivery",seed=3):
    return {"schema_version":1,"request_id":request_id,"family":family,"seed":seed}

def step_request(step=0,request_id="step"):
    return {"schema_version":1,"request_id":request_id,"step":step,
            "actions":{"a":{"action":"wait"},"b":{"action":"wait"}}}

class ServiceTests(unittest.TestCase):
    def test_live_artifact_outage_and_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            with running_service("artifacts",root) as (artifact,process):
                port = int(artifact.base_url.rsplit(":",1)[1])
                with running_service("simulation",root,artifact.base_url) as (sim,_):
                    created = sim.request("POST","/v1/episodes",start_request())
                    path = "/v1/episodes/"+created["episode_id"]+"/step"
                    process.terminate()
                    process.wait(timeout=10)
                    with self.assertRaises(RemoteError) as failure:
                        sim.request("POST",path,step_request())
                    self.assertEqual(failure.exception.status,503)
                    self.assertEqual(sim.request("GET",path[:-5])["step"],0)
                    with running_service("artifacts",root,port=port):
                        result = sim.request("POST",path,step_request())
                        self.assertEqual(result["step"],1)

    def test_live_end_to_end_and_replay(self):
        with tempfile.TemporaryDirectory() as root:
            with running_service("artifacts",root) as (artifact, _):
                with running_service("simulation",root,artifact.base_url) as (sim,_):
                    for family in ("switch_delivery","clue_search"):
                        response = sim.request("POST","/v1/episodes",start_request(family,family))
                        episode_id = response["episode_id"]
                        initial = artifact.request("GET","/v1/artifacts/"+response["artifact_id"])["payload"]
                        world = World.restore(initial["state"])
                        self.assertEqual(response,sim.request("POST","/v1/episodes",start_request(family,family)))
                        while not (world.terminated or world.truncated):
                            actions = scripted(world)
                            body = {"schema_version":1,"request_id":family+"-"+str(world.step),
                                    "step":world.step,"actions":{k:v.wire() for k,v in actions.items()}}
                            expected = world.advance(actions)
                            response = sim.request("POST","/v1/episodes/"+episode_id+"/step",body)
                            self.assertEqual(response["transition"],expected)
                        self.assertTrue(world.success)
                        replay = sim.request("GET","/v1/episodes/"+episode_id+"/replay")
                        self.assertEqual(len(replay["artifact_ids"]),world.step+1)
                        final = artifact.request("GET","/v1/artifacts/"+replay["artifact_ids"][-1])["payload"]
                        self.assertEqual(final["state"],world.snapshot())

    def test_restart_and_concurrent_duplicates(self):
        with tempfile.TemporaryDirectory() as root:
            with running_service("artifacts",root) as (artifact,_):
                with running_service("simulation",root,artifact.base_url) as (sim,_):
                    created = sim.request("POST","/v1/episodes",start_request())
                    path = "/v1/episodes/"+created["episode_id"]+"/step"
                    with ThreadPoolExecutor(4) as executor:
                        responses = list(executor.map(lambda _:sim.request("POST",path,step_request()),range(4)))
                    self.assertTrue(all(r == responses[0] for r in responses))
                    self.assertEqual(sim.request("GET",path[:-5])["step"],1)
                with running_service("simulation",root,artifact.base_url) as (sim,_):
                    self.assertEqual(sim.request("POST",path,step_request()),responses[0])
                    for bad in (step_request(0,"different"),step_request(1,"step")):
                        with self.assertRaises(RemoteError) as failure:
                            sim.request("POST",path,bad)
                        self.assertEqual(failure.exception.status,409)
                    self.assertEqual(sim.request("GET",path[:-5])["step"],1)

    def test_dependency_failure_does_not_commit_step(self):
        with tempfile.TemporaryDirectory() as root:
            artifacts = LocalArtifacts(Path(root)/"artifacts")
            app = SimulationApplication(Path(root)/"sim",artifacts)
            _,created = app("POST","/v1/episodes",start_request())
            path = "/v1/episodes/"+created["episode_id"]+"/step"
            original = artifacts.request
            def fail(*args,**kwargs):
                raise RemoteError(503,"down")
            artifacts.request = fail
            with self.assertRaises(RemoteError):
                app("POST",path,step_request())
            self.assertEqual(app("GET",path[:-5],None)[1]["step"],0)
            artifacts.request = original
            self.assertEqual(app("POST",path,step_request())[1]["step"],1)
