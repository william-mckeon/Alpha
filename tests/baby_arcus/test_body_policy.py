import tempfile
from pathlib import Path
import unittest
import torch
from baby_arcus.body_policy import BodyPolicy,load,save
from baby_arcus.body_vocabulary import ACTIONS
from baby_arcus.standing_environment import StandingEnvironment
class PolicyTests(unittest.TestCase):
    def test_controller_stops_on_pause(self):
        from baby_arcus.services.body_controller import run
        class Client:
            def request(self,*args):
                return {"sleep_state":"awake","held":False,"paused":True}
        class Policy:
            def choose(self,*args):
                raise AssertionError("Paused controller must not choose an action")
        self.assertEqual(run(Client(),Policy(),steps=3,interval=0),[])
    def test_checkpoint_optimizer_and_action_schema(self):
        torch.set_num_threads(2);torch.manual_seed(8)
        model=BodyPolicy();optimizer=torch.optim.AdamW(model.parameters())
        state=StandingEnvironment().observe()
        loss=model([state]).softmax(-1)[0,2]
        loss.backward();optimizer.step()
        self.assertTrue(all(a is None or a["kind"]=="joint" for a in ACTIONS))
        with tempfile.TemporaryDirectory() as root:
            path=Path(root,"policy.pt");save(path,model,optimizer,1)
            restored,data=load(path)
            torch.testing.assert_close(model([state]),restored([state]))
            resumed=torch.optim.AdamW(restored.parameters());resumed.load_state_dict(data["optimizer"])
            loss=restored([state]).softmax(-1)[0,2]
            resumed.zero_grad();loss.backward();resumed.step()
            self.assertTrue(any(float(row["step"])==2 for row in resumed.state.values()))
            torch.save({"schema":"grid"},path)
            with self.assertRaises(ValueError):load(path)
