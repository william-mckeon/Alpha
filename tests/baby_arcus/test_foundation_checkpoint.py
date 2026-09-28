import tempfile
import unittest
from pathlib import Path
import torch
from baby_arcus.foundation_checkpoint import save,load
from baby_arcus.nanotron_adapter import ArcusForTraining
from tests.baby_arcus.foundation_fixtures import tiny_model,State

class CheckpointTests(unittest.TestCase):
    def test_optimizer_rng_and_cursors_resume_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            model=tiny_model();opt=torch.optim.AdamW(model.parameters(),lr=.001);stream=State();cfg={'fixture':True}
            x=torch.randint(2,120,(1,8),device='cuda');mask=torch.ones_like(x,dtype=torch.bool)
            def step():
                opt.zero_grad();ArcusForTraining(model)(x,mask,x.roll(-1,1),mask)['loss'].backward();opt.step()
            step();stream.value['cursor']=7
            pointer=save(d,model,opt,{'updates':1},stream,cfg)
            rng=torch.rand(3,device='cuda');step()
            expected={n:p.detach().clone() for n,p in model.named_parameters()}
            # Interrupted pending output must not affect the last durable pointer.
            (Path(d)/'interrupted.pending').write_bytes(b'incomplete')
            stream.value['cursor']=100
            progress,_=load(d,model,opt,stream,cfg)
            self.assertEqual(stream.value['cursor'],7);torch.testing.assert_close(torch.rand(3,device='cuda'),rng)
            step()
            for n,p in model.named_parameters():torch.testing.assert_close(p,expected[n],rtol=0,atol=0)
            with self.assertRaises(ValueError):load(d,model,opt,stream,{'fixture':False})
            path=Path(d)/(pointer['generation']+'.pt');path.write_bytes(b'bad')
            with self.assertRaises(ValueError):load(d,model,opt,stream,cfg)
