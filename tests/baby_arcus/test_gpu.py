"""Measured CUDA capacity gate; skipped on CPU-only test hosts."""
import time
import unittest
import torch
from baby_arcus.checkpoint import seed_everything
from baby_arcus.model import BabyModel,precision
from baby_arcus.presets import configuration

class GpuTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),"CUDA not available")
    def test_cap4_full_context_backward_fits(self):
        seed_everything(19)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model=BabyModel(configuration("baby-125m-cap4")).to("cuda")
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0001)
        contexts=[[1,2,3]+[200+i%49 for i in range(509)] for _ in range(8)]
        with precision(model):
            output=model(contexts)
            loss=output["value"].square().mean()+output["aux"]
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        reserved=torch.cuda.max_memory_reserved()/1024**3
        print({"preset":"baby-125m-cap4","peak_reserved_gib":reserved,"overflow":output["overflow"]})
        self.assertEqual(output["overflow"],0)
        self.assertLess(reserved,torch.cuda.get_device_properties(0).total_memory/1024**3*.85)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA not available")
    def test_initial_model_full_context_adam_step(self):
        seed_everything(19)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        model=BabyModel(configuration("baby-125m")).to("cuda")
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0001)
        contexts=[[1,2,3]+[200+i%49 for i in range(509)] for _ in range(8)]
        with precision(model):
            out=model(contexts)
            objective=sum(out[k].square().mean() for k in ("action","signal","value","cells","inventory","result"))+out["aux"]
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        measured={"device":torch.cuda.get_device_name(),"parameters":sum(p.numel() for p in model.parameters()),
                  "microbatch":8,"context":512,"optimizer":"AdamW","seconds":time.monotonic()-started,
                  "peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,
                  "peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3}
        print(measured)
        self.assertLess(measured["peak_reserved_gib"],torch.cuda.get_device_properties(0).total_memory/1024**3*.85)
        self.assertTrue(torch.isfinite(objective))
