import unittest,tempfile
from pathlib import Path
import torch
from torch import nn
from baby_arcus.visual_model import ObjectPerceptionAdapter,SpatialObjectPerceptionAdapter
from baby_arcus.object_perception_environment import example
from baby_arcus.object_observation import learned_observation
import numpy as np

class Core(nn.Module):
    def __init__(self):
        super().__init__();self.blocks=nn.ModuleList();self.layer=nn.Linear(16,16)
        from types import SimpleNamespace
        self.cfg=SimpleNamespace(capacity=.25)
    def trunk_embedded(self,x):return self.layer(x)

class LearningTests(unittest.TestCase):
    def test_gradient_and_reload(self):
        torch.set_num_threads(2);core=Core().requires_grad_(False);model=ObjectPerceptionAdapter(16)
        pixels=torch.rand(1,3,96,96);out=model(core,pixels)
        self.assertEqual(tuple(out.shape),(1,4,96,96));out.square().mean().backward()
        self.assertGreater(float(model.project.weight.grad.abs().sum()),0)
        self.assertIsNone(core.layer.weight.grad)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'model.pt';torch.save(model.state_dict(),path)
            restored=ObjectPerceptionAdapter(16);restored.load_state_dict(torch.load(path,weights_only=True))
            self.assertTrue(torch.equal(out,restored(core,pixels)))
    def test_seeded_labels_separate_from_pixels(self):
        x,y,meta=example(123,2);again=example(123,2)
        np.testing.assert_array_equal(x,again[0]);np.testing.assert_array_equal(y,again[1])
        self.assertEqual(x.shape,(3,96,96));self.assertTrue(set(np.unique(y))<={0,1,2,3})
    def test_same_color_is_not_identity(self):
        pixels=np.ones((3,96,96),dtype=np.float32)*.5;labels=np.ones((96,96),dtype=np.int64)
        labels[10:20,10:20]=3;labels[40:50,40:50]=3
        result=learned_observation(pixels,labels)
        self.assertEqual(len(result['objects']),2)
        self.assertEqual(result['objects'][0]['mean_rgb'],result['objects'][1]['mean_rgb'])

    def test_spatial_decoder_has_core_gradient_path(self):
        core=Core().requires_grad_(False);model=SpatialObjectPerceptionAdapter(16)
        out=model(core,torch.rand(1,3,96,96));out.square().mean().backward()
        self.assertGreater(float(model.project.weight.grad.abs().sum()),0)
        self.assertIsNone(core.layer.weight.grad)

class GpuQualificationTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA parity needs GPU')
    def test_candidate_forward_and_cpu_parity(self):
        from baby_arcus.object_perception_learning import load
        import json
        cfg=json.loads(Path('configs/baby_arcus/object_perception.json').read_text())
        if not (Path(cfg['output'])/'perception.pt').exists():self.skipTest('Candidate not trained')
        cfg,body,model,device=load('configs/baby_arcus/object_perception.json',qualified=False)
        x=torch.tensor(example(991,2)[0])[None].to(device)
        with torch.no_grad():gpu=model(body.core,x).cpu()
        body.cpu();model.cpu()
        with torch.no_grad():cpu=model(body.core,x.cpu())
        self.assertTrue(torch.isfinite(gpu).all())
        self.assertLess(float((gpu-cpu).abs().max()),.02)
