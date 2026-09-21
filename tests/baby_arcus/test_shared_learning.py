import unittest
from copy import deepcopy
import torch
from baby_arcus.body_policy import BodyPolicy
from baby_arcus.language_model import LanguageAdapter
from baby_arcus.shared_model import SharedModel
from baby_arcus.shared_learning import update
from baby_arcus.shared_experience import capture
from baby_arcus.services.playroom import PlayroomApplication

class Tokenizer:
    def encode(self,text):return list(text.encode())

def fixture():
    torch.set_num_threads(2);torch.manual_seed(731)
    body=BodyPolicy(lying=True,sitting=True,approach=True)
    model=SharedModel(body,LanguageAdapter(body.cfg.dim,256,16))
    app=PlayroomApplication();app.world.environment.add_toys()
    row=capture(app);app.close();row['hearing']=[{'text':'hello'}];row['eligibility']['training']=True
    return model,row

class SharedLearningTests(unittest.TestCase):
    def test_one_core_and_joint_gradient(self):
        model,row=fixture();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        from arcus.model import ArcusMoDE
        self.assertEqual(sum(isinstance(m,ArcusMoDE) for m in model.modules()),1)
        loss=update(model,optimizer,[row],Tokenizer(),[{'body':0,'text':42,'rest':[0,1,0,0,0],
            'curiosity':[1.],'activity':2,'language_choice':1,'prediction':[0,0,0,0,0]}])
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        for p in (model.visual_input.weight,model.internal_input.weight,model.language.embedding.weight,
                  model.object_input.weight,model.core.blocks[0].attn.wq.weight if hasattr(model.core.blocks[0].attn,'wq') else model.core.token_embed.weight):
            self.assertIsNotNone(p.grad);self.assertGreater(float(p.grad.abs().sum()),0)
    def test_hearing_and_internal_context_change_predictions(self):
        model,row=fixture();a=model([row],Tokenizer())['hidden']
        changed=deepcopy(row);changed['hearing']=[{'text':'different'}];changed['internal'][0]=1
        b=model([changed],Tokenizer())['hidden'];self.assertFalse(torch.allclose(a,b))
    def test_training_requires_eligibility(self):
        model,row=fixture();row['eligibility']['training']=False
        with self.assertRaises(ValueError):update(model,torch.optim.AdamW(model.parameters()),[row],Tokenizer(),[{'body':0}])

    def test_invalid_training_does_not_update_weights(self):
        model,row=fixture();optimizer=torch.optim.AdamW(model.parameters())
        before=model.query.detach().clone()
        for targets in ({},{'typo':1},{'rest':[float('nan')]*5}):
            with self.assertRaises(ValueError):update(model,optimizer,[row],Tokenizer(),[targets])
        self.assertTrue(torch.equal(before,model.query))
        row['internal'][0]=float('nan')
        from baby_arcus.shared_experience import validate
        from baby_arcus.contracts import ContractError
        with self.assertRaises(ContractError):validate(row)
