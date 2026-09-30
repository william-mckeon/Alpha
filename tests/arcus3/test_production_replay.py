"""GPU integration: migration and a subsequent batch must preserve exact updates."""
import os,tempfile,unittest,copy
from pathlib import Path

class ProductionReplay(unittest.TestCase):
    def test_checkpoint_migration_and_batch_replay(self):
        if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1':self.skipTest('Docker CUDA only')
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from arcus3.model import expand
        from arcus3.adapters import train_added_experts
        from arcus3.backbone_adaptation import update
        from arcus3.distillation import targets
        from arcus3.expanded_checkpoint import save,restore
        from arcus3.production import transition,identity
        from arcus3.config import read
        from arcus3.checkpoint import digest
        from baby_arcus.gpu_job_control import gpu_job
        with gpu_job(),tempfile.TemporaryDirectory() as temp:
            torch.manual_seed(2101);torch.use_deterministic_algorithms(True)
            torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
            model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)).cuda().eval()
            rows=[{'input_ids':[1,2,3,4],'labels':[-100,2,3,4]},{'input_ids':[5,6,7,8,9],'labels':[-100,6,7,8,9]}]
            with torch.no_grad():teacher=[targets(model(torch.tensor([r['input_ids']],device='cuda')).logits[0,:-1],4) for r in rows]
            expand(model,[0]);train_added_experts(model)
            frozen={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,foreach=False)
            cfg=read('configs/arcus3/backbone_adaptation.json');policy=read('configs/arcus3/production.json')
            update(model,optimizer,rows[0],teacher[0],cfg)
            state={'updates':1,'input_tokens':4,'target_tokens':3,'parent_sha256':'parent','data_sha256':'old','teacher_sha256':'teacher',
                   'config_sha256':'config','campaign':'backbone-adaptation-v1','accumulation_position':0,'evaluation_pending':[],
                   'evaluation_completed':[{'nll':1.0}],'stream':{'offset':12},'batch_complete':True}
            cp=save(Path(temp)/'before',model,optimizer,state)
            receipt={'checkpoint_sha256':digest(cp/'manifest.json'),'old_data_sha256':'old','new_data_sha256':'next',
                     'new_teacher_sha256':'nextteacher','policy_sha256':identity(policy),'batch_id':'batch-2'}
            # Uninterrupted expected update.
            update(model,optimizer,rows[1],teacher[1],cfg)
            expected={n:p.clone() for n,p in model.named_parameters() if p.requires_grad}
            expected_opt=copy.deepcopy(optimizer.state_dict())
            restored=restore(cp,model,optimizer,'parent','old','config')
            migrated=transition(restored,receipt,digest(cp/'manifest.json'),'next','nextteacher',policy)
            self.assertEqual(migrated['updates'],1);self.assertEqual(migrated['input_tokens'],4)
            self.assertIsNone(migrated['stream'])
            update(model,optimizer,rows[1],teacher[1],cfg)
            self.assertTrue(all(torch.equal(expected[n],p) for n,p in model.named_parameters() if p.requires_grad))
            for key,value in optimizer.state_dict()['state'].items():
                for name,tensor in value.items():self.assertTrue(torch.equal(tensor,expected_opt['state'][key][name]))
            self.assertTrue(all(torch.equal(frozen[n],p) for n,p in model.named_parameters() if not p.requires_grad))
