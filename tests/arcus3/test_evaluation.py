import copy
import math
import unittest
from pathlib import Path
from arcus3.evaluation import aggregate, compatible, load_suite, score, summarize


class EvaluationTests(unittest.TestCase):
    def test_comparison_rejects_incomplete(self):
        from scripts.report_arcus3 import compare_reports
        a={k:'x' for k in ('suite_sha256','tokenizer_revision','settings_sha256','precision','orchestration')}
        a['execution_complete']=False
        with self.assertRaises(ValueError):compare_reports(a,a)
    def test_weighted_loss(self):
        result=aggregate([{'nll_sum':2,'target_tokens':1},{'nll_sum':12,'target_tokens':3}])
        self.assertEqual(result['nll'],3.5)
        self.assertAlmostEqual(result['perplexity'],math.exp(3.5))
        with self.assertRaises(ValueError): aggregate([])

    def test_frozen_suite(self):
        _,rows=load_suite(Path(__file__).resolve().parents[2])
        self.assertEqual(len(rows),36)

    def test_tools_distinguish_parse_schema_semantics_execution(self):
        _,rows=load_suite(Path(__file__).resolve().parents[2])
        item=rows[-6];before=copy.deepcopy(item)
        bad=score(item,'{"name":"echo","arguments":{"text":"wrong"}}')
        self.assertTrue(bad['schema_valid']);self.assertFalse(bad['executed'])
        good=score(item,'{"name":"echo","arguments":{"text":"hello"}}')
        self.assertTrue(good['executed']);self.assertEqual(good['tool_result'],'hello')
        self.assertFalse(score(item,'{"name":"echo","arguments":{"text":7}}')['schema_valid'])
        self.assertFalse(score(item,'```json\n{}\n```')['parseable'])
        self.assertEqual(item,before)

    def test_strict_and_pending(self):
        self.assertFalse(score({'check':'exact','answers':['blue']},'Blue')['task_success'])
        self.assertIsNone(score({'check':'python'},'print(1)')['task_success'])
        self.assertIsNone(score({},'Hi')['human_review']['fluency'])

    def test_comparison_identity(self):
        self.assertFalse(compatible({},{}))
        a={k:'x' for k in ('suite_sha256','tokenizer_revision','settings_sha256','precision','orchestration')}
        self.assertTrue(compatible(a,a))
        self.assertTrue(compatible(a,{**a,'conversion_manifest_sha256':'converted'}))
        self.assertFalse(compatible(a,{**a,'tokenizer_revision':'old-alpha'}))

    def test_truncation_retained(self):
        value=summarize([{'category':'conversation','truncated':True,'metrics':{'task_success':None}}])
        self.assertEqual(value['conversation']['truncated'],1)
        self.assertEqual(value['conversation']['measured'],0)


class CudaMetricTests(unittest.TestCase):
    def test_tiny_model_masking_and_nonmutation(self):
        import os
        if os.environ.get('ARCUS3_CONTROLLED_DOCKER')!='1': self.skipTest('Docker CUDA only')
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from arcus3.evaluation import masked_nll
        from baby_arcus.gpu_job_control import gpu_job
        with gpu_job(),torch.inference_mode():
            torch.manual_seed(4)
            model=LlamaForCausalLM(LlamaConfig(vocab_size=16,hidden_size=16,intermediate_size=32,
                 num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2)).cuda().eval()
            before={k:v.clone() for k,v in model.state_dict().items()}
            ids=torch.tensor([[1,2,3,4]],device='cuda')
            logits=model(ids).logits
            actual=masked_nll(logits,ids,2)
            labels=ids.clone();labels[:,:2]=-100
            expected=model(ids,labels=labels).loss.item()*2
            self.assertAlmostEqual(actual['nll_sum'],expected,places=5)
            self.assertEqual(actual['target_tokens'],2)
            self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__': unittest.main()
