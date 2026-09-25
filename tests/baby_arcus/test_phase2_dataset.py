import tempfile
import unittest
from pathlib import Path
from arcus.tokenizer import get_tokenizer
from baby_arcus.contracts import digest
from baby_arcus.language_stream import documents
from baby_arcus.sft_source_lessons import demonstrate
from baby_arcus.sft_dataset import packing_report
from scripts.build_alpha_phase2_dataset import write_corpus, generated_spec, round_robin


class Phase2DatasetTests(unittest.TestCase):
    def test_balanced_layout_matches_actual_reader_holdout_even_when_validation_is_short(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.jsonl.zst'
            rows=[{'text':'train-'+str(i)} for i in range(24)]
            layout=write_corpus(path, rows, [{'text':'held-out'}])
            training=[text for n,text in documents(path) if n%10]
            self.assertEqual(training,[row['text'] for row in rows])
            self.assertEqual(layout['empty_validation_slots'],2)
            self.assertEqual(list(round_robin([[1,3],[2]])),[1,2,3])

    def test_synthetic_trajectories_have_real_readback_and_explicit_provenance(self):
        base=Path('runs/test2'); base.mkdir(parents=True,exist_ok=True)
        tokenizer=get_tokenizer('o200k_base')
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            for index in (0,1,359):
                spec=generated_spec(index)
                rows,evidence=demonstrate(spec,Path(tmp)/str(index),tokenizer,
                    'alpha:synthetic:'+str(index),'group:'+str(index),'training',
                    {'input_sha256':digest(spec),'lesson':{'source_file_sha256':'a'*64,
                     'source_session_sha256':digest(index)}},adapter='alpha-synthetic-tool-lesson-v1')
                self.assertEqual(evidence['events'][-1]['result']['content'],spec['new'])
                self.assertEqual(evidence['events'][0]['call']['name'],'tool_search')
                self.assertFalse(packing_report(rows,tokenizer)['quarantined'])
                self.assertTrue(all(r['provenance']['adapter']=='alpha-synthetic-tool-lesson-v1' for r in rows))
