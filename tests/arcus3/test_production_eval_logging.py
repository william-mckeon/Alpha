"""Evaluator result logging fixture; never loads model weights or reports scores."""
import json
import tempfile
import unittest
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace


class ProductionEvalLoggingTests(unittest.TestCase):
    def test_chat_details_hash_aggregate_and_parquet_roundtrip(self):
        from lighteval.logging.evaluation_tracker import EvaluationTracker, EnhancedJSONEncoder
        from lighteval.logging.info_loggers import DetailsLogger
        from lighteval.models.model_output import GenerativeResponse
        from lighteval.metrics.utils.metric_utils import MetricCategory
        logger=DetailsLogger()
        categories=defaultdict(bool,{MetricCategory.GENERATIVE:True})
        task=SimpleNamespace(has_metric_category=categories)
        query=[{'role':'user','content':'Example fixture question'}]
        doc=SimpleNamespace(query=query,instruction='',ctx='Rendered fixture prompt',
                            num_effective_few_shots=0,num_asked_few_shots=0,
                            get_golds=lambda:['Fixture reference'],specific={})
        response=GenerativeResponse(result=['Fixture response'],input_tokens=[1,2],generated_tokens=[3])
        task_name='custom|fixture|0'
        logger.log(task_name,task,doc,[response],{'fixture_metric':0.0})
        logger.aggregate()
        self.assertEqual(logger.compiled_details[task_name].non_truncated,1)
        self.assertTrue(logger.compiled_hashes[task_name].hash_examples)
        self.assertEqual(doc.query,query)
        with tempfile.TemporaryDirectory() as directory:
            tracker=EvaluationTracker(output_dir=directory,save_details=True,push_to_hub=False)
            tracker.details_logger=logger
            tracker.general_config_logger.model_name='fixture'
            tracker.save()
            paths=list(Path(directory).rglob('*.parquet'))
            self.assertEqual(len(paths),1)
            self.assertTrue(paths[0].stat().st_size>0)
            serialized=json.loads(json.dumps(tracker.generate_final_dict(),cls=EnhancedJSONEncoder))
            self.assertEqual(serialized['summary_tasks']['custom:fixture:0']['non_truncated'],1)
