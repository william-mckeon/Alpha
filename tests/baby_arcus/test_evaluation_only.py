import copy
import tempfile
import unittest
from unittest.mock import patch
from baby_arcus.services.controller import ControllerApplication


class EvaluationOnlyTests(unittest.TestCase):
    def prepare(self,root):
        app=ControllerApplication(root,{})
        options={'preset':'tiny','seed':1,'learning':{'min_samples':8},'max_steps':2,'max_cycles':2,'evaluate':False,'seconds':60}
        app.state.start('trained',60,'checkpoint',{'options':options,'cycles':2,'sample_index':24,
            'evaluation_index':200,'curriculum':{},'gate':{'streak':0},'metrics':[{'updates':2}]})
        app.update('paused')
        return app

    def result(self,app):
        from baby_arcus.evaluation import summary
        job=app.state.value['evaluation_job']
        rows=[{'family':family,'index':i,'episode_id':family+str(i),'success':False}
              for family in ('switch_delivery','clue_search') for i in range(job['start_index'],job['start_index']+200)]
        return {'batch_id':job['batch_id'],'checkpoint_id':job['checkpoint_id'],'complete':True,'provenance':rows,
                'split':'evaluation','start_index':job['start_index'],'requested_per_family':200,
                'results':{f:summary([False]*200) for f in ('switch_delivery','clue_search')}}

    def dispatch(self,app,body,failure=False):
        def evaluate(*args,**kwargs):
            if failure: raise TimeoutError('interrupted')
            return self.result(app)
        with patch('threading.excepthook') as errors,patch.object(app,'check'),patch.object(app,'load_policy'),patch.object(app,'acquire',return_value='lease') as acquire,patch.object(app,'release'),patch.object(app,'evaluate_batch',side_effect=evaluate),patch.object(app,'execute',side_effect=AssertionError('No training operation allowed')):
            response=app('POST','/v1/evaluate',body)
            app.thread.join(5)
            self.assertFalse(app.thread.is_alive())
            self.assertFalse(errors.called)
            self.assertEqual(acquire.call_args.args,('inference',))
        return response

    def test_evaluation_keeps_checkpoint_training_and_gates_frozen(self):
        with tempfile.TemporaryDirectory() as root:
            app=self.prepare(root)
            before=copy.deepcopy(app.state.value)
            try:
                body={'request_id':'review','seconds':60}
                response=self.dispatch(app,body)
                self.assertEqual(app('POST','/v1/evaluate',body),response)
                self.assertEqual(app.state.value['status'],'completed')
                for key in ('checkpoint_id','cycles','sample_index','metrics','gate','curriculum'):
                    self.assertEqual(app.state.value[key],before[key])
                self.assertEqual(app.state.value['evaluation_index'],400)
                self.assertFalse(app.state.value['evaluation'][-1]['gate_applied'])
                self.assertEqual(app.state.value['training_options'],before['options'])
                with patch.object(app,'run'):
                    app('POST','/v1/resume',{'request_id':'return-to-training','seconds':60,'max_cycles':1})
                    app.thread.join(5)
                self.assertNotIn('mode',app.state.value['options'])
                self.assertEqual(app.state.value['options']['preset'],'tiny')
                self.assertEqual(app.state.value['options']['max_cycles'],3)
            finally: app.close()

    def test_controller_restart_resumes_same_population(self):
        with tempfile.TemporaryDirectory() as root:
            app=self.prepare(root)
            self.dispatch(app,{'request_id':'first','seconds':60},failure=True)
            job=copy.deepcopy(app.state.value['evaluation_job'])
            self.assertEqual(app.state.value['status'],'paused')
            app.close()
            app=ControllerApplication(root,{})
            try:
                self.dispatch(app,{'request_id':'continue','seconds':60,'resume':True})
                self.assertEqual(app.state.value['evaluation_job']['batch_id'],job['batch_id'])
                self.assertEqual(app.state.value['evaluation_index'],400)
                self.assertTrue(app.state.value['evaluation_job']['complete'])
            finally: app.close()

    def test_rejects_pending_training_or_different_checkpoint(self):
        from baby_arcus.artifacts import Conflict
        with tempfile.TemporaryDirectory() as root:
            app=self.prepare(root)
            try:
                with self.assertRaises(Conflict):
                    app('POST','/v1/evaluate',{'request_id':'wrong','checkpoint_id':'other'})
                app.update(pending_update={'request_id':'unresolved'})
                with self.assertRaises(Conflict):
                    app('POST','/v1/evaluate',{'request_id':'pending'})
            finally: app.close()
