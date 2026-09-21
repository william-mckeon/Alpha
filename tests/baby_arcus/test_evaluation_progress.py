import time
import unittest
from unittest.mock import patch
from baby_arcus.services.evaluator import EvaluatorApplication

class EvaluationProgressTests(unittest.TestCase):
    def test_live_http_process_kill_retains_completed_episode(self):
        import json
        import subprocess
        import sys
        import tempfile
        from concurrent.futures import ThreadPoolExecutor
        from baby_arcus.transport import Client,RemoteError
        from baby_arcus.evaluation import batch_identity
        code="""
import json,sys,time,uuid
from baby_arcus.services import evaluator
from baby_arcus.transport import serve
def diagnostic_episode(*args):
    time.sleep(.3)
    return [],{'success':False,'episode_id':uuid.uuid4().hex}
evaluator.episode=diagnostic_episode
app=evaluator.EvaluatorApplication('http://unused','http://unused',root=sys.argv[1])
server=serve('127.0.0.1',0,app)
print(json.dumps({'port':server.server_address[1]}),flush=True)
server.serve_forever()
"""
        processes=[]
        def start(root):
            process=subprocess.Popen([sys.executable,'-c',code,root],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            processes.append(process)
            line=process.stdout.readline()
            if not line:
                raise RuntimeError(process.stderr.read())
            return process,Client('http://127.0.0.1:'+str(json.loads(line)['port']),timeout=5,attempts=1)
        with tempfile.TemporaryDirectory() as root:
            try:
                process,client=start(root)
                body={'split':'evaluation','episodes':2,'start_index':0,'deadline':time.time()+30,
                      'checkpoint_id':'diagnostic','lease_id':'diagnostic'}
                route='/v1/evaluations/'+batch_identity(body)
                with ThreadPoolExecutor(1) as pool:
                    running=pool.submit(client.request,'POST','/v1/evaluate',body)
                    end=time.monotonic()+5
                    while True:
                        try:
                            receipt=client.request('GET',route)
                            if receipt['provenance']:
                                break
                        except RemoteError:
                            pass
                        if time.monotonic()>end:
                            self.fail('No live evaluation progress')
                        time.sleep(.02)
                    retained=receipt['provenance'][0]['episode_id']
                    process.terminate()
                    process.wait(timeout=5)
                    with self.assertRaises(RemoteError):
                        running.result(timeout=6)
                _,restarted=start(root)
                result=restarted.request('POST','/v1/evaluate',{**body,'lease_id':'replacement'})
                self.assertTrue(result['complete'])
                self.assertEqual(result['provenance'][0]['episode_id'],retained)
                self.assertEqual(len(result['provenance']),4)
                self.assertEqual(len({(r['family'],r['index']) for r in result['provenance']}),4)
                self.assertEqual(restarted.request('GET',route),result)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=5)
                    process.stdout.close()
                    process.stderr.close()

    def test_durable_batch_resumes_after_restart_and_completed_retry_is_cached(self):
        import tempfile
        from baby_arcus.evaluation import batch_identity
        with tempfile.TemporaryDirectory() as root:
            app=EvaluatorApplication('http://unused','http://unused',root=root)
            body={'split':'evaluation','episodes':2,'start_index':8,'deadline':time.time()+60,
                  'checkpoint_id':'checkpoint','lease_id':'first'}
            with patch('baby_arcus.services.evaluator.episode',side_effect=[([],{'success':True,'episode_id':'retained'}),TimeoutError('disconnect')]):
                _,partial=app('POST','/v1/evaluate',body)
            self.assertEqual(partial['results']['switch_delivery']['episodes'],1)
            restarted=EvaluatorApplication('http://unused','http://unused',root=root)
            self.assertEqual(restarted('GET','/v1/evaluations/'+batch_identity(body),None)[1],partial)
            visited=[]
            def complete(*args):
                visited.append((args[5],args[6]))
                return [],{'success':False,'episode_id':args[5]+str(args[6])}
            with patch('baby_arcus.services.evaluator.episode',side_effect=complete):
                _,result=restarted('POST','/v1/evaluate',{**body,'lease_id':'replacement','deadline':time.time()+120})
            self.assertEqual(visited,[('switch_delivery',9),('clue_search',8),('clue_search',9)])
            self.assertTrue(result['complete'])
            self.assertEqual(result['results']['switch_delivery']['wins'],1)
            self.assertEqual(len(result['provenance']),4)
            with patch('baby_arcus.services.evaluator.episode',side_effect=AssertionError('must not rerun')):
                self.assertEqual(restarted('POST','/v1/evaluate',body)[1],result)

    def test_controller_recovers_receipt_after_lost_post_response(self):
        import tempfile
        from baby_arcus.services.controller import ControllerApplication
        from baby_arcus.curriculum import Curriculum
        from baby_arcus.evaluation import batch_identity
        with tempfile.TemporaryDirectory() as root:
            app=ControllerApplication(root,{'evaluator':'http://unused'})
            app.state.start('run',60,'checkpoint')
            command={'checkpoint_id':'checkpoint','split':'evaluation','episodes':200,'start_index':0,
                     'difficulty':{'switch_delivery':2,'clue_search':2}}
            recovered={'batch_id':batch_identity(command),'complete':False,'results':{'switch_delivery':{'episodes':3}},'provenance':[1,2,3]}
            with patch.object(app,'request',side_effect=TimeoutError('wire')),patch('baby_arcus.services.controller.Client.request',return_value=recovered):
                with self.assertRaises(TimeoutError):
                    app.evaluate_batch('lease','evaluation',200,Curriculum())
            self.assertEqual(app.state.value['evaluation'][-1]['results']['switch_delivery']['episodes'],3)
            complete={**recovered,'complete':True}
            with patch.object(app,'request',side_effect=TimeoutError('lost final response')),patch('baby_arcus.services.controller.Client.request',return_value=complete):
                self.assertEqual(app.evaluate_batch('lease','evaluation',200,Curriculum(),index=0),complete)
            self.assertEqual(len(app.state.value['evaluation']),1)
            self.assertIsNone(app.state.value['pending_evaluation'])
            app.close()

    def test_missing_batch_receipt_is_explicit_without_fabricated_denominator(self):
        import tempfile
        from baby_arcus.services.controller import ControllerApplication
        from baby_arcus.curriculum import Curriculum
        with tempfile.TemporaryDirectory() as root:
            app=ControllerApplication(root,{})
            app.state.start("run",60,"checkpoint")
            with patch.object(app,"request",side_effect=TimeoutError("network")):
                with self.assertRaises(TimeoutError):
                    app.evaluate_batch("lease","evaluation",200,Curriculum())
            result=app.state.value["evaluation"][-1]
            self.assertFalse(result["complete"])
            self.assertEqual(result["results"],{})
            self.assertIn("partial totals unknown",result["reason"])
            app.close()

    def test_partial_batch_retains_provenance_without_inventing_failures(self):
        app=EvaluatorApplication("http://127.0.0.1:1","http://127.0.0.1:2")
        body={"split":"practice","episodes":50,"start_index":0,"deadline":time.time()+10,
              "checkpoint_id":"checkpoint","lease_id":"lease"}
        with patch("baby_arcus.services.evaluator.episode",side_effect=[([],{"success":True,"episode_id":"one"}),TimeoutError("interrupt")]):
            _,result=app("POST","/v1/evaluate",body)
        self.assertFalse(result["complete"])
        self.assertEqual(result["results"]["switch_delivery"]["episodes"],1)
        self.assertNotIn("clue_search",result["results"])
        self.assertEqual(result["provenance"][0]["index"],0)
        self.assertEqual(result["requested_per_family"],50)
