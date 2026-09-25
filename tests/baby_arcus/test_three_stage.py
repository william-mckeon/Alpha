import copy
import tempfile
import threading
import unittest
from pathlib import Path
import torch
from baby_arcus.coding_tools import DEFINITIONS
from baby_arcus.tool_catalog import ToolCatalog
from baby_arcus.tool_search import search
from baby_arcus.tool_context import ToolContext
from baby_arcus.data_staging import StagingStore
from baby_arcus.staging_graph import graph
from baby_arcus.sft_dataset import windows, validate_window
from baby_arcus.services.data_review import Application
from baby_arcus.transport import serve, Client, RemoteError
from baby_arcus.shared_factory import create
from baby_arcus.shared_curriculum import example
from baby_arcus.shared_objectives import step
from baby_arcus.training_mixture import validate as validate_mixture


class Tokenizer:
    def encode(self,text): return list(text.encode())


def record(split='training'):
    return {'version':1,'source':'fixture:unit','group':'fixture-task-one','split':split,
            'messages':[{'role':'user','content':'What is one plus one?'},
                        {'role':'assistant','content':'Two.'}]}


class DiscoveryTests(unittest.TestCase):
    def test_discovery_and_reuse(self):
        catalog=ToolCatalog(DEFINITIONS); context=ToolContext(catalog)
        call={'name':'run_tests','version':1,'arguments':{}}
        with self.assertRaises(ValueError): context.resolve(call)
        results=search(catalog,'execute tests',limit=1)
        self.assertEqual(results['tools'][0]['name'],'run_tests')
        context.accept(results)
        context.resolve(call); context.resolve(call)

    def test_unavailable_unknown_and_invalid_args(self):
        catalog=ToolCatalog(DEFINITIONS,allowed={'tool_search','read_file'})
        self.assertEqual(search(catalog,'execute tests')['status'],'no_matches')
        for call in ({'name':'run_tests','version':1,'arguments':{}},
                     {'name':'tool_search','version':True,'arguments':{'query':'hi'}},
                     {'name':'read_file','version':1,'arguments':{'path':2}}):
            with self.assertRaises(ValueError): catalog.resolve(call)

    def test_forged_schema_and_budget(self):
        catalog=ToolCatalog(DEFINITIONS); result=search(catalog,'read file')
        changed=copy.deepcopy(result); changed['tools'][0]['description']='forged'
        with self.assertRaises(ValueError): ToolContext(catalog).accept(changed)
        with self.assertRaises(ValueError): ToolContext(catalog,max_bytes=1).accept(result)
        changed=copy.deepcopy(result); changed['catalog_version']='old'
        with self.assertRaises(ValueError): ToolContext(catalog).accept(changed)

    def test_unfamiliar_name_discovery(self):
        defs=copy.deepcopy(DEFINITIONS)
        defs[3]['name']='verify_solution'
        result=search(ToolCatalog(defs),'tests correctness',limit=1)
        self.assertEqual(result['tools'][0]['name'],'verify_solution')


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'store.sqlite'
        self.secret='human-review-fixture-secret-12345'
        self.store=StagingStore(self.path,self.secret,fixture=True)
        self.addCleanup(self.store.close)

    def test_approval_is_separate_from_recommendation(self):
        identity=graph(self.store).invoke({'records':[record()]})['batch_id']
        self.store.recommend(identity,'Looks suitable, awaiting human decision.')
        with self.assertRaises(ValueError): self.store.approved([identity],True)
        with self.assertRaises(PermissionError): self.store.review(identity,'approved','fixture','wrong')
        self.store.review(identity,'approved','fixture reviewer',self.secret)
        self.assertEqual(self.store.approved([identity],True),[record()])
        with self.assertRaises(ValueError): self.store.approved([identity])
        with self.assertRaises(ValueError): self.store.review(identity,'rejected','fixture reviewer',self.secret)

    def test_split_and_store_separation(self):
        self.store.stage([record()])
        with self.assertRaises(ValueError): self.store.stage([record('test')])
        item=record(); item['source']='real:dataset'
        with self.assertRaises(ValueError): self.store.stage([item])
        with self.assertRaises(ValueError): StagingStore(self.path,fixture=False)

    def test_live_http_machine_cannot_approve(self):
        machine='fixture-ingestion-secret-12345'
        server=serve('127.0.0.1',0,Application(self.store,machine))
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            client=Client('http://127.0.0.1:'+str(server.server_port))
            result=client.request('POST','/stage',{'credential':machine,'records':[record()]})
            body={'credential':machine,'batch_id':result['batch_id'],'decision':'approved','reviewer':'fixture'}
            with self.assertRaises(RemoteError) as caught: client.request('POST','/review',body)
            self.assertEqual(caught.exception.status,403)
            body['credential']=self.secret
            self.assertEqual(client.request('POST','/review',body)['decision'],'approved')
        finally: server.shutdown(); server.server_close(); thread.join()


class LearningTests(unittest.TestCase):
    def test_target_masks_exclude_user_and_tool(self):
        item=record()
        item['messages'].append({'role':'tool','content':'tool observation'})
        chunks=list(windows(item,Tokenizer(),128))
        trained=bytes(i for chunk in chunks for i,mask in zip(chunk['ids'][1:],chunk['mask'][1:]) if mask).decode()
        self.assertEqual(trained,'Two.\n</assistant>\n')
        for chunk in chunks: validate_window(chunk,256,128)

    def test_sft_updates_existing_core_and_language(self):
        torch.set_num_threads(2)
        model=create({'depth_capacity':1.,'seed':2101,'preset':'tiny','text_dim':16},256)
        row,_,_=example(0,'training','commands'); row['hearing']=[]; row.pop('language_prefix_ids',None)
        target={'sft':list(windows(record(),Tokenizer()))[0]}
        result=step(model,torch.optim.AdamW(model.parameters(),lr=.00001),Tokenizer(),row,target)
        self.assertGreater(result['gradient_norms']['core'],0)
        self.assertGreater(result['gradient_norms']['language'],0)
        self.assertEqual(result['trained_tokens'],sum(target['sft']['mask'][1:]))
        row['hearing']=[{'text':'Two.'}]
        with self.assertRaises(ValueError): step(model,torch.optim.AdamW(model.parameters()),Tokenizer(),row,target)

    def test_default_real_training_is_paused(self):
        import json
        with self.assertRaises(ValueError): validate_mixture(json.loads(Path('configs/baby_arcus/alpha_three_stage.json').read_text()))

    def test_review_reader_cannot_modify_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'review.sqlite'
            writer=StagingStore(path,'fixture-human-credential-12345',fixture=True)
            identity=writer.stage([record()])
            writer.review(identity,'approved','fixture','fixture-human-credential-12345')
            reader=StagingStore(path,fixture=True,readonly=True)
            try:
                self.assertEqual(reader.approved([identity],True),[record()])
                with self.assertRaises(PermissionError): reader.review(identity,'approved','fixture','fixture-human-credential-12345')
            finally: reader.close(); writer.close()

    def test_container_viewer_requires_local_host_and_origin(self):
        from baby_arcus.services.playroom import viewer_server
        import urllib.request
        server=viewer_server(0,lambda method,path,body:(200,{'ok':True}),host='0.0.0.0')
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        client=Client('http://127.0.0.1:'+str(server.server_port))
        try:
            self.assertTrue(client.request('GET','/')['ok'])
            request=urllib.request.Request(client.base_url,headers={'Host':'evil.invalid'})
            with self.assertRaises(urllib.error.HTTPError) as caught: client.opener.open(request)
            self.assertEqual(caught.exception.code,403)
            request=urllib.request.Request(client.base_url,data=b'{}',headers={'Content-Type':'application/json','Origin':'https://evil.invalid'})
            with self.assertRaises(urllib.error.HTTPError) as caught: client.opener.open(request)
            self.assertEqual(caught.exception.code,403)
        finally: server.shutdown(); server.server_close(); thread.join()


class ExperimentContractTests(unittest.TestCase):
    def test_real_plan_requires_pinned_authorized_gates(self):
        import json
        cfg=json.loads(Path('configs/baby_arcus/alpha_three_stage.json').read_text())
        cfg.update(training_enabled=True,mixture=['embodied','coding_corpus','sft'],token_budget=100)
        with self.assertRaises(ValueError): validate_mixture(cfg)
        cfg['fixture']=True
        validate_mixture(cfg)
        cfg['exhaustion_policy']='repeat'
        with self.assertRaises(ValueError): validate_mixture(cfg)


if __name__ == '__main__': unittest.main()
