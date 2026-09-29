import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from scripts.publish_alpha_3 import staged_weights,upload_metadata

class PublicationResumeTests(unittest.TestCase):
    def test_metadata_matches_installed_api(self):
        import inspect
        from huggingface_hub import HfApi
        class API:
            def upload_folder(self,**kwargs):
                inspect.signature(HfApi.upload_folder).bind(None,**kwargs)
                return kwargs
        result=upload_metadata(API(),'repo',Path('.'),['manifest.json'])
        self.assertEqual(result['allow_patterns'],['manifest.json'])
    def test_skip_verified_and_commit_remaining_sequentially(self):
        class API:
            def __init__(self):self.files=[NS(rfilename='a.safetensors',lfs=NS(sha256='a'))];self.calls=[]
            def model_info(self,*a,**kw):return NS(private=True,siblings=self.files)
            def upload_file(self,**kw):
                self.calls.append(kw['path_in_repo']);self.files.append(NS(rfilename=kw['path_in_repo'],lfs=NS(sha256='b')));return NS(oid='revision')
        with tempfile.TemporaryDirectory() as root:
            api=API();staged_weights(api,'repo',Path(root),['a.safetensors','b.safetensors','manifest.json'],{'a.safetensors':{'sha256':'a'},'b.safetensors':{'sha256':'b'}})
            self.assertEqual(api.calls,['b.safetensors'])

    def test_public_repo_rejected_before_write(self):
        class API:
            def model_info(self,*a,**kw):return NS(private=False)
        with self.assertRaises(ValueError):staged_weights(API(),'repo',Path('.'),[],{})
