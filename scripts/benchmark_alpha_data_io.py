"""Read-only 39k receipt measurements plus synthetic indexed-corpus timing."""
import io
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from baby_arcus.shared_checkpoint import read_data,digest,construct
    from baby_arcus.training_receipt_journal import pack,unpack
    from baby_arcus.gpu_job_control import gpu_job
    from baby_arcus.runtime_contract import require_gpu
    from baby_arcus.language_stream import inventory
    from baby_arcus.sustained_curriculum import corpus_windows
    from arcus.tokenizer import get_tokenizer
    import zstandard
    require_gpu();torch.set_num_threads(2)
    root=Path('/parent');manifest=json.loads((root/'candidate.json').read_text())
    if manifest['updates']!=39000:raise ValueError('Expected retained 39k snapshot')
    source=root/(manifest['generation']+'.pt')
    if digest(source)!=manifest['sha256']:raise ValueError('Checkpoint hash mismatch')
    report={'candidate':manifest,'complete':False,'training':False}
    with gpu_job():
        data=read_data(source);receipts=data['progress']['receipts']
        start=time.perf_counter();journal=pack(receipts);elapsed=time.perf_counter()-start
        assert unpack(journal)==receipts
        raw=io.BytesIO();torch.save(receipts,raw)
        compact=io.BytesIO();torch.save(journal,compact)
        report['receipts']={'count':len(receipts),'original_archive_bytes':len(raw.getvalue()),
            'compressed_archive_bytes':len(compact.getvalue()),'compression_seconds':elapsed,'roundtrip_equal':True}
        model=construct(data,manifest,'cuda').eval();del data,receipts
        tokenizer=get_tokenizer('o200k_base')
        with torch.inference_mode():
            ids=torch.tensor([tokenizer.encode('Arcus is learning to write code.')],device='cuda')
            assert bool(torch.isfinite(model.language(model.core,ids,last_only=True)).all())
        report['cuda_language_inference_finite']=True
        corpus=Path('/evidence/fixture-corpus');corpus.mkdir()
        text='\n'.join(json.dumps({'text':f'def function_{i}(value): return value + {i}\n'*20}) for i in range(1500))
        (corpus/'synthetic.zst').write_bytes(zstandard.ZstdCompressor().compress(text.encode()))
        manifest_data=inventory(corpus,['*.zst']);cache=Path('/evidence/corpus-index')
        options={'cache_root':cache,'tokenizer_identity':{'encoding':'o200k_base','version':'0.14.0'}}
        start=time.perf_counter();expected=list(corpus_windows(manifest_data,tokenizer,{},64));legacy=time.perf_counter()-start
        start=time.perf_counter();indexed=list(corpus_windows(manifest_data,tokenizer,{},64,**options));cold=time.perf_counter()-start
        assert indexed==expected
        offset=len(indexed)*3//4;cursor=indexed[offset][1]
        start=time.perf_counter();a=list(corpus_windows(manifest_data,tokenizer,cursor,64));legacy_resume=time.perf_counter()-start
        start=time.perf_counter();b=list(corpus_windows(manifest_data,tokenizer,cursor,64,**options));warm=time.perf_counter()-start
        assert a==b
        report['synthetic_corpus']={'windows':len(indexed),'legacy_full_seconds':legacy,'index_build_and_full_seconds':cold,
            'legacy_resume_seconds':legacy_resume,'indexed_resume_seconds':warm,'resume_equal':True}
        report['checkpoint_unchanged']=digest(source)==manifest['sha256'];report['complete']=report['checkpoint_unchanged']
    Path('/evidence/io-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
