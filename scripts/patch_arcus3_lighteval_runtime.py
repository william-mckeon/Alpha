"""Documented compatibility patch: never eagerly load an unused BLEURT model."""
from pathlib import Path
import importlib.util
root=Path(importlib.util.find_spec('lighteval').origin).parent
path=root/'metrics/metrics_sample.py'
text=path.read_text()
start=text.index('class BLEURT:');end=text.index('\n\nclass BLEU:',start)
text=text[:start]+'''class BLEURT:
    def __init__(self):
        pass

    def compute(self, *args, **kwargs):
        raise RuntimeError("BLEURT is outside the pinned Arcus donor suite")
'''+text[end:]
path.write_text(text)
path=root/'data.py'
text=path.read_text()
old='from torch.utils.data.distributed import DistributedSampler, T_co'
if old not in text:raise RuntimeError('Unexpected pinned LightEval data module')
path.write_text(text.replace(old,'from torch.utils.data.distributed import DistributedSampler\nfrom typing import TypeVar\nT_co = TypeVar("T_co", covariant=True)'))
path=root/'tasks/extended/__init__.py'
text=path.read_text()
for name in ('mix_eval','mt_bench','tiny_benchmarks'):
    line=f'    import lighteval.tasks.extended.{name}.main as {name}\n'
    if line not in text:raise RuntimeError('Unexpected extended registry')
    text=text.replace(line,'')
text=text.replace('AVAILABLE_EXTENDED_TASKS_MODULES = [ifeval, tiny_benchmarks, mt_bench, mix_eval]','AVAILABLE_EXTENDED_TASKS_MODULES = [ifeval]')
path.write_text(text)
