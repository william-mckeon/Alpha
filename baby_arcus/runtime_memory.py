"""Process/container observations; never labelled as Windows host memory."""
from pathlib import Path
import time


def snapshot(model=None):
    result={'sampled_at':time.time(),'scope':'learner process and Linux container',
            'parameter_count':None,'parameter_bytes':None,'process_rss_bytes':None,
            'container_used_bytes':None,'container_limit_bytes':None,
            'cuda_allocated_bytes':None,'cuda_reserved_bytes':None}
    if model is not None:
        parameters=list(model.parameters())
        result.update(parameter_count=sum(p.numel() for p in parameters),
                      parameter_bytes=sum(p.numel()*p.element_size() for p in parameters),
                      context_tokens=model.core.cfg.max_seq_len)
    try:
        import os
        result['process_rss_bytes']=int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    except (OSError,ValueError,IndexError):pass
    for key,name in [('container_used_bytes','memory.current'),('container_limit_bytes','memory.max')]:
        try:result[key]=int((Path('/sys/fs/cgroup')/name).read_text())
        except (OSError,ValueError):pass
    import torch
    if torch.cuda.is_initialized():
        result['cuda_allocated_bytes']=torch.cuda.memory_allocated()
        result['cuda_reserved_bytes']=torch.cuda.memory_reserved()
    return result
