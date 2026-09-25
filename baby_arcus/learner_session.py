"""One inference owner; CPU residency between GPU leases avoids competing jobs."""
from contextlib import contextmanager
import gc
import json
from pathlib import Path
from baby_arcus.inference_artifact import load


class LearnerSession:
    def __init__(self):
        self.model=None; self.metadata=None; self.key=None

    def invalidate(self):
        self.model=None;self.metadata=None;self.key=None
        gc.collect()

    @contextmanager
    def use(self,root,manifest,device,verify):
        # Caller holds its thread lock AND the cross-process GPU lease.
        root=Path(root)
        key=(str(root.resolve()),json.dumps(manifest,sort_keys=True))
        if self.key!=key:
            self.invalidate()
            model,metadata=load(root,manifest,'cpu')
            verify(metadata)
            self.model=model;self.metadata=metadata;self.key=key
        else:
            verify(self.metadata)
        try:
            self.model.to(device).eval()
            for block in self.model.core.blocks:block.routing_telemetry=False
            yield self.model
        finally:
            # Other authorized jobs may acquire the lease after this request.
            # Never leave a resident GPU model outside that ownership boundary.
            if self.model is not None:
                self.model.to('cpu')
                for block in self.model.core.blocks:
                    block.last_p_soft=None;block.last_aux=None
                    block.last_expert_fraction=None;block.last_expert_overflow=None
                    block.last_compute_fraction=1.0
                self.model.core.last_aux_loss=0.
                self.model.core.last_compute_fraction=1.
            if str(device).startswith('cuda'):
                import torch
                torch.cuda.empty_cache()
