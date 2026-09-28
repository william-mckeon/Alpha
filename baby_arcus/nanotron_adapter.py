"""Single-rank Nanotron model/1F1B engine adapter; retains Arcus tensor sharing.

Uses upstream's pipeline engine, not its dense-model DistributedTrainer builder.
Multi-rank/TP/PP support is deliberately rejected, not silently emulated.
"""
import torch
from nanotron.models import NanotronModel
from nanotron.parallel.pipeline_parallel.engine import OneForwardOneBackwardPipelineEngine
from baby_arcus.routing_trace import freeze_forward, resume_forward, route_phase


def create_model(cfg, device='cuda'):
    from baby_arcus.runtime_contract import require_gpu
    require_gpu()
    from arcus.model_config import ModelConfig
    from baby_arcus.body_policy import BodyPolicy
    from baby_arcus.language_model import LanguageAdapter
    from baby_arcus.shared_continuity_model import ContinuityModel
    body = BodyPolicy(ModelConfig(**cfg['body_config']), lying=True, sitting=True, approach=True)
    model = ContinuityModel(body, LanguageAdapter(body.cfg.dim, cfg['vocab_size'], cfg['text_dim']), 11)
    model.integrated_motor = True
    model.experiment_depth_capacity = body.cfg.capacity
    model.core.gradient_checkpointing = True
    from baby_arcus.model_inventory import assert_inventory
    assert_inventory(model, cfg['required_unique_parameters'])
    return model.to(device)


class ArcusForTraining(NanotronModel):
    def __init__(self, model, chunk_size=32):
        super().__init__()
        self.model, self.chunk_size = model, chunk_size
        self.input_pp_rank = self.output_pp_rank = 0

    def init_model_randomly(self, config):
        raise RuntimeError('Initialize through create_model exactly once; never reset a resumed learner')

    def forward(self, input_ids, input_mask, label_ids, label_mask):
        if input_ids.shape != label_ids.shape or not bool(input_mask.all()) or not bool(label_mask.all()):
            raise ValueError('This pretraining adapter requires full, unpadded equal-length windows')
        resume_forward()
        with route_phase('foundation-language'):
            residual = self.model.language.embedding.weight.new_zeros((input_ids.shape[0], self.model.language.embedding.embedding_dim))
            nll = self.model.language.loss(self.model.core, input_ids, label_ids, residual, self.chunk_size)
            aux = self.model.core.last_aux_loss
        freeze_forward()
        return {'loss': nll + aux, 'nll': nll.detach(), 'router_loss': aux.detach()}


def train_microbatches(adapter, batches, count, process_group):
    if process_group.size() != 1:
        raise ValueError('Only validated single-GPU execution is supported')
    # 1F1B frees each graph before the next microbatch, unlike AFAB at 64 accumulations.
    engine = OneForwardOneBackwardPipelineEngine()
    return engine.train_batch_iter(adapter, process_group, batches, count, None)
