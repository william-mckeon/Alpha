"""Disposable CUDA fixtures; never read a production checkpoint."""
import torch
from arcus.model_config import ModelConfig
from baby_arcus.body_policy import BodyPolicy
from baby_arcus.language_model import LanguageAdapter
from baby_arcus.shared_continuity_model import ContinuityModel


def tiny_model():
    from baby_arcus.runtime_contract import require_gpu
    require_gpu()
    cfg=ModelConfig(vocab_size=512,dim=16,n_heads=2,n_kv_heads=1,head_dim=8,n_layers=1,
                    expert_hidden=32,n_experts=2,moe_top_k=1,capacity=1.,capacity_factor=4.,max_seq_len=128)
    model=ContinuityModel(BodyPolicy(cfg,lying=True,sitting=True,approach=True),LanguageAdapter(16,128,8),11)
    model.integrated_motor=True;model.core.gradient_checkpointing=True
    return model.cuda()


class Tokenizer:
    eot_token=1
    def encode(self,text):return [2+ord(c)%100 for c in text]
    def decode(self,ids):return ''.join(chr(32+i%90) for i in ids)


def corpus(path):
    from baby_arcus.foundation_data import open_store,add_document
    with open_store(path,create=True) as db:
        for name in ('a','b'):
            for i in range(100):
                add_document(db,name,f'{name} document {i}. This is a bounded training fixture.',Tokenizer(),{'revision':'fixture-v1','license_url':'fixture'},2)


class State:
    def __init__(self):self.value={'cursor':0}
    def state_dict(self):return self.value.copy()
    def load_state_dict(self,value):self.value=value.copy()
