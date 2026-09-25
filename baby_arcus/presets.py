"""Explicit Baby configurations; checkpoint metadata stores the exact shape."""
from arcus.model_config import ModelConfig
from baby_arcus.vocabulary import SIZE

def configuration(name="baby-125m"):
    shared = dict(vocab_size=SIZE,capacity=1.0,capacity_factor=1.0,lb_loss_weight=0.01)
    if name == "tiny":
        return ModelConfig(**shared,dim=32,n_heads=2,n_kv_heads=1,head_dim=16,n_layers=2,
                           expert_hidden=64,n_experts=2,max_seq_len=256)
    if name == "baby-128m-cap4":
        shared["capacity_factor"] = 4.0
        return ModelConfig(**shared, dim=512, n_heads=8, n_kv_heads=2, head_dim=64,
                           n_layers=8, expert_hidden=1952, n_experts=4, max_seq_len=16384)
    if name == "baby-256m-cap4":
        shared["capacity_factor"] = 4.0
        return ModelConfig(**shared, dim=512, n_heads=8, n_kv_heads=2, head_dim=64,
                           n_layers=15, expert_hidden=2368, n_experts=4, max_seq_len=16384)
    if name in ("baby-125m","baby-125m-cap4"):
        if name=="baby-125m-cap4":
            shared["capacity_factor"]=4.0
        return ModelConfig(**shared,dim=512,n_heads=8,n_kv_heads=2,head_dim=64,n_layers=8,
                           expert_hidden=2432,n_experts=4,max_seq_len=512)
    raise ValueError("Unknown Baby preset")
