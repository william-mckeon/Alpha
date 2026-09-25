"""Storage-aware model inventory; aliases are not additional parameters."""
def inventory(model):
    parameters=list(model.parameters())
    storages={}
    for tensor in parameters:
        storage=tensor.untyped_storage()
        storages[(str(tensor.device),storage.data_ptr())]=storage.nbytes()
    return {'parameters':sum(p.numel() for p in parameters),
            'parameter_bytes':sum(p.numel()*p.element_size() for p in parameters),
            'unique_storage_bytes':sum(storages.values()),
            'parameter_tensors':len(parameters),'unique_storages':len(storages),
            'depth_capacity':model.body.cfg.capacity,'context_tokens':model.body.cfg.max_seq_len,
            'expert_count_per_layer':[b.moe.n_experts for b in model.core.blocks]}
