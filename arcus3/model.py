"""Replace selected donor FFNs without changing attention, cache or tokenizer."""
from arcus3.routing import SelectiveExperts

def enable_full_depth(model,layers,capacity=1.0):
    from arcus3.depth import FullDepthGate
    for index in layers:
        block=model.model.layers[index].mlp
        if not isinstance(block,SelectiveExperts) or hasattr(block,'depth_gate'):
            raise ValueError('Expected an expanded FFN without a depth gate')
        block.depth_gate=FullDepthGate(block.router.in_features,block.router.weight.device,capacity)
    return model

def depth_metadata(model):
    gates=[m.depth_gate for m in model.modules() if hasattr(m,'depth_gate')]
    return {'enabled':bool(gates),'capacity':1.0 if gates else None,
            'gate_parameters':sum(p.numel() for g in gates for p in g.parameters()),
            'trainable':any(p.requires_grad for g in gates for p in g.parameters()),
            'snapshots':[g.last_observation for g in gates]}


def expand(model, layers):
    if len(set(layers)) != len(layers) or any(type(i) is not int or not 0 <= i < len(model.model.layers) for i in layers):
        raise ValueError('Invalid selected layers')
    for index in layers:
        block = model.model.layers[index]
        if isinstance(block.mlp, SelectiveExperts):
            raise ValueError('Already converted')
        block.mlp = SelectiveExperts(block.mlp)
    return model


def inventory(model, layers):
    independent = all(a.data_ptr() != b.data_ptr()
                      for i in layers
                      for a, b in zip(model.model.layers[i].mlp.experts[0].parameters(),
                                      model.model.layers[i].mlp.experts[1].parameters()))
    return {'unique_parameters': sum(p.numel() for p in model.parameters()),
            'weight_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
            'independent_experts': independent,
            'tied_embeddings': model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(),
            'routing_counts_last_forward': {str(i): model.model.layers[i].mlp.last_counts for i in layers}}
