"""Replace selected donor FFNs without changing attention, cache or tokenizer."""
from arcus3.routing import SelectiveExperts


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
