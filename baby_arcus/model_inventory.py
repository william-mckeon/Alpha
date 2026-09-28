"""Count unique parameters once and expose registration aliases and expert slices."""

def assert_inventory(model, expected):
    report = inventory(model)
    if report['unique_parameters'] != expected:
        raise ValueError('Unique parameter inventory changed')
    if model.core is not model.body.core or model.core.head.weight is not model.core.token_embed.weight:
        raise ValueError('Shared core or tied body embedding was broken')
    return report

def inventory(model):
    groups = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        item = groups.setdefault(id(parameter), {'names': [], 'shape': list(parameter.shape),
                                                'elements': parameter.numel(), 'dtype':str(parameter.dtype),
                                                'weight_bytes':parameter.numel()*parameter.element_size(), 'trainable': parameter.requires_grad})
        item['names'].append(name)
    modules = []
    for name, module in model.named_modules():
        entry = {'name': name, 'type': type(module).__name__,
                 'children': list(module._modules),
                 'direct_parameter_names': list(module._parameters)}
        if type(module).__name__ == 'BatchedExperts':
            entry['expert_slices'] = [{'expert': i, 'axis': 0,
                'parameters': sum(p[i].numel() for p in module.parameters(recurse=False))}
                for i in range(module.n_experts)]
        modules.append(entry)
    connections=[]
    for name,module in model.named_modules():
        for child in module._modules:
            connections.append({'source':name or '<root>','target':name+'.'+child if name else child,'kind':'registered_child'})
    for group in groups.values():
        if len(group['names'])>1:
            connections.extend({'source':group['names'][0],'target':alias,'kind':'shared_parameter_alias'} for alias in group['names'][1:])
    return {'schema': 'alpha-inventory-v2', 'unique_parameters': sum(g['elements'] for g in groups.values()),
            'unique_weight_bytes':sum(g['weight_bytes'] for g in groups.values()),
            'connections':connections,'connection_note':'Registration edges and shared-tensor aliases, not inferred runtime dataflow; observed routing sequences are reported separately.',
            'parameters': list(groups.values()), 'modules': modules,
            'note': 'Registration tree and shared parameter aliases; runtime connections are in route traces. Routing does not create additional unique weights.'}
