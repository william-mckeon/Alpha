"""Opt-in observations of decisions already made by the model; never reroutes."""
from contextlib import contextmanager
from contextvars import ContextVar

_collector = ContextVar('alpha_route_collector', default=None)
_phase = ContextVar('alpha_route_phase', default='unlabelled')


def load_mapping_config(path='configs/baby_arcus/alpha_mapping.json'):
    import json
    from pathlib import Path
    cfg=json.loads(Path(path).read_text())
    bounds={'max_events':65536,'max_positions':512,'max_new_tokens':128,
            'training_sample_every':65536,'training_max_positions':512,'training_max_events':65536}
    for name,upper in bounds.items():
        value=cfg.get(name)
        if type(value) is not int or not 1<=value<=upper:
            raise ValueError('Invalid mapping bound: '+name)
    return cfg


@contextmanager
def route_phase(name):
    token = _phase.set(name)
    try:
        yield
    finally:
        _phase.reset(token)


def tracing():
    collector = _collector.get()
    return collector is not None and collector.recording


def observe(module, kind, **values):
    collector = _collector.get()
    if collector is not None and collector.recording:
        collector.record(module, kind, values)


def freeze_forward():
    """Called before backward so checkpoint recomputation cannot double count."""
    collector = _collector.get()
    if collector is not None:
        collector.recording = False


def resume_forward():
    """Re-enable tracing for a new accumulation forward, never recomputation."""
    collector = _collector.get()
    if collector is not None:
        collector.recording = True


def observe_gradients(model):
    collector = _collector.get()
    if collector is not None:
        collector.gradients = {name: float(p.grad.detach().float().norm().cpu())
                               for name, p in model.named_parameters()
                               if '.router.' in name and p.grad is not None}


class RoutingTrace:
    def __init__(self, model, max_events=4096, max_positions=256, count_usage=False):
        if not 1 <= max_events <= 65536 or not 1 <= max_positions <= 512:
            raise ValueError('Unbounded trace request')
        self.names = {id(m): name for name, m in model.named_modules()}
        self.max_events, self.max_positions = max_events, max_positions
        self.events, self.gradients = [], {}
        self.dropped_events = 0
        self.recording = True
        from baby_arcus.route_usage import RouteUsage
        self.usage = RouteUsage(model) if count_usage else None
        self.model = model

    def __enter__(self):
        if _collector.get() is not None:
            raise ValueError('Nested route collectors are not supported')
        self.token = _collector.set(self)
        if self.usage: self.usage.attach(self.model, lambda: self.recording)
        return self

    def __exit__(self, *exc):
        if self.usage: self.usage.close()
        _collector.reset(self.token)

    def record(self, module, kind, values):
        if self.usage: self.usage.observe(module, kind, values, _phase.get())
        if len(self.events) >= self.max_events:
            self.dropped_events += 1
            return
        event = {'module': self.names.get(id(module), type(module).__name__),
                 'kind': kind, 'phase': _phase.get(), 'invocation': len(self.events)}
        if kind == 'expert':
            accepted, valid = values['accepted'], values['valid']
            event['logical_accepted_tokens'] = int(accepted.sum().detach().cpu())
            event['overflow_tokens'] = int((valid & ~accepted).sum().detach().cpu())
            event['physical_slots'] = (event['logical_accepted_tokens'] if values['dispatch_mode'] == 'compact' else values['physical_slots'])
        for key, value in values.items():
            if hasattr(value, 'detach'):
                tensor = value.detach()
                event[key + '_shape'] = list(tensor.shape)
                # Batch one is sufficient for token maps; disclose omitted entries.
                if tensor.ndim >= 2:
                    event[key + '_truncated'] = tensor.shape[0] > 1 or tensor.shape[1] > self.max_positions
                    tensor = tensor[:1, :self.max_positions]
                event[key] = tensor.cpu().tolist()
            else:
                event[key] = value
        self.events.append(event)

    def report(self):
        return {'schema': 'alpha-routing-v1', 'events': self.events,
                'usage': self.usage.report() if self.usage else None,
                'dropped_events': self.dropped_events, 'max_positions': self.max_positions,
                'router_gradient_norms': self.gradients,
                'gradient_note': 'Missing means no observed gradient, not a measured zero.',
                'scope': 'Observed module invocations and token routes, not all possible neural paths or equivalent dense size.'}
