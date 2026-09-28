"""Streaming computational-use counts, independent of sampled event retention.

Nodes here are routed blocks/experts or invoked modules, not individual neurons.
Repeated token visits are counted again; identities are deduplicated only for coverage.
"""
from collections import Counter


class RouteUsage:
    def __init__(self, model, max_routes=4096):
        self.names = {id(m): n for n, m in model.named_modules()}
        self.blocks = [n for n, m in model.named_modules() if type(m).__name__ == 'MoDEBlock']
        self.max_routes = max_routes
        self.totals = Counter()
        self.nodes, self.routes, self.modules = Counter(), Counter(), Counter()
        self.direct_parameter_uses = Counter()
        self.depth = {}
        self.paths = None
        self.handles = []

    def attach(self, model, active):
        def hook(module, inputs, output):
            if not active(): return
            name = self.names[id(module)] or '<root>'
            self.modules[name] += 1
            # Count direct registered weights once per module invocation, not per token.
            # Functional accesses and methods bypassing Module.__call__ are excluded.
            self.direct_parameter_uses[name] += sum(p.numel() for p in module.parameters(recurse=False))
        self.handles = [m.register_forward_hook(hook) for m in model.modules()]

    def close(self):
        for handle in self.handles: handle.remove()
        self.handles = []

    def observe(self, module, kind, values, phase):
        name = self.names.get(id(module), type(module).__name__)
        if kind == 'depth':
            kept = values['kept'].detach().cpu().tolist()
            slots = values.get('packed_slot')
            slots = slots.detach().cpu().tolist() if slots is not None else None
            if self.blocks and name == self.blocks[0]:
                self.paths = [[[] for _ in row] for row in kept]
                self.totals['forward_passes'] += 1
            self.depth[name] = (kept, slots)
            count = sum(len(row) for row in kept)
            accepted = sum(sum(row) for row in kept)
            self.totals['block_token_visits'] += count
            self.totals['depth_kept_token_visits'] += accepted
            self.totals['depth_skipped_token_visits'] += count - accepted
            self.nodes[phase + ':' + name] += count
        elif kind == 'expert':
            indices = values['expert'].detach().cpu().tolist()
            accepted = values['accepted'].detach().cpu().tolist()
            valid = values['valid'].detach().cpu().tolist()
            hits = Counter(e for row, mask in zip(indices, accepted) for e, ok in zip(row, mask) if ok)
            count = sum(hits.values())
            self.totals['expert_invocations'] += 1
            self.totals['expert_token_visits'] += count
            self.totals['expert_overflow_token_visits'] += sum(sum(row) for row in valid) - count
            slots = count if values['dispatch_mode'] == 'compact' else values['physical_slots']
            self.totals['physical_expert_slots'] += slots
            weights = sum(p[0].numel() for p in module.experts.parameters(recurse=False))
            self.totals['expert_weight_token_uses'] += count * weights
            self.totals['physical_expert_weight_slot_uses'] += slots * weights
            for expert, visits in hits.items():
                self.nodes[phase + ':' + name + ':expert-' + str(expert)] += visits
            parent = name.rsplit('.moe', 1)[0]
            if self.paths is None or parent not in self.depth: return
            kept, packed = self.depth[parent]
            for b, row in enumerate(kept):
                for t, ok in enumerate(row):
                    slot = packed[b][t] if packed is not None else t
                    choice = ('skip' if not ok else
                              str(indices[b][slot]) if accepted[b][slot] else 'overflow')
                    self.paths[b][t].append(parent + '=' + choice)
            if self.blocks and parent == self.blocks[-1]:
                for row in self.paths:
                    for path in row:
                        key = phase + '|' + '>'.join(path)
                        self.totals['token_route_traversals'] += 1
                        if key in self.routes or len(self.routes) < self.max_routes:
                            self.routes[key] += 1
                        else:
                            self.totals['unretained_route_identity_visits'] += 1
                self.paths = None

    def report(self):
        return {'schema':'alpha-computational-use-v1', 'totals':dict(self.totals),
                'node_visits':dict(self.nodes), 'route_traversals':dict(self.routes),
                'distinct_routes_retained':len(self.routes), 'max_route_identities':self.max_routes,
                'module_invocations':dict(self.modules),
                'direct_parameter_invocation_uses':dict(self.direct_parameter_uses),
                'definitions': {
                    'token_route_traversals':'One token position through the routed block stack in one forward pass; repeat passes/calls count again. Includes prefill and sensory passes.',
                    'node_visits':'Block token visits plus accepted expert token assignments, separately labelled. These are component nodes, not scalar neurons.',
                    'expert_weight_token_uses':'Accepted expert token visits times that expert weight count; repeated weights count again. Excludes dense attention, routers and output projections.',
                    'physical_expert_weight_slot_uses':'Expert weights times physical dispatched slots, including padding; not logical visits.',
                    'direct_parameter_invocation_uses':'Direct registered parameters counted once each time Module.__call__ executes. Functional weight accesses and custom method calls are excluded. Do not add to expert token-use counts.',
                    'distinct_routes_retained':'Distinct phase-labelled per-token expert/skip/overflow sequences. A lower bound if unretained_route_identity_visits is nonzero.',
                    'scope':'Observed routed sequences, not all mathematical neuron paths, FLOPs or independently learned model size.'}}


def aggregate(reports, generated_tokens=0):
    totals, nodes, routes, modules, uses = (Counter() for _ in range(5))
    for report in reports:
        for target, key in ((totals,'totals'),(nodes,'node_visits'),(routes,'route_traversals'),
                            (modules,'module_invocations'),(uses,'direct_parameter_invocation_uses')):
            target.update(report.get(key, {}))
    return {'observations':len(reports), 'totals':dict(totals), 'node_visits':dict(nodes),
            'route_traversals':dict(routes), 'distinct_routes_retained':len(routes),
            'module_invocations':dict(modules), 'direct_parameter_invocation_uses':dict(uses),
            'generated_tokens':generated_tokens,
            'expert_weight_token_uses_per_generated_token': totals['expert_weight_token_uses']/generated_tokens if generated_tokens else None,
            'normalization_note':'Generation totals include sensory and prompt-prefill work; division amortizes that work over output tokens. Not decode-only cost.'}
