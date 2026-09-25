"""Read-only capacity checks before launching Alpha containers; no GPU imports."""
GIB = 1024 ** 3
SERVICES = ('learner', 'playroom', 'review', 'executor', 'worker')


def assess(compose, engine, running, mode='stack', service='learner'):
    if mode not in ('stack', 'job'):
        raise ValueError('Unknown capacity check mode')
    names = SERVICES if mode == 'stack' else (service,)
    specs = compose['services']
    limits = {}
    for name in names:
        spec = specs[name]
        memory, cpus, pids = spec.get('mem_limit'), spec.get('cpus'), spec.get('pids_limit')
        if isinstance(memory, str) and memory.isdecimal():
            memory = int(memory)  # Compose JSON renders byte limits as strings on Windows.
        if (type(memory) is not int or memory <= 0 or not 0 < float(cpus or 0) <= 2
                or type(pids) is not int or not 1 <= pids <= 256):
            raise ValueError('Bounded memory, CPU and PID limits required: ' + name)
        limits[name] = memory
    external = 0
    conflicts = []
    unbounded = []
    for container in running:
        if not container.get('State', {}).get('Running'):
            continue
        host = container['HostConfig']
        labels = container.get('Config', {}).get('Labels') or {}
        managed = (mode == 'stack' and labels.get('com.docker.compose.project') == compose.get('name')
                   and labels.get('com.docker.compose.service') in names)
        if managed:
            continue
        if host.get('DeviceRequests'):
            conflicts.append(container['Id'][:12])
        memory = host.get('Memory', 0)
        if type(memory) is not int or memory <= 0:
            measured = container.get('MeasuredMemoryBytes')
            if type(measured) is not int or measured < 0:
                raise ValueError('Running unbounded container lacks memory measurement: ' + container['Id'][:12])
            # Do not alter unrelated applications. Reserve measured use plus 25%;
            # this is a point-in-time capacity estimate, not a hard global bound.
            memory = (measured * 5 + 3) // 4
            unbounded.append(container['Id'][:12])
        external += memory
    # Leave room for the Docker VM and one bounded child coding sandbox.
    overhead = GIB + (256 * 1024**2 if mode == 'stack' or service == 'learner' else 0)
    required = sum(limits.values()) + external + overhead
    capacity = engine.get('MemTotal', 0)
    return {'schema': 'alpha-runtime-resources-v1', 'mode': mode,
            'service_memory_limits': limits, 'external_limit_bytes': external,
            'reserved_overhead_bytes': overhead, 'required_bytes': required,
            'docker_vm_bytes': capacity, 'gpu_conflicts': conflicts,
            'unbounded_external_containers': unbounded,
            'external_capacity_is_estimate': bool(unbounded),
            'checks': {'fits_docker_vm': required <= capacity,
                       'no_competing_gpu_container': not conflicts},
            'complete': required <= capacity and not conflicts,
            'host_stability_established': False}
