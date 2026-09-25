import unittest
from copy import deepcopy
from baby_arcus.runtime_resources import assess, GIB


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.compose = {'name': 'arcus-alpha-three-stage', 'services': {
            name: {'mem_limit': int(size*GIB), 'cpus': 1, 'pids_limit': 128}
            for name, size in [('learner',11),('playroom',1),('review',.5),('executor',.25),('worker',.5)]}}

    def test_rejects_original_overcommit_and_accepts_bounded_stack(self):
        engine = {'MemTotal': 16458608640}
        self.assertTrue(assess(self.compose, engine, [])['complete'])
        old = deepcopy(self.compose)
        for name, size in [('learner',12),('playroom',3),('review',1),('worker',2)]:
            old['services'][name]['mem_limit'] = size*GIB
        self.assertFalse(assess(old, engine, [])['complete'])
        for spec in self.compose['services'].values():
            spec['mem_limit'] = str(spec['mem_limit'])
        self.assertTrue(assess(self.compose, engine, [])['complete'])

    def test_counts_external_memory_and_detects_gpu_conflict(self):
        running = [{'Id': 'a'*64, 'State': {'Running': True}, 'Config': {'Labels': {}},
                    'HostConfig': {'Memory': 2*GIB, 'DeviceRequests': [{'Count': -1}]}}]
        report = assess(self.compose, {'MemTotal': 16*GIB}, running)
        self.assertFalse(report['complete'])
        self.assertEqual(report['gpu_conflicts'], ['a'*12])
        with self.assertRaises(ValueError):
            running[0]['HostConfig']['Memory'] = 0
            assess(self.compose, {'MemTotal': 16*GIB}, running)
        running[0]['MeasuredMemoryBytes'] = 128*1024**2
        running[0]['HostConfig']['DeviceRequests'] = []
        report = assess(self.compose, {'MemTotal': 16*GIB}, running)
        self.assertTrue(report['complete'])
        self.assertTrue(report['external_capacity_is_estimate'])

    def test_docker_memory_units(self):
        from scripts.inspect_alpha_resources import memory_bytes
        self.assertEqual(memory_bytes('1.5GiB '), int(1.5*GIB))
        self.assertEqual(memory_bytes('512MiB'), GIB//2)
        with self.assertRaises(ValueError): memory_bytes('unknown')

    def test_stack_does_not_double_count_its_managed_services(self):
        running = [{'Id': 'a'*64, 'State': {'Running': True}, 'Config': {'Labels': {
            'com.docker.compose.project': self.compose['name'], 'com.docker.compose.service': 'learner'}},
                    'HostConfig': {'Memory': 11*GIB, 'DeviceRequests': [{'Count': -1}]}}]
        self.assertTrue(assess(self.compose, {'MemTotal': 16*GIB}, running)['complete'])
        self.assertFalse(assess(self.compose, {'MemTotal': 16*GIB}, running, mode='job')['complete'])


if __name__ == '__main__': unittest.main()
