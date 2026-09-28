"""Frozen elementary Python tests for the restricted executor only."""
TASKS = {
    'development-0': {'cases': [], 'expected': '0\n1\n2\n3\n4', 'stdout': True},
    'development-1': {'cases': [0, -2, 9], 'expected': [1, -1, 10]},
    'development-2': {'cases': [0, -3, 7], 'expected': [0, -6, 14]},
    'development-3': {'cases': [[], [1, 2, 3], [-2, 5]], 'expected': [0, 6, 3]},
    'development-4': {'cases': [[], [1], [1, 2, 3]], 'expected': [0, 1, 3]},
    'development-5': {'cases': [0, -7, 4], 'expected': [0, 7, 4]},
}
