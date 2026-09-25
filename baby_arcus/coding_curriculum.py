"""Small reproducible Python lessons with externally evaluated outcomes."""
TASKS = {
    'negative_count': {'instruction':'Implement solve(values): count strictly negative integers.',
                       'initial':'def solve(values):\n    return len(values)\n',
                       'cases':[[0,-2,3],[],[-1,-4,0],[9]],'expected':[1,0,2,0]},
    'absolute_sum': {'instruction':'Implement solve(values): sum the absolute values of all integers.',
                     'initial':'def solve(values):\n    return sum(values)\n',
                     'cases':[[-3,2],[],[-2,-4],[0,5]],'expected':[5,0,6,5]},
    'positive_sum': {'instruction': 'Implement solve(values): return the sum of positive integers in values.',
                     'initial': 'def solve(values):\n    return sum(values)\n',
                     'cases': [[1,-2,3], [], [-9,-1], [5,2,0]], 'expected': [4,0,0,7]},
    'unique_count': {'instruction': 'Implement solve(values): return the number of distinct values.',
                     'initial': 'def solve(values):\n    return len(values)\n',
                     'cases': [[1,1,2], [], [3,3,3], [4,5,6]], 'expected': [2,0,1,3]},
}


def task(name):
    import copy
    if name not in TASKS:
        raise ValueError('Unknown coding lesson')
    return copy.deepcopy(TASKS[name])
