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
    if name in ('fresh_weighted_even', 'fresh_adjacent_rises', 'fresh_unique_positive_squares'):
        cases = [[],[0],[1,2,2,-4,5],[-3,-2,-1],[2,2,4],[8,-6,3,0]]
        if name == 'fresh_weighted_even':
            instruction = 'Implement solve(values): sum each even integer multiplied by its one-based position in the list.'
            expected = [sum((i+1)*x for i,x in enumerate(xs) if x%2==0) for xs in cases]
        elif name == 'fresh_adjacent_rises':
            instruction = 'Implement solve(values): count adjacent pairs where the second integer is strictly greater than the first.'
            expected = [sum(b>a for a,b in zip(xs,xs[1:])) for xs in cases]
        else:
            instruction = 'Implement solve(values): sum the squares of distinct strictly positive integers.'
            expected = [sum(x*x for x in set(xs) if x>0) for xs in cases]
        return {'instruction':instruction,'initial':'def solve(values):\n    return 0\n',
                'cases':cases,'expected':expected}
    if name not in TASKS:
        raise ValueError('Unknown coding lesson')
    return copy.deepcopy(TASKS[name])


def correction_lessons(folder, tokenizer):
    """Executed train-only teacher trajectories, including failed-test recovery.

    Requires the restricted executor; never evaluates Python on the host.
    Held-out fresh_* tasks are deliberately absent from these demonstrations.
    """
    import json
    from pathlib import Path
    from baby_arcus.coding_environment import CodingEnvironment
    from baby_arcus.coding_tools import CodingTools
    from baby_arcus.sft_source_lessons import action_record
    solutions={'negative_count':'def solve(values):\n    return sum(x < 0 for x in values)\n',
               'absolute_sum':'def solve(values):\n    return sum(abs(x) for x in values)\n'}
    for name,solution in solutions.items():
        workspace=Path(folder)/name
        if workspace.exists():raise ValueError('Teacher workspace must be new')
        tools=CodingTools(CodingEnvironment(workspace,name))
        history=[{'role':'user','content':task(name)['instruction']}]
        calls=[('tool_search',{'query':'read inspect file','limit':1}),('read_file',{'path':'solution.py'}),
               ('tool_search',{'query':'run tests','limit':1}),('run_tests',{}),
               ('tool_search',{'query':'write replace file','limit':1}),
               ('write_file',{'path':'solution.py','content':solution}),('run_tests',{})]
        records=[];events=[]
        for tool,args in calls:
            call={'name':tool,'version':1,'arguments':args}
            record=action_record(history,tools.context.definitions(),call,tokenizer,16384,128,
                                 'alpha:coding-correction','coding-teacher:'+name,'training')
            result=tools.execute(call)
            if tool=='run_tests' and result.get('passed') is not (len(events)>4):
                raise ValueError('Expected failed baseline and successful repaired solution')
            records.append(record);events.append({'call':call,'result':result})
            history.extend([{'role':'assistant','content':json.dumps(call)},
                            {'role':'tool','content':json.dumps(result)}])
        yield {'records':records,'events':events,'task':name,'verified':True}
