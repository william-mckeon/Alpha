"""Bounded coding episodes; learner inference and review remain separate services."""
import argparse
import json
import os
from pathlib import Path
from baby_arcus.coding_environment import CodingEnvironment
from baby_arcus.coding_tools import CodingTools
from baby_arcus.coding_practice import run_episode
from baby_arcus.trajectory_store import TrajectoryStore
from baby_arcus.shared_curriculum import example
from baby_arcus.transport import Client
from baby_arcus.contracts import identifier


def run(root, episode, task, learner_url, token, steps=4, observation=None, cancelled=lambda: False):
    identifier(episode)
    root = Path(root)
    tools = CodingTools(CodingEnvironment(root/'workspaces'/episode,task))
    store = TrajectoryStore(root/'trajectories.sqlite')
    client = Client(learner_url,token,timeout=180,attempts=1)
    if observation is None:
        raise ValueError('Live practice requires an observation provider; synthetic rows belong only in evaluation')
    def policy(state):
        row = observation()
        return client.request('POST','/coding-infer',dict(state,row=row,max_new_tokens=128))
    try: return run_episode(episode,tools,policy,store,steps,cancelled=cancelled)
    finally: store.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='runs/test2/alpha-coding-practice')
    parser.add_argument('--episode',required=True)
    parser.add_argument('--observation-file',required=True,help='Fresh scoped observation JSON; UI practice supplies live observations directly')
    parser.add_argument('--task',choices=('positive_sum','unique_count'),required=True)
    parser.add_argument('--learner',default='http://learner:8931')
    args=parser.parse_args()
    def observation():
        import time
        row=json.loads(Path(args.observation_file).read_text())
        if not 0 <= time.time()-row.get('captured_at',0) <= 60:
            raise ValueError('Observation expired; use the live playroom practice control')
        return row
    print(json.dumps(run(args.root,args.episode,args.task,args.learner,os.environ['ARCUS_TEST2_TOKEN'],observation=observation)))


if __name__ == '__main__': main()
