"""LangGraph with durable per-node receipts and idempotent tool execution."""
import json
from pathlib import Path
import sqlite3
import time
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from baby_arcus.model_adapter import runnable


class State(TypedDict, total=False):
    id: str
    observation: dict
    decision: dict
    outcome: dict
    after: dict


class InteractionGraph:
    def __init__(self, path, observe, infer, execute, accept, limit=10000):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(path)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS nodes (id TEXT, node TEXT, payload TEXT, PRIMARY KEY(id,node))')
        self.db.execute('CREATE TABLE IF NOT EXISTS node_metrics (id TEXT, node TEXT, seconds REAL, PRIMARY KEY(id,node))')
        model = runnable(infer)
        def cached(name, callback):
            def node(state):
                old = self.db.execute('SELECT payload FROM nodes WHERE id=? AND node=?', (state['id'], name)).fetchone()
                if old:
                    return json.loads(old[0])
                if self.db.execute('SELECT COUNT(*) FROM nodes').fetchone()[0] >= limit * 5:
                    raise RuntimeError('Graph journal full')
                started = time.monotonic()
                result = callback(state)
                with self.db:
                    self.db.execute('INSERT INTO nodes VALUES (?,?,?)', (state['id'], name, json.dumps(result)))
                    self.db.execute('INSERT INTO node_metrics VALUES (?,?,?)', (state['id'], name, time.monotonic()-started))
                return result
            return node
        builder = StateGraph(State)
        builder.add_node('observe', cached('observe', lambda s: {'observation': observe()}))
        builder.add_node('decide', cached('decide', lambda s: {'decision': model.invoke(s['observation'])}))
        builder.add_node('act', cached('act', lambda s: {'outcome': execute(s['id'], s['decision'], s['observation'])}))
        builder.add_node('outcome', cached('outcome', lambda s: {'after': observe()}))
        def receipt(state):
            accept(state)
            return {}
        builder.add_node('experience', cached('experience', receipt))
        nodes = [START, 'observe', 'decide', 'act', 'outcome', 'experience', END]
        for a, b in zip(nodes, nodes[1:]):
            builder.add_edge(a, b)
        self.graph = builder.compile()

    def invoke(self, identity):
        return self.graph.invoke({'id': identity})

    def close(self):
        self.db.close()

    def metrics(self):
        connection = sqlite3.connect(self.path)
        try:
            return dict(connection.execute('SELECT node,SUM(seconds) FROM node_metrics GROUP BY node'))
        finally:
            connection.close()
