"""Continuous sampled motor experience and disjoint corpus windows for Test 2."""
import copy
from pathlib import Path

from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.shared_experience import capture
from baby_arcus.standing_environment import StandingEnvironment
from baby_arcus.lying_environment import LyingEnvironment
from baby_arcus.sitting_environment import SittingEnvironment
from baby_arcus.language_stream import documents


ENVIRONMENTS = {'standing': StandingEnvironment, 'lying': LyingEnvironment,
                'sitting': SittingEnvironment}


class MotorStream:
    """Resume simulator state by replaying committed actions, without learning twice."""
    def __init__(self, state):
        self.state = state
        self.apps = {}
        self.envs = {}

    def environment(self, family):
        record = self.state.setdefault(family, {'episode': 0, 'actions': [], 'successes': 0})
        if family not in self.envs:
            env = ENVIRONMENTS[family](seed=3102101 + record['episode'] * 1009)
            for action in record['actions']:
                _, _, done = env.step(action)
                if done:
                    raise ValueError('Committed motor trace contains a finished episode')
            app = PlayroomApplication()
            app.world = env.session
            self.apps[family], self.envs[family] = app, env
        return self.envs[family], self.apps[family]

    def observe(self, family):
        env, app = self.environment(family)
        row = capture(app)
        row['objects'], row['hearing'] = [], []
        row['object_source'] = 'none'
        row['session'] = f'sustained:training:{family}:{self.state[family]["episode"]}'
        row['lesson_provenance'] = {'split': 'training', 'family': family}
        row['eligibility']['training'] = True
        return row

    def advance(self, family, action):
        env, app = self.environment(family)
        _, reward, done = env.step(action)
        record = self.state[family]
        record['actions'].append(action)
        result = {'reward': reward, 'done': done, 'success': env.success, 'steps': env.steps}
        if done:
            record['episode'] += 1
            record['successes'] += int(env.success)
            record['actions'] = []
            app.close()
            del self.apps[family], self.envs[family]
        return result

    def close(self):
        for app in self.apps.values():
            app.close()


def corpus_windows(manifest, tokenizer, cursor, window_tokens=64, *, cache_root=None, tokenizer_identity=None, cancelled=lambda:False, cache_max_bytes=1024**3, cache_reserve=lambda size:None):
    """Checkpoint-owned cursor. Every tenth document stays exclusively held out."""
    if type(window_tokens) is not int or not 1 <= window_tokens <= 65536:
        raise ValueError('Invalid corpus context window')
    if cache_root is not None:
        if not tokenizer_identity:raise ValueError('Explicit tokenizer identity required')
        from baby_arcus.indexed_corpus import corpus_windows as indexed_windows
        yield from indexed_windows(manifest,tokenizer,cursor,window_tokens,cache_root,tokenizer_identity,cancelled,cache_max_bytes,cache_reserve)
        return
    for file_index, entry in enumerate(manifest['files']):
        explicit = manifest.get('schema') == 'alpha-coding-corpus-v3' or (manifest.get('schema')=='alpha-coding-corpus-v4' and entry.get('split')!='document-holdout')
        if explicit and entry.get('split') != 'training':
            continue
        if file_index < cursor.get('file', 0):
            continue
        path = Path(manifest['root']) / entry['path']
        stat = path.stat()
        if stat.st_size != entry['size'] or stat.st_mtime_ns != entry['mtime_ns']:
            raise ValueError('Corpus source changed')
        for number, text in documents(path):
            if file_index == cursor.get('file', 0) and number < cursor.get('document', 0):
                continue
            if not explicit and number % 10 == 0:
                continue
            tokens = tokenizer.encode(text)
            start = cursor.get('token', 0) if (file_index, number) == (cursor.get('file', 0), cursor.get('document', 0)) else 0
            for offset in range(start, len(tokens)-1, window_tokens):
                ids = tokens[offset:offset+window_tokens+1]
                after = {'file': file_index, 'document': number, 'token': offset+window_tokens}
                yield ids, copy.deepcopy(after)
