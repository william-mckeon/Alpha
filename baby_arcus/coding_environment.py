"""Repository edits on a scoped workspace; untrusted Python executes only in Docker."""
import json
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
from baby_arcus.coding_curriculum import task

HARNESS = '''import importlib.util,json,sys
inputs=json.loads(sys.stdin.read())
spec=importlib.util.spec_from_file_location("solution","/workspace/solution.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps([module.solve(x) for x in inputs]))
'''


class CodingEnvironment:
    def __init__(self, root, task_name, image='arcus-test2:qualification'):
        raw = Path(root).absolute()
        if any(p.is_symlink() or (hasattr(p,'is_junction') and p.is_junction()) for p in (raw,*raw.parents)):
            raise ValueError('Workspace links are not allowed')
        self.root = raw.resolve()
        base = (Path(__file__).resolve().parents[1]/'runs'/'test2').resolve()
        if base not in self.root.parents or self.root.is_symlink():
            raise ValueError('Coding workspace must be under runs/test2')
        self.task_name, self.task, self.image = task_name, task(task_name), image
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path('solution.py')
        if not path.exists():
            path.write_text(self.task['initial'], encoding='utf-8')

    def path(self, name):
        if not isinstance(name, str) or name != 'solution.py':
            raise ValueError('Only the lesson solution.py is editable/readable')
        path = self.root/name
        if path.is_symlink() or path.resolve().parent != self.root or (path.exists() and path.stat().st_nlink > 1):
            raise ValueError('Workspace escape')
        return path

    def read(self, path):
        file = self.path(path)
        if file.stat().st_size>12000:
            raise ValueError('File exceeds tool budget')
        return {'path': path, 'content': file.read_text(encoding='utf-8')}

    def write(self, path, content):
        from hashlib import sha256
        digest = lambda path: sha256(Path(path).read_bytes()).hexdigest()
        if not isinstance(content,str) or len(content.encode())>12000:
            raise ValueError('Edit exceeds tool budget')
        file = self.path(path)
        before = digest(file)
        file.write_text(content, encoding='utf-8')
        return {'path': path, 'before_sha256': before, 'sha256': digest(file)}

    def tests(self):
        import os
        from baby_arcus.transport import Client
        url = os.environ.get('ALPHA_EXECUTOR_URL')
        token = os.environ.get('ALPHA_EXECUTOR_TOKEN')
        if not url or not token:
            raise RuntimeError('Restricted coding executor URL and credential required; no host execution fallback')
        source = self.read('solution.py')['content']
        return Client(url,token,timeout=65,attempts=1).request('POST','/execute',
                    {'task':self.task_name,'source':source})
