"""Download only pinned official evaluation source; no model or dataset execution."""
import hashlib,json,urllib.request
from pathlib import Path
REV='f54818907404ec3d6bb150357b7d0dea333f1aea'
root=Path('vendor/smollm2');root.mkdir(parents=True,exist_ok=True)
files={}
for name in ('README.md','tasks.py','math_utils.py','requirements.txt','smollm2_base.txt','smollm2_instruct.txt'):
    data=urllib.request.urlopen('https://raw.githubusercontent.com/huggingface/smollm/'+REV+'/text/evaluation/smollm2/'+name).read()
    (root/name).write_bytes(data);files[name]=hashlib.sha256(data).hexdigest()
(root/'manifest.json').write_text(json.dumps({'repository':'huggingface/smollm','revision':REV,'files':files},indent=2))
