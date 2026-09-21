"""Content-addressed camera examples; never include private toy effects."""
import hashlib,json,os
from pathlib import Path
from baby_arcus.object_observation import detect


def save_example(root,raw):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(raw).hexdigest()
    record=detect(raw)
    # Exclusive writes prevent overwrite; incomplete or corrupt examples fail closed.
    for path,data in ((root/(digest+'.png'),raw),
                      (root/(digest+'.json'),json.dumps(record,sort_keys=True).encode())):
        try:
            with path.open('xb') as stream:
                stream.write(data);stream.flush();os.fsync(stream.fileno())
        except FileExistsError:
            if path.read_bytes()!=data:raise ValueError('Dataset content mismatch')
    return digest


def load_example(root,digest):
    if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid frame hash')
    root=Path(root);raw=(root/(digest+'.png')).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Frame hash mismatch')
    record=json.loads((root/(digest+'.json')).read_text())
    if record!=detect(raw):raise ValueError('Observation mismatch')
    return raw,record


def write_split_manifest(root,seed,counts):
    """Pin disjoint generated scene identities without placing labels in model records."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    sources={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
             for name in ('object_perception_environment.py','playpen_capture.py','visual_model.py')}
    rows={name:{'seed':seed+offset,'examples':count} for (name,offset),count in zip(
        (('train',0),('validation',10000),('final',20000)),counts)}
    record={'schema':'arcus-perception-splits-v1','sources':sources,'splits':rows,
            'inputs':'RGB only','labels':'separate visible-surface masks; no object IDs or hidden effects'}
    path=root/'dataset.json'
    if path.exists() and json.loads(path.read_text())!=record:raise ValueError('Dataset provenance changed')
    path.write_text(json.dumps(record,indent=2));return record
