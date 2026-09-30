"""Request a graceful update-boundary pause; never claim it is already complete."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.language_stream import atomic_json
from arcus3.config import read

def request(root):
    root=Path(root).resolve()
    if (root/'session.json').is_file():
        session=read(root/'session.json');(root/'pause-training').write_text('User requested session pause\n')
        child=Path(session['active_run']).resolve()
        workspace=Path(__file__).resolve().parents[1]
        if not child.is_relative_to(workspace/'runs'/'arcus3'):raise ValueError('Invalid owned child run')
        if session['mode']=='controller':
            if (child/'session.json').exists():return request(child)
            return {'pause_requested':True,'pause_verified':False,'stage':'controller-startup'}
        if session['mode'] in ('baseline','donor-baseline','teacher-production'):
            (child/'pause-inference').write_text('User requested session pause\n')
            return {'pause_requested':True,'pause_verified':False,'stage':'evaluation'}
        return request(child)
    if not (root/'runtime.json').is_file():raise ValueError('Owned run runtime required')
    runtime=read(root/'runtime.json')
    if runtime.get('mode') not in ('adaptation','adaptation-qualification'):raise ValueError('Not an adaptation run')
    (root/'pause-training').write_text('User requested graceful pause\n')
    return {'pause_requested':True,'pause_verified':False,'container':runtime['container'],
            'next':'Wait for container exit, then verify report and durable checkpoint; do not start inference before exit.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);print(json.dumps(request(p.parse_args().root),indent=2))
