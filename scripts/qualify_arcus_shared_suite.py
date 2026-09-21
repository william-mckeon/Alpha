"""Run the complete immutable candidate qualification; never auto-publish."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--skip-posture',action='store_true')
    parser.add_argument('--live-first',action='store_true',help='Check the complete runtime before the longer frozen evaluations')
    args=parser.parse_args();root=Path(json.loads(Path(args.config).read_text())['root'])
    before=(root/'candidate.json').read_bytes()
    from baby_arcus.shared_qualification import source_snapshot
    sources=source_snapshot()
    commands=[]
    if not args.skip_posture:
        commands.append(('evaluate_arcus_shared.py',['--manifest','candidate.json','--episodes','200','--batch-size','16','--output',str(root/'posture-final')]))
    commands.extend([
        ('evaluate_arcus_shared_transfer.py',['--count','300','--split','confirmation']),
        ('evaluate_arcus_shared_retention.py',[]),
        ('evaluate_arcus_shared_pixels.py',['--confirmation']),
        ('evaluate_arcus_shared_curiosity.py',[]),
        ('qualify_arcus_shared_recovery.py',[]),
        ('qualify_arcus_shared.py',[]),
        ('qualify_arcus_shared_live.py',[]),
        ('qualify_arcus_shared_hearing.py',[]),
        ('qualify_arcus_shared_runtime.py',[]),
    ])
    if args.live_first:
        live_names={'qualify_arcus_shared_live.py','qualify_arcus_shared_hearing.py','qualify_arcus_shared_runtime.py'}
        commands=[item for item in commands if item[0] in live_names]+[item for item in commands if item[0] not in live_names]
    for script,extra in commands:
        if (root/'candidate.json').read_bytes()!=before:raise RuntimeError('Candidate changed during qualification')
        if source_snapshot()!=sources:raise RuntimeError('Runtime changed during qualification')
        print(json.dumps({'stage':script}),flush=True)
        subprocess.run([sys.executable,str(Path(__file__).with_name(script)),'--config',args.config,*extra],check=True)
    if (root/'candidate.json').read_bytes()!=before:raise RuntimeError('Candidate changed during qualification')
    if source_snapshot()!=sources:raise RuntimeError('Runtime changed during qualification')
    artifacts={'integration':'integration.json','posture':'posture-final/report.json',
        'retention':'approach-language/report.json','transfer':'confirmation-transfer.json',
        'pixels':'confirmation-pixels.json','live':'runtime-live.json','recovery':'recovery.json','curiosity':'confirmation-curiosity.json'}
    command=[sys.executable,str(Path(__file__).with_name('compile_arcus_shared_qualification.py')),'--config',args.config]
    for name,path in artifacts.items():command.extend(['--'+name,str(root/path)])
    if json.loads(Path(args.config).read_text()).get('continuity_enabled'):
        command.extend(['--continuity',str(root/'confirmation-continuity-diagnostics.json')])
    subprocess.run(command,check=True)

if __name__=='__main__':main()
