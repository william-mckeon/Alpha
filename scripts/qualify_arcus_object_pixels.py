"""Read-only live room camera baseline and content-addressed replay qualification."""
import argparse,json,sys,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.gaze import crop_frame
from baby_arcus.playpen_capture import capture_playpen
from baby_arcus.curiosity_dataset import save_example,load_example
from baby_arcus.object_observation import detect

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8890/api/state')
    parser.add_argument('--output',type=Path,default=Path('runs/arcus_object_pixels_v1'))
    parser.add_argument('--learned',action='store_true',help='Also assess a candidate on the live camera; never activate it')
    args=parser.parse_args()
    with urllib.request.urlopen(args.url,timeout=10) as response:state=json.load(response)
    if (state['paused'] or state['view']['held'] or state['view']['region']!='playpen'
        or state['arcus']['sleep_state']!='awake' or state['arcus']['eyelid_openness']<=0):
        raise ValueError('Live camera is not available')
    frame=crop_frame(capture_playpen(state),state['arcus']);raw=frame['bytes']
    key=save_example(args.output/'frames',raw);restored,record=load_example(args.output/'frames',key)
    report={'passed':restored==raw and record==detect(raw),'entity_id':state['arcus']['entity_id'],
        'tick':state['tick'],'frame_sha256':key,'camera':frame['camera'],
        'visible_regions':len(record['detections']),
        'scope':'Read-only live HTTP state rendered through playpen camera; palette baseline and replay integrity, not learned recognition or movement qualification'}
    if args.learned:
        import torch
        from baby_arcus.object_perception_learning import load
        from baby_arcus.object_perception_environment import arrays
        from baby_arcus.object_observation import learned_observation
        torch.set_num_threads(2)
        cfg,body,model,device=load('configs/baby_arcus/object_perception.json',qualified=False)
        pixels=arrays(raw)
        with torch.no_grad():labels=model(body.core,torch.tensor(pixels)[None].to(device)).argmax(1)[0].cpu().numpy()
        report['learned_observation']=learned_observation(pixels,labels)
        report['scope']='Read-only live camera diagnostic; candidate inference, not proof of accuracy or activation'
        report['offline_qualified']=json.loads((Path(cfg['output'])/'qualification.json').read_text())['passed']
        report['replay_integrity_passed']=report['passed']
        report['passed']=report['passed'] and report['offline_qualified']
        args.output=Path(cfg['output'])
    (args.output/'live-qualification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
