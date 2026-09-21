"""One-time migration of fully discovered legacy room exports; never invent effects."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.playroom import Playroom
from baby_arcus.room_store import RoomStore
from baby_arcus.embodiment_store import EmbodimentStore

def main():
    parser=argparse.ArgumentParser();parser.add_argument('export',type=Path);parser.add_argument('root',type=Path)
    parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if (args.root/'room-objects.json').exists():raise ValueError('Room save already exists; migration will not overwrite it')
    data=json.loads(args.export.read_text(encoding='utf-8-sig'));env=data['state']['environment']
    objects=env.get('objects',{})
    if any(o['discovered'] is None for o in objects.values()):raise ValueError('Undiscovered effects cannot be recovered from a public export')
    room=Playroom.restore({'schema':'arcus-room-objects-v1','environment_id':env['environment_id'],
                          'generation':env['generation'],'objects':objects,
                          'effects':{key:obj['discovered'] for key,obj in objects.items()}})
    if not args.check:
        owner=EmbodimentStore(args.root)
        try:
            if owner.load().entity_id!=data['state']['arcus']['entity_id']:raise ValueError('Saved body does not match export')
            RoomStore(args.root).save(room)
        finally:owner.close()
    print('Legacy room validated' if args.check else 'Legacy room migrated')

if __name__=='__main__':main()
