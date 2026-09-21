"""Empty environment geometry and placement, independent of any body."""
from copy import deepcopy
import math
import uuid
from baby_arcus.contracts import ContractError,fields,identifier

DT = .1
DIRECTIONS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

def coordinate(value, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ContractError("Coordinate outside room")
    return float(value)

class Playroom:
    width, height = 10.0, 7.0

    def __init__(self):
        self.environment_id = uuid.uuid4().hex
        self.generation = 0
        self.placements = {}
        self.human = {"name": "You", "x": 2.0, "y": 5.5, "present": True}
        self.objects={};self._effects={}
        self.colors={'floor':'#e2d2b8','wall':'#7d9472','rug':'#c5d1b4'}

    def color_lesson(self,colors,balls):
        import re
        if (not isinstance(colors,dict) or set(colors)!=set(self.colors)
            or not isinstance(balls,list) or len(balls)!=2
            or any(not isinstance(c,str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',c)
                   for c in list(colors.values())+balls)):
            raise ContractError('Expected three room colors and two ball colors')
        if self.objects:raise ContractError('Reset room before creating a color lesson')
        positions=((3.5,3.5),(6.5,3.5))
        if any(math.hypot(p['x']-x,p['y']-y)<.8 for p in self.placements.values() for x,y in positions):
            raise ContractError('Move Arcus away from lesson positions')
        self.colors=dict(colors)
        for i,((x,y),color) in enumerate(zip(positions,balls)):
            key=uuid.uuid4().hex
            self.objects[key]={'id':key,'name':f'ball {i+1}','x':x,'y':y,'radius':.25,
                              'color':color,'visits':0,'discovered':None}
            self._effects[key]='soft'
        self.generation+=1

    def add_toys(self,seed=None):
        if self.objects:return
        layout=[('blue cube',6.8,3.5,'#558dc6','glows'),
                                      ('yellow toy',3.5,2.,'#d6b844','vibrates'),
                                      ('purple toy',3.5,5.,'#996bb4','soft')]
        if seed is not None:
            import random
            rng=random.Random(seed);placed=[];effects=[row[4] for row in layout];rng.shuffle(effects)
            for index,(name,_,_,color,_) in enumerate(layout):
                for _ in range(1000):
                    x,y=rng.uniform(1.,9.),rng.uniform(1.,6.)
                    if (all(math.hypot(p['x']-x,p['y']-y)>=1.5 for p in self.placements.values())
                        and all(math.hypot(x-q[1],y-q[2])>=2 for q in placed)):break
                else:raise ContractError('Cannot place separated toys')
                placed.append((name,x,y,color,effects[index]))
            layout=placed
        if any(math.hypot(p['x']-x,p['y']-y)<.8 for p in self.placements.values() for _,x,y,_,_ in layout):
            raise ContractError('Move Arcus away from the toy positions before adding toys')
        for name,x,y,color,effect in layout:
            key=uuid.uuid4().hex
            self.objects[key]={'id':key,'name':name,'x':x,'y':y,'radius':.25,'color':color,'visits':0,'discovered':None}
            self._effects[key]=effect

    def interact(self,entity_id,object_id):
        if object_id not in self.objects:raise ContractError('Unknown room object')
        obj=self.objects[object_id];p=self.placements[entity_id]
        if math.hypot(obj['x']-p['x'],obj['y']-p['y'])>1.4:raise ContractError('Object is outside interaction reach')
        novel=obj['discovered'] is None
        obj['visits']+=1;obj['discovered']=self._effects[object_id]
        return {'object_id':object_id,'response':obj['discovered'],'new_discovery':novel}

    def record(self):
        # Private simulator persistence. Never expose effects through snapshot().
        return {'schema':'arcus-room-objects-v2','colors':dict(self.colors),'environment_id':self.environment_id,
                'generation':self.generation,'objects':deepcopy(self.objects),'effects':deepcopy(self._effects)}

    @classmethod
    def restore(cls,record):
        import re
        record=deepcopy(record)
        if record.get('schema')=='arcus-room-objects-v1':
            record['schema']='arcus-room-objects-v2';record['colors']=cls().colors
        fields(record,('schema','environment_id','generation','objects','effects','colors'))
        if record['schema']!='arcus-room-objects-v2':raise ContractError('Unknown room object schema')
        colors=record['colors']
        if not isinstance(colors,dict) or set(colors)!=set(cls().colors) or any(
            not isinstance(c,str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',c) for c in colors.values()):
            raise ContractError('Invalid room colors')
        identifier(record['environment_id'])
        if type(record['generation']) is not int or record['generation']<0:raise ContractError('Invalid room generation')
        objects,effects=record['objects'],record['effects']
        if not isinstance(objects,dict) or not isinstance(effects,dict) or len(objects)>32 or set(objects)!=set(effects):
            raise ContractError('Invalid room object inventory')
        for key,obj in objects.items():
            identifier(key);fields(obj,('id','name','x','y','radius','color','visits','discovered'))
            if obj['id']!=key or not isinstance(obj['name'],str) or not 1<=len(obj['name'])<=80:raise ContractError('Invalid object identity')
            coordinate(obj['x'],.5,9.5);coordinate(obj['y'],.5,6.5)
            if type(obj['radius']) not in (int,float) or obj['radius']!=.25:raise ContractError('Invalid object radius')
            if not isinstance(obj['color'],str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',obj['color']):raise ContractError('Invalid object color')
            if effects[key] not in ('glows','vibrates','soft'):raise ContractError('Unknown toy response')
            if type(obj['visits']) is not int or obj['visits']<0:raise ContractError('Invalid visit count')
            if obj['discovered']!=(None if obj['visits']==0 else effects[key]):raise ContractError('Invalid discovery state')
        world=cls();world.environment_id=record['environment_id'];world.generation=record['generation']
        world.objects=deepcopy(objects);world._effects=deepcopy(effects);world.colors=dict(colors);return world

    def attach(self, entity_id, radius):
        if entity_id not in self.placements:
            self.placements[entity_id] = {"x": self.width/2, "y": self.height/2}
        self.move(entity_id, 0, 0, radius)

    def move(self, entity_id, dx, dy, radius):
        p = self.placements[entity_id]
        x, y = p["x"]+dx, p["y"]+dy
        if any(math.hypot(x-o['x'],y-o['y'])<radius+o['radius'] for o in self.objects.values()):return True
        p.update(x=max(radius, min(self.width-radius, x)), y=max(radius, min(self.height-radius, y)))
        return (x, y) != (p["x"], p["y"])

    def reset(self):
        self.generation += 1
        self.objects={};self._effects={}
        self.colors={'floor':'#e2d2b8','wall':'#7d9472','rug':'#c5d1b4'}
        for p in self.placements.values():
            p.update(x=self.width/2, y=self.height/2)
        self.human = {"name": "You", "x": 2.0, "y": 5.5, "present": True}

    def snapshot(self):
        return {"environment_id": self.environment_id, "generation": self.generation,
                "colors":dict(self.colors),
                "width": self.width, "height": self.height, "walls": 4,
                "placements": deepcopy(self.placements), "human": deepcopy(self.human),'objects':deepcopy(self.objects)}

