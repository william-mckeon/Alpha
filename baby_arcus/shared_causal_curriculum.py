"""Real simulator transitions; seeded training and independent held-out scenes."""
import random
from copy import deepcopy
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.shared_experience import capture
from baby_arcus.shared_temporal import targets,sequence_targets
from baby_arcus.shared_memory import sensory_features
from baby_arcus.body_dynamics import JOINTS,pose

SEEDS={'training':182509,'validation':282509,'confirmation':382509}

def transition(index,split='training'):
    rng=random.Random(SEEDS[split]+index*1009);app=PlayroomApplication()
    try:
        world=app.world;world.body.motor_mode='independent'
        extension=rng.uniform(.25,.85)
        world.body.joint_positions={key:max(0,min(1,extension+rng.uniform(-.08,.08))) for key in JOINTS}
        world.body.previous_joints=dict(world.body.joint_positions)
        world.body.rest_need=.2;world.body.stimulation=.4
        colors={key:'#'+''.join(f'{rng.randrange(64,224):02x}' for _ in range(3)) for key in ('floor','wall','rug')}
        world.environment.color_lesson(colors,['#ed3342','#3366ed'])
        for obj in world.environment.objects.values():obj.update(x=rng.uniform(2,8),y=rng.uniform(1,5))
        for _ in range(20):app.advance()
        memory=[]
        if (index//2)%2:
            views=[(-.75,0),(.75,0),(0,-.75),(0,.75)]
            rng.shuffle(views)
            for yaw,pitch in views[:1+(index//4)%4]:
                app.world.action({'kind':'gaze','yaw':yaw,'pitch':pitch});app.advance();past=capture(app)
                memory.append({'id':past['id'],'features':sensory_features(past)})
        # Exploration must continue from a previous look, not only from center.
        start_yaw,start_pitch=rng.choice(((0,0),(-.75,0),(.75,0),(0,-.75),(0,.75)))
        app.world.action({'kind':'gaze','yaw':start_yaw,'pitch':start_pitch});app.advance()
        before=capture(app);before['memory']=memory
        before['objects']=[];before['hearing']=[]
        before['lesson_provenance']={'split':split,'family':'causal','seed':SEEDS[split]+index*1009}
        before['eligibility']['training']=split=='training'
        steps=1 if index%4 else (2 if split=='confirmation' else 3);transitions=[]
        for step in range(steps):
            if index%2==0:action={'kind':'joint','joint':JOINTS[(index+step)%12],'delta':rng.choice((-.12,.12) if split=='confirmation' else (-.15,.15))}
            else:
                yaw,pitch=rng.choice(((-.75,0),(.75,0),(0,-.75),(0,.75),(0,0)))
                action={'kind':'gaze','yaw':yaw,'pitch':pitch}
            status,result=app('POST','/v1/action',{'request_id':f'{index}-{step}','source':'human','action':action})
            outcome={'experience_id':before['id'],'action':action,'executed':status==200,'result':result,'after':capture(app)}
            for _ in range(rng.choice((3,12,24))):app.advance()
            later=capture(app);later['objects']=[];later['hearing']=[];later['memory']=deepcopy(memory)
            later['lesson_provenance']=dict(before['lesson_provenance']);later['eligibility']['training']=split=='training'
            transitions.append((before,outcome,later));before=later
        row,label=sequence_targets(transitions)
        if steps==1:
            row['executed_action']=deepcopy(transitions[0][1]['action'])
        return row,label,transitions
    finally:app.close()
