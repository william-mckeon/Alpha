"""Explicit, seeded lessons. Targets/provenance never enter model observations."""
import random,hashlib,base64
from copy import deepcopy
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.shared_experience import capture
from baby_arcus.body_senses import observe_body_senses
from baby_arcus.rest_environment import features

COMMANDS=(('stand up',{'activity':1,'posture_choice':0}),
 ('lie down',{'activity':1,'posture_choice':1}),('sit down',{'activity':1,'posture_choice':2}),
 ('open your eyes',{'activity':6,'gaze_choice':5}),('close your eyes',{'activity':6,'gaze_choice':6}),
 ('look left',{'activity':6,'gaze_choice':1}),('look right',{'activity':6,'gaze_choice':2}),
 ('listen',{'activity':4,'language_choice':1}),('pause listening',{'activity':4,'language_choice':2}),
 ('resume listening',{'activity':4,'language_choice':3}),('wait quietly',{'activity':0}),
 ('come here',{'activity':7}),('make a sound',{'activity':5}),
 ('replay that',{'activity':4,'language_choice':4}),('restart listening',{'activity':4,'language_choice':5}))
SEEDS={'training':9381000,'validation':9481000,'confirmation':9581000}

def example(index,split='training',family='commands',paired=False):
    if split not in SEEDS:raise ValueError('Unknown lesson split')
    scene=index//len(COMMANDS) if paired and family=='commands' else index//4 if paired and family=='color_reference' else index//16 if paired and family=='rest' else index
    rng=random.Random(SEEDS[split]+scene*1009);app=PlayroomApplication()
    try:
        app.world.body.rest_need=rng.random();app.world.body.stimulation=rng.random()
        app.world.body.motor_mode='independent'
        from baby_arcus.body_dynamics import JOINTS
        app.world.body.joint_positions={key:rng.random() for key in JOINTS}
        for _ in range(12):app.world.step()
        if family=='color_reference':
            app.world.environment.color_lesson({'floor':'#e2d2b8','rug':'#c5d1b4','wall':'#7d9472'},['#ed3342','#3366ed'])
            reverse=bool((index//2)%2) if paired else bool(rng.randrange(2));objects=list(app.world.environment.objects.values())
            for j,obj in enumerate(objects):obj.update(x=(4 if j==int(reverse) else 6),y=rng.uniform(2.1,2.6))
            target=index%2 if paired else rng.randrange(2);word=('red','blue')[target]
            text=f'look at the {word} ball';label={'activity':6,'gaze_choice':1 if target==int(reverse) else 2}
        elif family=='commands':text,label=COMMANDS[index%len(COMMANDS)];label=dict(label)
        elif family=='rest':
            from baby_arcus.body_dynamics import pose
            from baby_arcus.rest_environment import rewards
            state_index=scene if paired else index
            if state_index%2==0:
                app.world.body.joint_positions=pose(0);app.world.body.previous_joints=pose(0)
                for _ in range(30):app.world.step()
            if state_index%4==0:app.world.body.sleep_state='sleeping';app.world.body.eyelid_openness=0
            if paired:
                app.world.body.rest_need=(index%4)/3;app.world.body.stimulation=((index//4)%4)/3
                app.world.body.rest_mode='resting' if (scene//4)%2 else 'active'
            internal=features(app.world.body);text=''
            label={'activity':1,'posture_choice':1} if internal[0]>.65 and not internal[2] and not internal[3] else {'activity':2,'rest':rewards(internal)}
        else:raise ValueError('Unknown lesson family')
        # New compositional wording in validation/confirmation, same task semantics.
        prefix=rng.choice(('Arcus, ','Please ','','Can you ','Would you please ','Hello ','Now ')) if split=='training' else rng.choice(('Now please ','Hello Arcus, please ','Would you '))
        if split=='training' and text:
            if rng.randrange(4)==0:text=text[0].upper()+text[1:]
            if rng.randrange(4)==0:text+=', Arcus'
        row=capture(app);row['hearing']=[{'text':prefix+text,'source':'simulated_hearing'}] if text else []
        row['objects']=[];row['object_source']='none'  # no oracle location/colour features
        row['session']=f'curriculum:{split}:{family}:{SEEDS[split]+index*1009}'
        # Runtime supplies up to two prior body observations. Include variable
        # history during lessons so sequence length cannot become a rest cue.
        # These are stationary preceding observations, not future target labels.
        row['history']=[]
        for age in range(rng.randrange(3),0,-1):
            prior={key:deepcopy(row[key]) for key in ('session','entity_id','scope_id','epoch','tick','senses')}
            prior['tick']=max(0,row['tick']-age);row['history'].append(prior)
        row['eligibility']['training']=split=='training'
        row['lesson_provenance']={'split':split,'family':family,'seed':SEEDS[split]+index*1009}
        return row,label,{'split':split,'family':family,'seed':SEEDS[split]+index*1009,'index':index}
    finally:app.close()

def samples(count,split='training',include_rest=False,paired=False):
    families=('commands','color_reference','rest') if include_rest else ('commands','color_reference')
    return [example(i//len(families),split,families[i%len(families)],paired=paired) for i in range(count)]
