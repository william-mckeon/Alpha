"""Explicit symbolic room cue; no pixels, desktop coordinates or mouse access."""
from baby_arcus.body_senses import observe_body_senses

def observation(world,events,target=None,modality='simulated_hearing'):
    p=world.environment.placements[world.body.entity_id]
    return {'schema':'arcus-interaction-v1','tick':world.tick,
            'senses':observe_body_senses(world.body,world.view.held),
            'events':events,'region':world.view.region,'paused':world.paused,
            'position':dict(p),'target':target,
            'target_modality':modality if target else None,
            'hearing':{'schema':'arcus-symbolic-hearing-v1','localization':'ideal_simulated_location',
                       'relative_source':[target['x']-p['x'],target['y']-p['y']],
                       'audio_waveform':False} if target and modality=='simulated_hearing' else None,
            'relative_target':[target['x']-p['x'],target['y']-p['y']] if target else [0,0]}
