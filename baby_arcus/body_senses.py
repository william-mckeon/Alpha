"""Proprioception independent of images or privileged lesson goals."""
from baby_arcus.body_dynamics import sensations
def observe_body_senses(body, held=False):
    return {"schema":"arcus-proprioception-v1","entity_id":body.entity_id,
            "sleep_state":body.sleep_state,"eyelid_openness":body.eyelid_openness,
            'rest':{'schema':'arcus-rest-senses-v1','need':body.rest_need,'stimulation':body.stimulation,
                    'alertness':body.alertness,'mode':body.rest_mode,'simulated':True},
            **sensations(body,held)}
