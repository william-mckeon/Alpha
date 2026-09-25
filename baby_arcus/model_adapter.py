"""Arcus decisions exposed as a LangChain runnable, without an external LLM."""
import codecs
import torch
from baby_arcus.body_vocabulary import ACTIONS
from baby_arcus.services.shared_worker import gaze_action


def decide(model, tokenizer, row):
    with torch.no_grad():
        model.eval()
        out = model([row], tokenizer,requested=('decision',))
        activity = int(out['activity'][0].argmax())
        action, hearing, expression = None, None, ''
        if activity == 1:
            head = ('body', 'lying', 'sitting')[int(out['posture_choice'][0].argmax())]
            action = ACTIONS[int(out[head][0].argmax())]
        elif activity == 2:
            kind = (None, 'rest', 'alert', 'sleep_when_ready', 'wake_voluntarily')[int(out['rest'][0].argmax())]
            action = {'kind': kind} if kind else None
        elif activity == 4:
            hearing = (None, 'resume', 'pause', 'resume', 'replay', 'restart')[int(out['language_choice'][0].argmax())]
        elif activity == 5:
            token = int(out['text'][0].argmax())
            try:
                raw = tokenizer.enc.decode_single_token_bytes(token)
                expression = codecs.getincrementaldecoder('utf-8')().decode(raw, final=False)
            except KeyError:
                pass
        elif activity == 3:
            from baby_arcus.shared_causal import choose_experiment
            action, _ = choose_experiment(model, tokenizer, row)
        elif activity == 6:
            action = gaze_action(row, int(out['gaze_choice'][0].argmax()))
        elif activity == 7:
            if row['senses']['height'] < .92:
                action = ACTIONS[int(out['body'][0].argmax())]
            else:
                logits = out['approach']
                action = {'kind': 'move', 'direction': ('up', 'down', 'left', 'right')[int(logits[0].argmax())]}
        return {'actor': 'arcus', 'activity': activity, 'action': action,
                'hearing_action': hearing, 'expression': expression,
                'routing_fraction': float(model.core.last_compute_fraction), 'depth_capacity': model.body.cfg.capacity}


def runnable(infer):
    from langchain_core.runnables import RunnableLambda
    return RunnableLambda(infer)
