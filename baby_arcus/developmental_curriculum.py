"""Seeded mixed lessons. Simulator rewards are targets, never model inputs."""
import copy
import torch
from baby_arcus.shared_curriculum import example
from baby_arcus.shared_experience import capture
from baby_arcus.shared_temporal import body_vector
from baby_arcus.body_vocabulary import ACTIONS


FAMILIES = ('commands', 'color_reference', 'rest', 'standing', 'lying', 'sitting', 'approach', 'language', 'causal', 'perception', 'continuity')


def lesson(index, tokenizer, model, split='training', families=FAMILIES):
    if not families or any(f not in FAMILIES for f in families):
        raise ValueError('Unknown curriculum family')
    family = families[index % len(families)]
    scene_index = index // len(families)
    if family in ('commands', 'color_reference', 'rest'):
        row, target, _ = example(scene_index, split, family, paired=True)
        return row, target, family
    if family == 'language':
        row, _, _ = example(scene_index, split, 'commands')
        # Explicit small foundation lessons; not mislabeled as DatasetForge exposure.
        text = row['hearing'][0]['text']
        target = {'tokens': [tokenizer.eot_token] + tokenizer.encode(text)[:64]}
        row['language_prefix_ids'] = [tokenizer.eot_token]
        row['hearing'] = []  # next-token targets must not leak through hearing
        return row, target, family
    if family == 'causal':
        from baby_arcus.shared_causal_curriculum import transition
        row, target, _ = transition(scene_index, split)
        return row, target, family
    if family == 'continuity':
        from baby_arcus.shared_continuity_curriculum import scene, encode_scene
        frames, ambiguous = encode_scene(model, tokenizer, scene_index, split)
        records, _ = scene(scene_index, split)
        for i, before in enumerate(frames):
            for desc, identity in zip(before['descriptors'], before['labels']):
                if not identity:
                    continue
                j = (i + 1) % len(frames)
                after = frames[j]
                row = copy.deepcopy(records[i][0])
                row['search_query'] = desc + after['gaze']
                target = {'visual_search': [float(after['visible'][identity - 1])]}
                for other, other_id in zip(after['descriptors'], after['labels']):
                    if other_id and not ambiguous:
                        row['identity_pair'] = [desc, other]
                        target['identity_match'] = [float(identity == other_id)]
                        break
                return row, target, family
        # Random detectors cannot yet supply object matches. Train pixels and
        # report that prerequisite explicitly, without inventing object learning.
        family = 'perception'
    if family == 'perception':
        import base64, hashlib
        from io import BytesIO
        import numpy as np
        from PIL import Image
        from baby_arcus.object_perception_environment import example as pixel_example
        row, _, _ = example(scene_index, split, 'color_reference')
        pixels, labels, _ = pixel_example(2101 if split == 'training' else 102101, scene_index)
        buffer = BytesIO()
        Image.fromarray((pixels.transpose(1, 2, 0)*255).astype(np.uint8)).save(buffer, format='PNG')
        raw = buffer.getvalue()
        row['vision'].update(available=True, image_base64=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
        row['hearing'] = []
        row['lesson_provenance']['family'] = 'perception'
        return row, {'perception': labels.tolist()}, family
    from baby_arcus.services.playroom import PlayroomApplication
    from baby_arcus.standing_environment import StandingEnvironment
    app = PlayroomApplication()
    try:
        if family == 'approach':
            import random, math
            from baby_arcus.body_dynamics import pose
            rng = random.Random(index + (2101 if split == 'training' else 102101))
            app.world.body.joint_positions = pose(1)
            app.world.body.previous_joints = pose(1)
            app.world.body.motor_mode = 'independent'
            app.world.environment.human.update(x=rng.uniform(1,9), y=rng.uniform(1,6))
            for _ in range(20):
                app.world.step()
            row = capture(app)
            row['objects'] = []
            row['object_source'] = 'none'
            row['hearing'] = [{'text': 'Come here, Arcus', 'source': 'simulated_hearing'}]
            row['lesson_provenance'] = {'split': split, 'family': family, 'index': index}
            row['eligibility']['training'] = split == 'training'
            with torch.no_grad():
                out = model([row], tokenizer, requested=('approach',))
                choice = int(torch.distributions.Categorical(logits=out['approach']).sample()[0])
            action = {'kind': 'move', 'direction': ('up','down','left','right')[choice]}
            app.world.action(action)
            for _ in range(3):
                app.world.step()
            after = capture(app)
            reward = max(-10, min(10, math.hypot(*row['hearing_relative']) - math.hypot(*after['hearing_relative']) - .01))
            row['executed_action'] = action
            return row, {'policy': {'head': 'approach', 'index': choice, 'advantage': reward}, 'future_body': body_vector(after)}, family
        from baby_arcus.lying_environment import LyingEnvironment
        from baby_arcus.sitting_environment import SittingEnvironment
        kind = {'standing': StandingEnvironment, 'lying': LyingEnvironment, 'sitting': SittingEnvironment}[family]
        env = kind(seed=2101 + index + (100000 if split != 'training' else 0))
        app.world = env.session
        row = capture(app)
        row['objects'] = []
        row['object_source'] = 'none'
        row['session'] = f'test2:{split}:motor:{index}'
        row['lesson_provenance'] = {'split': split, 'family': 'motor', 'index': index}
        row['eligibility']['training'] = split == 'training'
        with torch.no_grad():
            head = 'body' if family == 'standing' else family
            logits = model([row], tokenizer, requested=(head,))[head]
            choice = int(torch.distributions.Categorical(logits=logits).sample()[0])
        _, reward, _ = env.step(choice)
        after = capture(app)
        row['executed_action'] = copy.deepcopy(ACTIONS[choice])
        return row, {'policy': {'head': head, 'index': choice, 'advantage': reward},
                     'future_body': body_vector(after)}, family
    finally:
        app.close()
