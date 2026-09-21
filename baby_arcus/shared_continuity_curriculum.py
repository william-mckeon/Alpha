"""Rendered view sequences; oracle instance masks are scoring labels only."""
import random
from copy import deepcopy
from io import BytesIO
import numpy as np
from PIL import Image
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.shared_experience import capture
from baby_arcus.object_perception_environment import color
from baby_arcus.playpen_capture import capture_playpen
from baby_arcus.gaze import crop_frame

# 1382509 is retained in the failed v3 evidence. Freeze a fresh confirmation
# cohort after the validation-only ambiguity-composition correction.
SEEDS = {'training': 1182509, 'validation': 1282509, 'confirmation': 1482509}
VIEWS = tuple((x, y) for y in (-1., 0., 1.) for x in (-1., 0., 1.))


def scene(index, split, moving=False, primed=False):
    rng = random.Random(SEEDS[split]+index*1009)
    app = PlayroomApplication()
    try:
        world = app.world
        colors = {key: color(rng) for key in world.environment.colors}
        appearance = color(rng)
        world.environment.color_lesson(colors, [appearance, appearance if index % 5 == 0 else color(rng)])
        world.environment.human['present'] = False
        for obj in world.environment.objects.values():
            obj.update(x=rng.uniform(1, 9), y=rng.uniform(1, 6))
        records = []
        sequence = VIEWS*2 if primed else VIEWS
        for step, (yaw, pitch) in enumerate(sequence):
            if moving and step == (len(VIEWS) if primed else 0)+len(VIEWS)//2:
                # The evaluator knows which object moved; the learner receives
                # only the newly rendered view and its own remembered regions.
                objects = list(world.environment.objects.values())
                first, second = [(obj['x'], obj['y']) for obj in objects]
                objects[0].update(x=second[0], y=second[1])
                objects[1].update(x=first[0], y=first[1])
            world.body.eye_yaw, world.body.eye_pitch = yaw, pitch
            row = capture(app)
            row['objects'] = []
            row['object_source'] = 'none'
            row['hearing'] = []
            row['lesson_provenance'] = {'split': split, 'family': 'object-continuity', 'seed': SEEDS[split]+index*1009}
            row['eligibility']['training'] = split == 'training'
            # IDs exist solely in this scoring image; never attach them to row.
            oracle = deepcopy(world.snapshot())
            oracle['environment']['colors'] = {key: '#000000' for key in colors}
            for j, obj in enumerate(oracle['environment']['objects'].values()):
                obj['color'] = ('#ff00ff', '#00ffff')[j]
            frame = crop_frame(capture_playpen(oracle), oracle['arcus'])
            with Image.open(BytesIO(frame['bytes'])) as image:
                rgb = np.asarray(image.convert('RGB').resize((96, 96), Image.Resampling.NEAREST))
            labels = np.zeros((96, 96), dtype=np.int64)
            labels[np.all(rgb == [255, 0, 255], axis=-1)] = 1
            labels[np.all(rgb == [0, 255, 255], axis=-1)] = 2
            records.append((row, labels))
        return records, index % 5 == 0
    finally:
        app.close()


def encode_scene(model, tokenizer, index, split, moving=False, primed=False):
    import torch
    from baby_arcus.object_observation import learned_observation, regions
    from baby_arcus.object_perception_environment import arrays
    from baby_arcus.shared_experience import validate
    from baby_arcus.shared_object_memory import descriptor
    records, ambiguous = scene(index, split, moving, primed)
    encoded = []
    with torch.no_grad():
        for row, labels in records:
            out = model([row], tokenizer, requested=('hidden', 'perception'))
            predicted = out['perception'][0].argmax(0).cpu().numpy()
            observation = learned_observation(arrays(validate(row)), predicted)
            descriptors = [descriptor(region, row['gaze']) for region in observation['objects']]
            identities = []
            for points in regions(predicted == 3):
                xs, ys = zip(*points)
                ids, counts = np.unique(labels[ys, xs], return_counts=True)
                best = int(ids[counts.argmax()])
                identities.append(best if counts.max()/len(points) >= .6 else 0)
            encoded.append({'hidden': out['hidden'][0].cpu(), 'descriptors': descriptors, 'labels': identities,
                            'visible': [int((labels == k).sum()) >= 8 for k in (1, 2)], 'gaze': row['gaze'][2:]})
    if primed:
        inventory = [d for frame in encoded[:len(VIEWS)] for d in frame['descriptors']][:32]
        encoded = encoded[len(VIEWS):]
        for frame in encoded:
            frame['inventory'] = inventory
    return encoded, ambiguous
