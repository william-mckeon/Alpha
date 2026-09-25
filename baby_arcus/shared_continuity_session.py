"""Experimental model-to-simulator bridge, gated off from the deployed worker."""
from pathlib import Path
import torch
from baby_arcus.shared_object_memory import ObjectMemory, descriptor
from baby_arcus.shared_object_planning import VisualPlan
from baby_arcus.shared_continuity_curriculum import VIEWS
from baby_arcus.shared_depth import verify_depth
from baby_arcus.shared_identity_context import identity_confidence


class ContinuitySession:
    def __init__(self, model, tokenizer, manifest, path, shared_connection=False):
        if model.version not in (10, 11):
            raise ValueError('Continuity requires experimental shared schema 10 or 11')
        verify_depth(model)
        self.model, self.tokenizer, self.manifest = model, tokenizer, manifest
        self.memory = ObjectMemory(Path(path), shared_connection=shared_connection)
        self.plan = VisualPlan()
        from baby_arcus.shared_identity_context import VisualSurvey
        self.survey = VisualSurvey(self.memory, manifest['generation'])

    def observe(self, row, now, plan=True):
        from baby_arcus.shared_experience import validate
        from baby_arcus.object_perception_environment import arrays
        from baby_arcus.object_observation import learned_observation
        raw = validate(row)
        if not raw or row['senses']['held'] or row.get('hearing') or row.get('events'):
            return {'plan': self.plan.cancel('Sensory interruption'), 'objects': []}
        if row.get('objects'):
            raise ValueError('Simulator object metadata must not enter continuity inference')
        device = next(self.model.parameters()).device
        with torch.no_grad():
            out = self.model([row], self.tokenizer, requested=('hidden', 'perception'))
            hidden = out['hidden']
            observation = learned_observation(arrays(raw), out['perception'][0].argmax(0).cpu().numpy())
            inventory = []
            if self.model.version >= 11:
                survey = self.survey.observe(row, [descriptor(region, row['gaze']) for region in observation['objects']])
                if not survey['complete']:
                    return {'objects': [], 'survey_views': len(survey['views']), 'survey_seen': list(survey['views']),
                            'plan': self.plan.cancel('Prior visual survey incomplete; identity unknown')}
                inventory = survey['inventory']
            tracks = self.memory.recall(row)['tracks']
            matrix = []
            views=[view for track in tracks for view in track['views']]
            previous=torch.tensor(views,device=device) if views else None
            for region in observation['objects']:
                description=descriptor(region,row['gaze'])
                observed = torch.tensor([description], device=device)
                risk = 0.0
                if self.model.version >= 11:
                    from baby_arcus.shared_identity_context import uncertainty
                    risk = uncertainty(self.model, hidden, inventory, description)
                scores=self.model.association_logits(hidden.expand(len(views),-1),previous,
                    observed.expand(len(views),-1)).sigmoid().cpu().tolist() if views else []
                row_scores=[];offset=0
                for track in tracks:
                    count=len(track['views'])
                    row_scores.append(identity_confidence(max(scores[offset:offset+count]),risk))
                    offset+=count
                matrix.append(row_scores)
            state = self.memory.observe(row, observation, matrix, self.manifest['generation'])
            target = None
            if self.plan.state:
                target = next((t for t in state['tracks'] if t['id'] == self.plan.state['target']), None)
                if target and target['visible']:
                    return {'objects': state['tracks'], 'plan': self.plan.cancel('Target observed')}
            if target is None:
                target = next((t for t in reversed(state['tracks']) if not t['visible']), None)
            if not plan or target is None:
                return {'objects': state['tracks'], 'plan': self.plan.cancel('No search requested or needed')}
            candidates = []
            scores=self.model.search_logits(hidden.expand(len(VIEWS),-1),
                torch.tensor([target['views'][-1]],device=device).expand(len(VIEWS),-1),
                torch.tensor(VIEWS,device=device)).sigmoid().cpu().tolist()
            for (yaw,pitch),score in zip(VIEWS,scores):
                candidates.append({'action': {'kind': 'gaze', 'yaw': yaw, 'pitch': pitch}, 'score': score})
            return {'objects': state['tracks'], 'plan': self.plan.step(row, candidates, target['id'], now)}

    def close(self):
        self.memory.close()
