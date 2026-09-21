"""Shared-model continuity service; production startup requires qualified evidence."""
import argparse
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import torch
from baby_arcus.services.shared_worker import Worker
from baby_arcus.shared_continuity_model import load_candidate
from baby_arcus.shared_continuity_session import ContinuitySession
from baby_arcus.shared_depth import verify_depth


class ContinuityWorker(Worker):
    def __init__(self, config, manifest=None, hearing_state_path=None):
        from arcus.tokenizer import get_tokenizer
        import sys
        started=time.monotonic()
        def stage(name):
            print(json.dumps({'startup_stage':name,'elapsed_seconds':time.monotonic()-started}),file=sys.stderr,flush=True)
        torch.set_num_threads(2)
        self.cfg = json.loads(Path(config).read_text())
        self.root = Path(self.cfg['root'])
        self.manifest = manifest or json.loads((self.root/'active.json').read_text())
        self.diagnostic = manifest is not None
        stage('configuration_loaded')
        if not self.diagnostic:
            from baby_arcus.shared_qualification import verify_evidence
            report = json.loads((self.root/'qualification.json').read_text())
            if report.get('candidate') != self.manifest or report.get('continuity') is not True:
                raise ValueError('Continuity deployment is not qualified')
            verify_evidence(report, self.root)
        stage('qualification_checked')
        if importlib.metadata.version('tiktoken') != self.cfg['tiktoken_version']:
            raise ValueError('Tokenizer release mismatch')
        self.model, self.data = load_candidate(self.root, self.manifest, 'cuda' if torch.cuda.is_available() else 'cpu')
        stage('weights_loaded')
        # Inference owns the model, not a second CPU copy of every weight and
        # optimizer accumulator. Training restores those from immutable snapshots.
        self.data={key:self.data[key] for key in ('schema','progress','body_config')}
        if self.model.version < 11:
            raise ValueError('Uncertainty-aware continuity requires schema 11')
        verify_depth(self.model, self.cfg)
        self.model.eval().requires_grad_(False)
        self.tokenizer = get_tokenizer(self.cfg['encoding'])
        self.lock = threading.Lock()
        self.transaction_lock = threading.Lock()
        self.stream = None
        self.hearing_state_path = Path(hearing_state_path) if hearing_state_path else self.root/'live-hearing.json'
        self.isolated_hearing = hearing_state_path is not None
        if self.diagnostic and self.isolated_hearing and self.hearing_state_path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('Diagnostic hearing cursor must be outside the candidate directory')
        self.temporary = tempfile.TemporaryDirectory() if self.diagnostic else None
        memory_path = Path(self.temporary.name)/'objects.sqlite3' if self.temporary else self.root/'objects.sqlite3'
        self.continuity = ContinuitySession(self.model, self.tokenizer, self.manifest, memory_path, shared_connection=True)
        self.pending = None
        stage('ready')

    def predict(self, row):
        with self.transaction_lock:
            return self._predict(row)

    def _predict(self, row):
        if row.get('op') == 'continuity_outcome':
            if self.pending is None:
                raise ValueError('No pending continuity proposal')
            outcome = row['outcome']
            result = self.continuity.plan.acknowledge(self.pending, outcome)
            self.pending = None
            return {'continuity_acknowledgement': result, **self.manifest}
        if self.pending is not None and row.get('op') != 'hearing':
            raise ValueError('A verified action outcome is required before reobservation')
        reply = super().predict(row)
        if row.get('op') == 'hearing':
            return reply
        clean = deepcopy(row)
        clean['objects'] = []
        permitted = (bool(row.get('exploration_permitted') or self.continuity.plan.state)
                     and not row['hearing'] and row['internal'][0] < .65)
        with self.lock, torch.no_grad():
            result = self.continuity.observe(clean, row['captured_at'], plan=permitted)
            action = result['plan'].get('action')
            if permitted and result.get('survey_views', 9) < 9:
                from baby_arcus.shared_continuity_curriculum import VIEWS
                from baby_arcus.shared_memory import sensory_features
                previous = torch.tensor(sensory_features(clean)[24:], device=next(self.model.parameters()).device)
                candidates = []
                for index, (yaw, pitch) in enumerate(VIEWS):
                    if str(index) in result['survey_seen']:
                        continue
                    conditioned = deepcopy(clean)
                    proposal = {'kind': 'gaze', 'yaw': yaw, 'pitch': pitch}
                    conditioned['executed_action'] = proposal
                    forecast = self.model([conditioned], self.tokenizer, requested=('future_rgb',))['future_rgb'][0]
                    candidates.append((float((forecast-previous).square().mean()), proposal))
                if candidates:
                    score, proposal = max(candidates, key=lambda item: item[0])
                    reply.update(proposal=proposal, activity=6, predicted_outcome=None, experimentation=None)
                    result['plan'] = {'action': proposal, 'status': 'survey_step', 'predicted_visual_change': score}
            if action is not None:
                self.pending = clean
                reply.update(proposal=action, activity=6, predicted_outcome=None, experimentation=None)
            reply['continuity'] = result
            reply['continuity']['requires_acknowledgement'] = self.pending is not None
        return reply

    def close(self):
        super().close()
        self.continuity.close()
        if self.temporary:
            self.temporary.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--port', type=int)
    args = parser.parse_args()
    worker = ContinuityWorker(args.config)
    try:
        if args.port:
            from baby_arcus.transport import serve
            token = os.environ.get('ARCUS_SHARED_TOKEN')
            if not token:
                raise ValueError('Shared HTTP authentication required')
            server = serve('0.0.0.0', args.port, worker, token)
            try:
                server.serve_forever()
            finally:
                server.server_close()
        else:
            import sys
            print(json.dumps(worker.ready()), flush=True)
            for line in sys.stdin:
                try:
                    reply = worker.predict(json.loads(line))
                except Exception as exc:
                    reply = {'error': str(exc)}
                print(json.dumps(reply), flush=True)
    finally:
        worker.close()


if __name__ == '__main__':
    main()
