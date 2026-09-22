"""Browser-visible test-2 simulator; learner remains a separate HTTP service."""
import argparse
import os
from pathlib import Path
from baby_arcus.test2_runtime import Test2Runtime
from baby_arcus.transport import Client, StaticResponse
from baby_arcus.services.playroom import viewer_server


class Application:
    def __init__(self, runtime):
        self.runtime = runtime

    def __call__(self, method, path, body):
        r = self.runtime
        if method == 'GET' and path in ('/health', '/ready'):
            return 200, {'ready': True, 'mode': 'isolated-training'}
        if method == 'GET' and path == '/api/test2':
            return 200, r.snapshot()
        if method == 'POST' and path == '/api/test2/step':
            return 200, r.step(body['request_id'])
        if method == 'POST' and path == '/api/test2/control':
            return 200, r.control(body['action'], body.get('cycles', 10))
        if method == 'POST' and path == '/api/test2/message':
            return 200, r.message(body)
        if method == 'POST' and path == '/api/test2/action':
            return 200, r.human_action(body['request_id'], body['action'])
        if method == 'POST' and path == '/api/test2/hear':
            return 200, r.hear()
        if method == 'POST' and path == '/api/test2/learn':
            (r.root / 'pause-training').unlink(missing_ok=True)
            return 200, r.learn_one()
        if method == 'POST' and path == '/api/test2/train':
            (r.root / 'pause-training').unlink(missing_ok=True)
            return 200, r.client.request('POST', '/train', {'updates': body.get('updates', 5), 'request_id': body['request_id']})
        files = {'/': 'test2.html', '/learning-status.js': 'learning-status.js',
                 '/playroom.css': 'playroom.css', '/arcus-renderer.js': 'arcus-renderer.js',
                 '/environment-renderer.js': 'environment-renderer.js', '/arcus-body.png': 'arcus-body.png',
                 '/arcus-lying.png': 'arcus-lying.png', '/arcus-sitting.png': 'arcus-sitting.png'}
        if method == 'GET' and path in files:
            name = files[path]
            mime = 'image/png' if name.endswith('.png') else 'text/javascript' if name.endswith('.js') else 'text/css' if name.endswith('.css') else 'text/html'
            return 200, StaticResponse((Path(__file__).resolve().parents[1] / 'web' / name).read_bytes(), mime)
        raise KeyError(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/baby_arcus/test2.json')
    p.add_argument('--learner-url', default='http://127.0.0.1:8901')
    p.add_argument('--port', type=int, default=8900)
    p.add_argument('--host', default='127.0.0.1')
    args = p.parse_args()
    token = os.environ.get('ARCUS_TEST2_TOKEN')
    if not token:
        raise ValueError('ARCUS_TEST2_TOKEN required')
    client = Client(args.learner_url, token, timeout=300)
    runtime = Test2Runtime(args.config, client)
    server = viewer_server(args.port, Application(runtime), host=args.host)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        runtime.close()


if __name__ == '__main__':
    main()
