"""Local review service with distinct ingestion and human-approval credentials."""
import argparse
import os
import secrets
from pathlib import Path
from baby_arcus.data_staging import StagingStore
from baby_arcus.staging_graph import graph
from baby_arcus.transport import serve, StaticResponse
from baby_arcus.services.playroom import viewer_server


class Application:
    def __init__(self, store, ingestion_secret):
        if len(ingestion_secret) < 24 or ingestion_secret == store.secret:
            raise ValueError('Separate strong ingestion and human-review credentials required')
        self.store, self.ingestion_secret = store, ingestion_secret
        self.graph = graph(store)

    def __call__(self, method, path, body):
        if method == 'GET' and path in ('/','/data-review.js'):
            name = 'data-review.html' if path == '/' else 'data-review.js'
            return 200, StaticResponse((Path(__file__).resolve().parents[1]/'web'/name).read_bytes(),
                                       'text/html; charset=utf-8' if path == '/' else 'application/javascript')
        if method == 'GET' and path == '/health':
            return 200, {'ready':True,'fixture':self.store.fixture,'training_enabled':False}
        if method != 'POST' or not isinstance(body,dict):
            raise KeyError(path)
        credential = body.get('credential','')
        if not isinstance(credential,str): return 401, {'error':'Authentication required'}
        human = self.store.secret is not None and secrets.compare_digest(credential,self.store.secret)
        machine = secrets.compare_digest(credential,self.ingestion_secret)
        if not human and not machine: return 401, {'error':'Authentication required'}
        if path == '/queue': return 200, {'batches':self.store.queue(body.get('offset',0),body.get('limit',20))}
        if path == '/stage': return 200, self.graph.invoke({'records':body['records']})
        if path == '/sources': return 200, {'batch_id':self.store.stage_sources(body['manifest'])}
        if path == '/batch': return 200, self.store.get(body['batch_id'])
        if path == '/recommend':
            self.store.recommend(body['batch_id'],body['recommendation'])
            return 200, {'recommended':True,'approved':False}
        if path == '/revoke':
            if not human: return 403, {'error':'Human review required'}
            return 200, self.store.revoke(body['batch_id'],body['reviewer'],body['reason'],credential)
        if path == '/review':
            if not human: return 403, {'error':'Only the separate human review credential can approve'}
            return 200, self.store.review(body['batch_id'],body['decision'],body['reviewer'],credential)
        raise KeyError(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store',default='runs/test2/alpha-three-stage-review/staging.sqlite')
    parser.add_argument('--port',type=int,default=8932)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--fixture',action='store_true',help='Isolated synthetic review store; never mix with real records')
    args = parser.parse_args()
    store = StagingStore(args.store,os.environ['ALPHA_REVIEW_TOKEN'],fixture=args.fixture)
    app = Application(store,os.environ['ALPHA_INGEST_TOKEN'])
    server = viewer_server(args.port,app,host=args.host,internal_host='review:'+str(args.port),internal_token=os.environ['ALPHA_INGEST_TOKEN'])
    try: server.serve_forever()
    finally: server.server_close(); store.close()


if __name__ == '__main__': main()
