"""Temporary fixture-only review UI for visual QA; contains no private data."""
import argparse
import secrets
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baby_arcus.data_staging import StagingStore
from baby_arcus.services.data_review import Application
from baby_arcus.services.playroom import viewer_server


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--port',type=int,default=8934)
    args=parser.parse_args()
    store=StagingStore('runs/test2/three-stage-ui-fixture/staging.sqlite',secrets.token_urlsafe(32),fixture=True)
    server=viewer_server(args.port,Application(store,secrets.token_urlsafe(32)))
    try: server.serve_forever()
    finally: server.server_close(); store.close()
