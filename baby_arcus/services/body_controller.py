"""Bounded checkpoint-driven tool controller; never sends mouse or keyboard input."""
import argparse
import os
import time
import uuid
from baby_arcus.body_vocabulary import ACTIONS
from baby_arcus.body_policy import load
from baby_arcus.transport import Client
def run(client,model,steps=30,interval=.12,audit=None):
    receipts=[]
    for _ in range(steps):
        senses=client.request("POST","/v1/tools/call",{"request_id":uuid.uuid4().hex,"name":"observe_senses","arguments":{}})
        if audit:audit.emit("policy.observation",{"senses":senses})
        if senses["sleep_state"]!="awake" or senses["held"] or senses.get("paused"):break
        action=ACTIONS[model.choose(senses)]
        if audit:audit.emit("policy.decision",{"action":action},durable=True)
        if action:
            result=client.request("POST","/v1/tools/call",{"request_id":uuid.uuid4().hex,"name":"body_action","arguments":action})
            receipts.append(result["event"])
            if audit:audit.emit("policy.action_result",{"result":result},durable=True)
        if interval:time.sleep(interval)
    return receipts
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",required=True);p.add_argument("--url",default="http://127.0.0.1:8892");p.add_argument("--steps",type=int,default=30)
    a=p.parse_args()
    if not 1<=a.steps<=180:p.error("steps must be 1–180")
    token=os.environ.get("ARCUS_BODY_TOOL_TOKEN")
    if not token:p.error("ARCUS_BODY_TOOL_TOKEN is required")
    model,_=load(a.checkpoint);model.eval()
    from baby_arcus.audit import AuditLog
    audit=AuditLog("runs/arcus_body_controller/audit","body-controller")
    audit.emit("policy.loaded",{"checkpoint":a.checkpoint},durable=True)
    try:
        print({"actions":len(run(Client(a.url,token,attempts=1),model,a.steps,audit=audit))})
    except Exception as exc:
        audit.emit("policy.error",{"exception_type":type(exc).__name__},durable=True)
        raise
    finally:audit.close()
if __name__=="__main__":main()
