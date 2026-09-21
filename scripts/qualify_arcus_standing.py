"""Checkpoint reload and learned actions over real authenticated HTTP services."""
import json
from pathlib import Path
import sys
import tempfile
import threading
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from baby_arcus.body_policy import load,save
from baby_arcus.body_learning import assess
from baby_arcus.body_tools import BodyToolsApplication
from baby_arcus.services.playroom import PlayroomApplication
from baby_arcus.services.body_controller import run
from baby_arcus.body_dynamics import pose
from baby_arcus.transport import Client,serve
torch.set_num_threads(2)
model,data=load("runs/arcus_standing_pilot/standing.pt")
model.eval()
with tempfile.TemporaryDirectory() as root:
    app=PlayroomApplication(root)
    app.world.body.motor_mode="independent"
    app.world.body.height=.25
    app.world.body.target_posture="lying"
    app.world.body.eyelid_openness=0
    app.world.body.joint_positions=pose(0)
    app.world.body.previous_joints=pose(0)
    server=serve("127.0.0.1",0,BodyToolsApplication(app),"qualification-secret")
    threading.Thread(target=server.serve_forever,daemon=True).start()
    threading.Thread(target=app.run_clock,daemon=True).start()
    try:
        client=Client(f"http://127.0.0.1:{server.server_port}","qualification-secret",attempts=1)
        events=run(client,model,steps=110,interval=.11)
        assert app.world.body.height>=.92, "Learned controller did not stand over HTTP"
        assert all(e["action"]["kind"]=="joint" for e in events)
        assert app.world.body.eyelid_openness==0
        report={"http_learned_stand":True,"actions":len(events),"height":app.world.body.height,
                "eyes_closed":True,"checkpoint_schema":data["schema"],"updated_existing_body":False}
        app.world.action({"kind":"sleep"})
        assert run(client,model,steps=3,interval=0)==[]
        report["sleep_stops_controller"]=True
    finally:
        app.stop.set();server.shutdown();server.server_close();app.close()
# Separate fresh evaluation, no updates selected using these starts.
report["fresh_evaluation"]=assess(model.choose,range(900000,900016))
optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
optimizer.load_state_dict(data["optimizer"])
assert optimizer.state_dict()["state"]
report["optimizer_reload"]=True
Path("runs/arcus_standing_pilot/live-qualification.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report))

