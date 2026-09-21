"""Native loopback integration across all seven independent services."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from baby_arcus.transport import Client,RemoteError

class PipelineTests(unittest.TestCase):
    def test_complete_policy_update_and_viewer(self):
        processes=[]
        ports={}
        for name in ("artifacts","simulation","controller","inference","training","evaluator","dashboard"):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1",0))
                ports[name]=sock.getsockname()[1]
        urls={k:"http://127.0.0.1:"+str(v) for k,v in ports.items()}
        with tempfile.TemporaryDirectory() as root:
            try:
                for name in ports:
                    command=[sys.executable,"-m","baby_arcus.cli","serve",name,"--port",str(ports[name]),
                             "--state-root",root,"--device","cpu"]
                    for key in ("artifact","simulation","inference","training","controller","evaluator"):
                        command += ["--"+key+"-url",urls["artifacts" if key=="artifact" else key]]
                    log=open(Path(root)/(name+".log"),"w")
                    process=subprocess.Popen(command,stdout=log,stderr=log)
                    log.close()
                    processes.append(process)
                controller=Client(urls["controller"],timeout=5)
                deadline=time.monotonic()+40
                for name,url in urls.items():
                    while True:
                        try:
                            Client(url,timeout=1).request("GET","/ready")
                            break
                        except RemoteError:
                            if time.monotonic()>deadline:
                                self.fail("Service startup failed: "+name)
                            time.sleep(.1)
                body={"request_id":"pipeline","seconds":120,"preset":"tiny","max_steps":2,
                      "max_cycles":1,"learning":{"min_samples":8,"microbatch":4,"epochs":1},"evaluate":False}
                response=controller.request("POST","/v1/start",body)
                self.assertEqual(response,controller.request("POST","/v1/start",body))
                deadline=time.monotonic()+135
                while True:
                    status=controller.request("GET","/v1/status")
                    if status["run"]["status"] in ("paused","completed"):
                        break
                    if time.monotonic()>deadline:
                        self.fail("Controller exceeded integration deadline")
                    time.sleep(.2)
                self.assertEqual(status["run"]["status"],"completed",status["run"].get("reason"))
                self.assertEqual(status["run"]["cycles"],1)
                self.assertIsNone(status["resource"]["lease"])
                self.assertEqual(status["run"]["metrics"][0]["samples"],8)
                metric=status['run']['metrics'][0]
                self.assertEqual(metric['diagnostic_samples'],8)
                self.assertGreaterEqual(metric['approx_kl'],0)
                self.assertTrue(0<=metric['clip_fraction']<=1)
                for episode in status['run']['episodes']:
                    diagnostic=episode['diagnostics']
                    self.assertEqual(diagnostic['agent_actions'],2*episode['steps'])
                    self.assertEqual(sum(diagnostic['results'].values()),diagnostic['agent_actions'])
                    self.assertEqual(sum(diagnostic['actions'].values()),diagnostic['agent_actions'])
                    self.assertEqual(diagnostic['objective_completed'],episode['success'])
                self.assertTrue((Path(root)/"controller"/"report.md").exists())
                from urllib.request import urlopen
                with urlopen(urls["dashboard"]) as page:
                    self.assertIn(b"Baby Arcus observatory",page.read())
                viewer=Client(urls["dashboard"])
                viewed=viewer.request("GET","/api/frame?episode="+status["live"]["episode_id"])
                self.assertEqual(viewed["state"]["step"],2)
                self.assertEqual(set(viewed["observations"]),{"a","b"})
                report_list=viewer.request("GET","/api/reports")["reports"]
                self.assertTrue(any(r["run_id"]==status["run"]["run_id"] for r in report_list))
                report=viewer.request("GET","/api/reports/"+status["run"]["run_id"])
                self.assertEqual(report["cycles"],1)
                self.assertEqual(report['episodes'],status['run']['episodes'])
                report_text=(Path(root)/'controller'/'report.md').read_text()
                self.assertIn('approximate KL',report_text)
                self.assertIn('agent_actions',report_text)
                controller.request("POST","/v1/resume",{"request_id":"continue-trained","seconds":120,"max_cycles":1})
                resume_deadline=time.monotonic()+130
                while True:
                    continued=controller.request("GET","/v1/status")
                    if continued["run"]["status"] in ("paused","completed"):
                        break
                    if time.monotonic()>resume_deadline:
                        self.fail("Continuation exceeded its deadline")
                    time.sleep(.1)
                self.assertEqual(continued["run"]["status"],"completed",continued["run"].get("reason"))
                self.assertEqual(continued["run"]["cycles"],2)
                self.assertEqual(continued["run"]["metrics"][-1]["updates"],2)
                self.assertNotEqual(continued["run"]["checkpoint_id"],status["run"]["checkpoint_id"])
                # A one-second run exercises the actual subprocess deadline path.
                controller.request("POST","/v1/start",{**body,"request_id":"deadline-run","seconds":1})
                stop_deadline=time.monotonic()+15
                while True:
                    stopped=controller.request("GET","/v1/status")
                    if stopped["run"]["status"]=="paused" and stopped["resource"]["lease"] is None:
                        break
                    if time.monotonic()>stop_deadline:
                        self.fail("Short run did not pause and release its worker")
                    time.sleep(.1)
                print({"native_services":7,"updates":2,"checkpoint":continued["run"]["checkpoint_id"],
                       "metrics":status["run"]["metrics"][0]})
                if os.environ.get("BABY_ARCUS_VIEWER_SECONDS"):
                    print("VIEWER_URL="+urls["dashboard"],flush=True)
                    end=time.monotonic()+min(300,int(os.environ["BABY_ARCUS_VIEWER_SECONDS"]))
                    while time.monotonic()<end:
                        time.sleep(1)
            finally:
                for process in reversed(processes):
                    process.terminate()
                    process.wait(timeout=10)
                if sys.exc_info()[0]:
                    for path in Path(root).glob("*.log"):
                        print(path.name,path.read_text()[-3000:])
