import unittest
from baby_arcus.services.dashboard import DashboardApplication
from baby_arcus.transport import StaticResponse
from baby_arcus.contracts import ContractError

class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.app=DashboardApplication({name:"http://127.0.0.1:1" for name in ("controller","simulation","artifact")})

    def test_packaged_assets_and_read_only_routes(self):
        for name in ("index.html","app.js","api.js","world.js","agent-panel.js","replay.js","training-panel.js","styles.css"):
            code,response=self.app("GET","/"+name,None)
            self.assertEqual(code,200)
            self.assertIsInstance(response,StaticResponse)
            self.assertTrue(response.data)
        for method,path in (("POST","/v1/start"),("GET","/../README.md"),("GET","/services/controller.py")):
            with self.assertRaises(KeyError):
                self.app(method,path,{})

    def test_report_proxy_uses_validated_id(self):
        calls=[]
        class Controller:
            def request(self,method,path,body=None):
                calls.append((method,path))
                return {"run_id":"report"}
        self.app.clients["controller"]=Controller()
        self.assertEqual(self.app("GET","/api/reports/report",None)[1]["run_id"],"report")
        with self.assertRaises(ContractError):
            self.app("GET","/api/reports/../../report",None)
        self.assertEqual(calls,[("GET","/v1/reports/report")])

