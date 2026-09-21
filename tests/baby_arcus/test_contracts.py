import unittest
from baby_arcus.contracts import EpisodeStart, ContractError, decode, canonical
from baby_arcus.actions import Action
from baby_arcus.config import Settings
from baby_arcus.messages import validate_catalog

class ContractTests(unittest.TestCase):
    def test_start_roundtrip_and_boundaries(self):
        body = {"schema_version":1,"request_id":"r1","family":"clue_search","seed":3}
        self.assertEqual(EpisodeStart.parse(decode(canonical(body))).seed,3)
        for key,value in (("schema_version",True),("schema_version",2),("seed",-1),
                          ("seed",True),("max_steps",0),("request_id","../x"),("family","unknown")):
            bad = dict(body, **{key:value})
            with self.subTest(key=key,value=value), self.assertRaises(ContractError):
                EpisodeStart.parse(bad)
        with self.assertRaises(ContractError):
            EpisodeStart.parse(dict(body,hidden_answer="red"))

    def test_json_and_actions(self):
        for raw in (b'{"a":1,"a":2}',b'{"a":NaN}',b'{'):
            with self.assertRaises(ContractError):
                decode(raw)
        with self.assertRaises(ContractError):
            canonical({"x":float("inf")})
        with self.assertRaises(ContractError):
            Action.parse({"action":"teleport"})
        with self.assertRaises(ContractError):
            Action.parse({"action":"wait","signal":"secret"})

    def test_checked_in_config(self):
        settings = Settings.load("configs/baby_arcus/local.json")
        self.assertEqual(settings.simulation_port,8765)
        validate_catalog(settings.signal_config)
