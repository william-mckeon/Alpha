import unittest
from baby_arcus.actions import Action
from baby_arcus.world import World
from baby_arcus.lessons import generate
from baby_arcus.baselines import scripted
from baby_arcus.contracts import ContractError

class WorldTests(unittest.TestCase):
    def test_replay_exact(self):
        for family in ("switch_delivery","clue_search"):
            world = generate(family,14)
            initial = world.snapshot()
            records = []
            while not (world.terminated or world.truncated):
                actions = scripted(world)
                result = world.advance(actions)
                records.append((actions,result,world.snapshot()))
            replay = World.restore(initial)
            for actions,result,state in records:
                self.assertEqual(replay.advance(actions),result)
                self.assertEqual(replay.snapshot(),state)

    def test_same_cell_and_swap(self):
        world = generate("clue_search",0)
        world.walls = []
        world.agents["a"]["position"] = [1,2]
        world.agents["b"]["position"] = [3,2]
        world.advance({"a":Action("east"),"b":Action("west")})
        self.assertEqual(world.agents["a"]["position"],[1,2])
        self.assertEqual(world.agents["b"]["position"],[3,2])
        world.agents["b"]["position"] = [2,2]
        world.advance({"a":Action("east"),"b":Action("west")})
        self.assertEqual(world.agents["a"]["position"],[1,2])
        self.assertEqual(world.agents["b"]["position"],[2,2])

    def test_timeout_and_terminal_immutability(self):
        world = generate("switch_delivery",0,max_steps=1)
        result = world.advance({"a":Action(),"b":Action()})
        self.assertTrue(result["truncated"])
        self.assertFalse(result["terminated"])
        state = world.snapshot()
        with self.assertRaises(ContractError):
            world.advance({"a":Action(),"b":Action()})
        self.assertEqual(world.snapshot(),state)

    def test_missing_agent_rejected(self):
        world = generate("switch_delivery",2)
        state = world.snapshot()
        with self.assertRaises(ContractError):
            world.advance({"a":Action()})
        self.assertEqual(state,world.snapshot())
