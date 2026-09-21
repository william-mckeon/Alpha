import unittest
from baby_arcus.lessons import generate
from baby_arcus.observations import observe, visible
from baby_arcus.actions import Action
from baby_arcus.contracts import canonical

class ObservationTests(unittest.TestCase):
    def test_clue_cannot_leak_to_seeker(self):
        for seed in range(0,40,2):
            red,blue = generate("clue_search",seed),generate("clue_search",seed+1)
            seeker = next(k for k,a in red.agents.items() if a["role"] == "seeker")
            scout = next(k for k,a in red.agents.items() if a["role"] == "scout")
            self.assertEqual(observe(red,seeker),observe(blue,seeker))
            self.assertNotEqual(observe(red,scout),observe(blue,scout))
            text = canonical(observe(red,seeker)).decode()
            for forbidden in ("correct_marker","seed","clue:"):
                self.assertNotIn(forbidden,text)

    def test_wall_occlusion(self):
        world = generate("clue_search",2)
        world.walls = [[2,2]]
        self.assertTrue(visible(world,[1,2],[2,2]))
        self.assertFalse(visible(world,[1,2],[3,2]))
        self.assertFalse(visible(world,[1,2],[4,2]))
        world.walls = [[2,1],[1,2]]
        self.assertFalse(visible(world,[1,1],[2,2]), "Cannot see diagonally through wall corners")

    def test_signals_deliver_next_observation_only(self):
        world = generate("clue_search",3)
        self.assertEqual(observe(world,"b")["messages"],[])
        world.advance({"a":Action(signal="help"),"b":Action()})
        self.assertEqual(observe(world,"b")["messages"],[{"sender":"a","signal":"help","step":1}])
        self.assertEqual(observe(world,"a")["messages"],[])
        world.advance({"a":Action(),"b":Action()})
        self.assertEqual(observe(world,"b")["messages"],[])
