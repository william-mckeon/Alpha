import unittest
from baby_arcus.lessons import generate
from baby_arcus.actions import Action
from baby_arcus.baselines import scripted
from baby_arcus.rewards import award

class RewardTests(unittest.TestCase):
    def test_event_repetition_cannot_farm(self):
        world = generate("switch_delivery",2)
        first = award(world,["plate_held","object_collected","object_collected"],False)
        self.assertAlmostEqual(first["intermediate"],0.2)
        for _ in range(10):
            self.assertEqual(award(world,["plate_held","object_collected"],False)["intermediate"],0)

    def test_actual_completion_and_invalid_actions(self):
        world = generate("switch_delivery",7)
        result = world.advance({"a":Action("drop"),"b":Action("interact")})
        self.assertEqual(sum(result["reward"].values()),0)
        total = 0
        while not (world.terminated or world.truncated):
            result = world.advance(scripted(world))
            total += sum(result["reward"].values())
        self.assertTrue(world.success)
        self.assertAlmostEqual(total,1.2)

    def test_wrong_choice_has_no_correctness_shaping(self):
        world = generate("clue_search",1)
        seeker = next(k for k,a in world.agents.items() if a["role"] == "seeker")
        wrong = next(c for c in world.containers if c["marker"] != world.correct_marker)
        world.agents[seeker]["position"] = wrong["position"]
        actions = {k:Action("interact") if k == seeker else Action() for k in world.agents}
        result = world.advance(actions)
        self.assertTrue(result["terminated"])
        self.assertFalse(result["success"])
        self.assertEqual(sum(result["reward"].values()),0)
