import unittest
from baby_arcus.lessons import generate,validate_solvable
from baby_arcus.baselines import scripted
from baby_arcus.actions import Action

def finish(world,use_messages=True,disable_holder=False):
    while not (world.terminated or world.truncated):
        actions = scripted(world,use_messages=use_messages)
        if disable_holder:
            holder = next(k for k,a in world.agents.items() if a["role"] == "holder")
            actions[holder] = Action()
        world.advance(actions)
    return world.success

class LessonTests(unittest.TestCase):
    def test_many_seeds_solvable_and_roles_vary(self):
        for family in ("switch_delivery","clue_search"):
            roles = set()
            for seed in range(100):
                world = generate(family,seed)
                self.assertTrue(validate_solvable(world))
                roles.add(world.agents["a"]["role"])
                self.assertTrue(finish(world),(family,seed))
            self.assertEqual(len(roles),2)

    def test_message_ablation_balanced_chance(self):
        wins = sum(finish(generate("clue_search",seed),use_messages=False) for seed in range(100))
        self.assertEqual(wins,50)

    def test_switch_needs_partner(self):
        for seed in range(20):
            self.assertFalse(finish(generate("switch_delivery",seed),disable_holder=True))

    def test_generation_reproducible(self):
        self.assertEqual(generate("switch_delivery",91).snapshot(),generate("switch_delivery",91).snapshot())
