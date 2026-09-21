"""Approved practice scheduler; no reward/skill/growth mutation authority."""
from copy import deepcopy
from random import Random
from baby_arcus.contracts import ContractError

FAMILIES = ("switch_delivery","clue_search")

class Curriculum:
    def __init__(self,state=None):
        self.state = deepcopy(state) if state else {
            family:{"level":0,"history":[],"low":0,"recovering":False} for family in FAMILIES}

    def record(self,family,level,wins,total):
        if family not in FAMILIES or total != 50 or not 0<=wins<=total:
            raise ContractError("Practice gate requires 50 episodes per approved family")
        row = self.state[family]
        if level != row["level"]:
            raise ContractError("Practice result does not match current difficulty")
        score = wins/total
        row["history"] = (row["history"]+[score])[-3:]
        row["low"] = row["low"]+1 if score<0.5 else 0
        if row["low"]>=2:
            row["recovering"] = True
        if score>=0.8:
            row["recovering"] = False
        if len(row["history"])==3 and min(row["history"])>=0.8 and row["level"]<2:
            row["level"] += 1
            row["history"] = []
        return deepcopy(row)

    def choose(self,rng):
        family = rng.choice(FAMILIES)
        row = self.state[family]
        level = row["level"]
        if level:
            draw = rng.random()
            if row["recovering"]:
                level = rng.randrange(level) if draw<0.6 else level
            elif draw>=0.6:
                level = rng.randrange(level) if draw<0.9 else 0
        return family,level
