"""Shared learned-posture success rules for training, evaluation and live control."""
GOALS=("standing","lying","sitting")
POSTURE_SCHEMA="arcus-posture-v2"
SITTING_SCHEMA="arcus-posture-v3"

def achieved(senses,goal):
    if goal not in GOALS:raise ValueError("Unknown posture goal")
    if not senses["stable"]:return False
    if goal=="standing":return senses["height"]>=.92
    if goal=="sitting":
        q=senses["joint_positions"]
        return (.58<=senses["height"]<=.68 and senses.get("haunch_contact",False)
                and min(v for k,v in q.items() if k.startswith("front"))>=.9
                and max(v for k,v in q.items() if k.startswith("rear"))<=.12)
    # A fall can lower height without tucking the legs; it is not lying down.
    return senses["height"]<=.30 and max(senses["joint_positions"].values())<=.12
