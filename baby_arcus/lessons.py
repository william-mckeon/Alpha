"""Procedural variation is reproducible; seeds are never actor observations."""
from random import Random
from collections import deque
from baby_arcus.contracts import ContractError, integer
from baby_arcus.world import World

def generate(family, seed, max_steps=64, layout_id=None, split="train", difficulty=0):
    integer(seed)
    integer(max_steps, 1, 256)
    integer(difficulty,0,2)
    rng = Random(seed // 2)  # Paired seeds differ only in balanced clue answer.
    rotated = rng.randrange(4)
    mirror = rng.choice((False, True))
    roles = rng.sample(["a", "b"], 2)
    def transform(p):
        x, y = p
        if mirror:
            x = 6-x
        for _ in range(rotated):
            x, y = 6-y, x
        return [x, y]
    walls = [[x,y] for x in range(7) for y in range(7) if x in (0,6) or y in (0,6)]
    walls += [[3,y] for y in range(1,6)]
    if layout_id is not None:
        integer(layout_id,0,9)
    metadata = {"split":split,"layout_id":layout_id,"difficulty":difficulty}
    if family == "switch_delivery":
        templates = ((3,2,3,5),(3,4,2,5),(2,1,4,4),(4,5,2,4),(2,2,5,5),
                     (4,4,1,5),(3,1,5,4),(3,5,1,4),(2,5,4,5),(4,1,2,5))
        door_y,plate_y,target_y,target_x = templates[layout_id] if layout_id is not None else (3,2,3,rng.choice((4,5)))
        walls.remove([3,door_y])
        agents = {roles[0]: {"position": transform([1,2]), "role": "holder", "inventory": None},
                  roles[1]: {"position": transform([1,4]), "role": "carrier", "inventory": None}}
        if layout_id is not None and difficulty<2:
            agents[roles[0]]["position"]=transform([2 if difficulty==0 else 1,plate_y])
            agents[roles[1]]["position"]=transform([2 if difficulty==0 else 1,door_y])
            if agents[roles[0]]["position"]==agents[roles[1]]["position"]:
                agents[roles[1]]["position"]=transform([1,3 if door_y!=3 else 4])
        return World(family, seed, agents, [transform(p) for p in walls],
                     door=transform([3,door_y]), plate=transform([2,plate_y]),
                     object_position=transform([target_x,target_y]), delivery=transform([1,3]),
                     max_steps=max_steps,**metadata)
    if family == "clue_search":
        markers = rng.sample(["marker_red","marker_blue"], 2)
        agents = {roles[0]: {"position": transform([1,3]), "role": "scout", "inventory": None},
                  roles[1]: {"position": transform([5,3]), "role": "seeker", "inventory": None}}
        pairs = (((5,2),(5,4)),((4,2),(5,4)),((4,2),(4,4)),((4,3),(5,2)),
                 ((4,1),(4,5)),((5,1),(5,5)),((4,1),(5,5)),((4,1),(4,4)),
                 ((4,1),(5,4)),((5,1),(4,4)))
        positions = pairs[layout_id] if layout_id is not None else ((5,2),(5,4))
        if layout_id is not None:
            agents[roles[1]]["position"]=transform([5 if difficulty<2 else 4,3])
            if difficulty==0:
                agents[roles[1]]["position"]=transform(list(positions[0]))
        return World(family, seed, agents, [transform(p) for p in walls],
                     containers=[{"position": transform(p), "marker": marker}
                                 for p,marker in zip(positions,markers)],
                     clue_position=transform([1,3]),
                     correct_marker=("marker_red","marker_blue")[seed % 2], max_steps=max_steps,**metadata)
    raise ContractError("Unknown lesson family")

def reachable(world, start, goal, open_door=False):
    todo = deque([tuple(start)])
    seen = set(todo)
    while todo:
        pos = todo.popleft()
        if pos == tuple(goal):
            return True
        for dx,dy in ((0,1),(0,-1),(1,0),(-1,0)):
            nxt = (pos[0]+dx,pos[1]+dy)
            if (0 <= nxt[0] < world.size and 0 <= nxt[1] < world.size
                and list(nxt) not in world.walls
                and (list(nxt) != world.door or open_door) and nxt not in seen):
                seen.add(nxt)
                todo.append(nxt)
    return False

def validate_solvable(world):
    if world.family == "switch_delivery":
        holder = next(a for a in world.agents.values() if a["role"] == "holder")
        carrier = next(a for a in world.agents.values() if a["role"] == "carrier")
        return (reachable(world, holder["position"], world.plate)
                and reachable(world, carrier["position"], world.object_position, True)
                and reachable(world, world.object_position, world.delivery, True)
                and not reachable(world, carrier["position"], world.object_position, False)
                and world.plate != world.door)
    scout = next(a for a in world.agents.values() if a["role"] == "scout")
    seeker = next(a for a in world.agents.values() if a["role"] == "seeker")
    return (reachable(world, scout["position"], world.clue_position)
            and all(reachable(world,seeker["position"],c["position"]) for c in world.containers)
            and not reachable(world,seeker["position"],world.clue_position)
            and {c["marker"] for c in world.containers} == {"marker_red","marker_blue"})
