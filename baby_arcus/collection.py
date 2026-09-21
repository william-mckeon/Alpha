"""One complete episode at one frozen checkpoint; no privileged actor input."""
import time
import uuid
from baby_arcus.actions import ACTIONS
from baby_arcus.messages import SIGNALS
from baby_arcus.evaluation import seed_for,layout_for
from baby_arcus.contracts import ContractError

def episode(simulation,inference,checkpoint_id,lease_id,deadline,family,index,split="train",
            difficulty=0,max_steps=64,run_id=None,on_step=lambda r:None,check=lambda:None):
    def policy(operation,state):
        check()
        if time.time()>=deadline:
            raise TimeoutError("Episode deadline")
        return inference.request("POST","/v1/execute",{
            "operation":operation,"checkpoint_id":checkpoint_id,"lease_id":lease_id,"deadline":deadline,
            "episode_id":state["episode_id"],"observations":state["observations"],"policy_seed":seed_for(split,index)})
    state=simulation.request("POST","/v1/episodes",{
        "schema_version":1,"request_id":uuid.uuid4().hex,"family":family,"seed":seed_for(split,index),
        "split":split,"layout_id":layout_for(split,index),"difficulty":difficulty,"max_steps":max_steps,
        "checkpoint_id":checkpoint_id,"run_id":run_id})
    rows=[]
    reward_components={}
    diagnostics={"agent_actions":0,"actions":{},"results":{},"subgoals":{}}
    previous={}
    on_step(state)
    while not (state["terminated"] or state["truncated"]):
        sampled=policy("act",state)
        if sampled["checkpoint_id"]!=checkpoint_id:
            raise ContractError("Policy changed inside episode")
        actions={}
        for agent,row in sampled["agents"].items():
            if agent in previous:
                previous[agent]["next_value"]=row["value"]
            actions[agent]={"action":ACTIONS[row["action"]],"signal":SIGNALS[row["signal"]]}
        check()
        after=simulation.request("POST","/v1/episodes/"+state["episode_id"]+"/step",{
            "schema_version":1,"request_id":uuid.uuid4().hex,"step":state["step"],"actions":actions})
        for component,value in after["transition"]["reward"].items():
            reward_components[component]=reward_components.get(component,0.0)+value
        for agent,row in sampled["agents"].items():
            diagnostics["agent_actions"]+=1
            action=actions[agent]["action"]
            result=after["observations"][agent]["last_result"]
            for key,value in (("actions",action),("results",result)):
                diagnostics[key][value]=diagnostics[key].get(value,0)+1
            observation=after["observations"][agent]
            if observation["inventory"]=="parcel":
                diagnostics["subgoals"]["object_collected"]=True
            if any(cell["position"]==observation["position"] and "plate" in cell["features"] for cell in observation["cells"]):
                diagnostics["subgoals"]["plate_held"]=True
            record={**row,"checkpoint_id":checkpoint_id,"actor":"agent","split":split,
                    "episode_id":state["episode_id"],"agent_id":agent,"step":state["step"],
                    "reward":sum(after["transition"]["reward"].values()),
                    "terminated":after["terminated"],"truncated":after["truncated"],
                    "next_value":0.0,"next_observation":after["observations"][agent]}
            rows.append(record)
            previous[agent]=record
        state=after
        on_step(state)
    if state["truncated"]:
        values=policy("value",state)
        for agent,row in values["agents"].items():
            previous[agent]["next_value"]=row["value"]
    return rows,{"episode_id":state["episode_id"],"family":family,"split":split,"difficulty":difficulty,
                 "success":state["transition"]["success"],"steps":state["step"],
                 "checkpoint_id":checkpoint_id,"reward_components":reward_components,
                 "diagnostics":{**diagnostics,"timed_out":state["truncated"],
                    "selection_made":state["transition"].get("selection_made",False),
                    "objective_completed":state["transition"]["success"]}}
