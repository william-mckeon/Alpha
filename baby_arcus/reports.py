"""Reports label implementation, learning smoke, and research milestones separately."""
from pathlib import Path
from baby_arcus.contracts import canonical
from baby_arcus.contracts import identifier

def public_report(state):
    """Bound read-only HTTP views; complete evidence stays in the archived file."""
    from copy import deepcopy
    result=deepcopy({k:v for k,v in state.items() if k not in
                     ("metrics","episodes","evaluation","resource_history","pending_update","gate")})
    result["has_pending_update"]=bool(state.get("pending_update"))
    for key,limit in (("metrics",80),("episodes",100),("resource_history",120)):
        result[key]=state.get(key,[])[-limit:]
        result[key+"_retained_count"]=len(state.get(key,[]))
    def compact_batch(batch):
        row={k:v for k,v in batch.items() if k not in ("provenance","episode_ids","paired_reference")}
        row["provenance_count"]=len(batch.get("provenance",[]))
        if batch.get("paired_reference"):
            row["paired_reference"]=compact_batch(batch["paired_reference"])
        return row
    result["evaluation"]=[compact_batch(batch) for batch in state.get("evaluation",[])[-20:]]
    gate=state.get("gate",{})
    result["gate"]={k:v for k,v in gate.items() if k not in ("seen","reserved_used")}
    result["gate"]["seen_count"]=len(gate.get("seen",[]))
    return deepcopy(result)

def write_report(root,state):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    temporary=root/"report.json.tmp"
    temporary.write_bytes(canonical(state))
    temporary.replace(root/"report.json")
    if state.get("run_id"):
        archive=root/"reports"
        archive.mkdir(exist_ok=True)
        path=archive/(identifier(state["run_id"])+".json")
        tmp=path.with_suffix(".tmp")
        tmp.write_bytes(canonical(state))
        tmp.replace(path)
    lines = ["# Baby Arcus run report","",f"Run: {state.get('run_id','none')}",
             f"Status: {state.get('status','idle')}",f"Checkpoint: {state.get('checkpoint_id','none')}",
             f"Completed update cycles: {state.get('cycles',0)}",
             f"Reason: {state.get('reason') or 'none'}","",
             "No transfer milestone is claimed unless a reserved evaluation explicitly passes.",
             "Human sessions and growth are not implemented in Phase 2.","",
             "## Learning updates",""]
    for metric in state.get("metrics",[]):
        lines.append(f"- Samples {metric.get('samples')}; loss {metric.get('loss')}; prediction {metric.get('prediction')}; overflow {metric.get('overflow')}.")
        if 'approx_kl' in metric:
            lines.append(f"  - During update: approximate KL {metric['approx_kl']}; clipping fraction {metric['clip_fraction']}; {metric.get('diagnostic_samples')} sample presentations across PPO passes. These are pre-step minibatch diagnostics, not final-policy measurements.")
    lines += ["","## Collected episode diagnostics",""]
    for episode in state.get('episodes',[]):
        if episode.get('split') in ('train','practice') and episode.get('diagnostics'):
            lines.append(f"- {episode['split']} {episode['family']} {episode['episode_id']}: {episode['diagnostics']}")
    lines += ["","## Evaluation",""]
    for result in state.get("evaluation",[]):
        lines.append(f"- {result['split']}: {'complete' if result.get('complete') else 'INCOMPLETE'}; checkpoint {result.get('checkpoint_id')}.")
        for family,score in result.get("results",{}).items():
            lines.append(f"  - {family}: {score['wins']}/{score['episodes']}; 95% Wilson interval {score.get('interval95')}.")
        if result.get("reason"):
            lines.append("  - "+result["reason"])
    if state.get("resource_history"):
        lines += ["","## Latest storage sample","",str(state["resource_history"][-1])]
    temporary=root/"report.md.tmp"
    temporary.write_text("\n".join(lines),encoding="utf-8")
    temporary.replace(root/"report.md")
    return state
