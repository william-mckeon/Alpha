"""Controlled routing comparisons on retained training contexts; never publishes policies."""
import argparse
import json
import time
from pathlib import Path
import torch
from baby_arcus.checkpoint import load
from baby_arcus.model import precision

def compare(model, contexts, factors=(1.0,1.5,2.0,4.0)):
    if not contexts:
        raise ValueError("At least one retained context is required")
    original=[block.moe.capacity_factor for block in model.core.blocks]
    was_training=model.training
    device=next(model.parameters()).device
    results=[]
    model.eval()
    try:
        for factor in factors:
            if factor<=0:
                raise ValueError("Capacity factor must be positive")
            for block in model.core.blocks:
                block.moe.capacity_factor=factor
            if device.type=="cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started=time.monotonic()
            totals={}
            final_drops=[]
            handles=[]
            # Count the last token separately: actions are predicted from this position.
            def last_drop(module, inputs, output):
                with torch.no_grad():
                    chosen=module.router(inputs[0]).argmax(-1)
                    final=chosen[:,-1:]
                    used=(chosen==final).sum(-1)
                    final_drops.extend((used>module._capacity(chosen.shape[1])).float().cpu().tolist())
            handles=[block.moe.register_forward_hook(last_drop) for block in model.core.blocks]
            try:
                with torch.no_grad(),precision(model):
                    for context in contexts:
                        output=model([context])
                        for key,value in {"overflow":output["overflow"],**output["routing"]}.items():
                            totals[key]=totals.get(key,0.0)+value
            finally:
                for handle in handles:
                    handle.remove()
            if device.type=="cuda":
                torch.cuda.synchronize()
            results.append({"capacity_factor":factor,"contexts":len(contexts),
                "seconds":time.monotonic()-started,"last_token_drop":sum(final_drops)/max(1,len(final_drops)),
                "peak_reserved_gib":torch.cuda.max_memory_reserved()/1024**3 if device.type=="cuda" else None,
                **{key:value/len(contexts) for key,value in totals.items()}})
    finally:
        for block,factor in zip(model.core.blocks,original):
            block.moe.capacity_factor=factor
        model.train(was_training)
    return results

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--experience",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args()
    records=json.loads(Path(args.experience).read_text())
    contexts=[row["context"] for row in records[::max(1,len(records)//32)][:32]]
    model,_,_=load(args.checkpoint,args.device,training=False,restore_rng=False)
    result={"comparison":"same weights and retained contexts; no training or policy publication",
            "results":compare(model,contexts)}
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps([{k:r[k] for k in ("capacity_factor","overflow","last_token_drop","seconds","peak_reserved_gib")} for r in result["results"]]))

if __name__=="__main__":
    main()
