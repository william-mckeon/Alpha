"""Experimental local context stages, distinct from demonstrated capability."""
STAGES=(512,2048,8192)


def context_tokens(config):
    value=config.get('context_tokens',512)
    if type(value) is not int or value not in STAGES:
        raise ValueError('Unqualified context stage; supported experimental stages are 512, 2048, 8192')
    return value


def extend_core(model,length):
    from arcus.backbone import build_rope_cache
    context_tokens({'context_tokens':length})
    core=model.core
    device=next(core.parameters()).device
    cos,sin=build_rope_cache(core.cfg.head_dim,length,core.cfg.rope_theta,device=device)
    core.rope_cos=cos; core.rope_sin=sin
    core.cfg.max_seq_len=length
    model.body.cfg.max_seq_len=length
