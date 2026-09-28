"""Reusable deterministic inference, without optimizer updates or synthetic answers."""
import time
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from arcus.kv_cache import KVCache
from baby_arcus.shared_curriculum import example
from baby_arcus.conversation_format import pack
from baby_arcus.routing_trace import route_phase

IDENTITY = 'You are Arcus, the techno-dragon powered by the Alpha model family.'


def respond(model, tokenizer, prompt, max_new_tokens=128, tool_schema=None):
    if model.training or not 1 <= max_new_tokens <= 128:
        raise ValueError('Probe requires eval mode and a bounded generation')
    system = IDENTITY
    if tool_schema:
        import json
        system += '\nAvailable tool: ' + json.dumps(tool_schema, sort_keys=True) + '\nReturn one JSON object with name and arguments.'
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]
    _, ids = pack(messages, tokenizer, model.body.cfg.max_seq_len - max_new_tokens)
    row, _, _ = example(0, 'training', 'commands')
    row['hearing'] = []
    generated = []
    started = time.monotonic()
    device = next(model.parameters()).device
    from contextlib import nullcontext
    attention = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) if device.type == 'cuda' else nullcontext()
    with torch.inference_mode(), attention:
        hidden = model([row], tokenizer, requested=('hidden',))['hidden']
        residual = torch.nn.functional.linear(model.text_context(hidden), model.language.embedding.weight)
        cache = KVCache(model.body.cfg.max_seq_len) if model.core.supports_cache() else None
        with route_phase('language-generation'):
            for _ in range(max_new_tokens):
                current = [generated[-1]] if cache is not None and generated else ids + generated
                tokens = torch.tensor([current], device=device)
                logits = (model.language.cached_logits(model.core, tokens, cache) if cache is not None else
                          model.language(model.core, tokens, last_only=True)[:, -1]) + residual
                generated.append(int(logits.argmax(-1)[0]))
                if '\n</assistant>' in tokenizer.decode(generated):
                    break
    raw = tokenizer.decode(generated)
    return {'messages': messages, 'response': raw.split('\n</assistant>')[0], 'raw_response': raw,
            'input_tokens': len(ids), 'generated_tokens': len(generated), 'token_ids': generated,
            'decoding': 'greedy', 'max_new_tokens': max_new_tokens,
            'truncated': len(generated) == max_new_tokens and '\n</assistant>' not in raw,
            'seconds': time.monotonic() - started, 'cache': cache is not None}
