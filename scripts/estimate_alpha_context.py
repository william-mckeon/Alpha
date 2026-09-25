"""Allocation arithmetic only; never loads a model or allocates token tensors."""
import argparse
import json


def estimate(tokens, capacity=1.0, element_bytes=4):
    if type(tokens) is not int or tokens < 1 or not 0 < capacity <= 1:
        raise ValueError('Positive token count and valid capacity required')
    if element_bytes not in (2,4): raise ValueError('Use two or four bytes per floating element')
    return {
        'tokens':tokens, 'batch_size':1, 'depth_capacity':capacity,
        'router_single_bool_matrix_bytes':tokens*tokens if capacity < 1 else 0,
        'full_language_logits_bytes':tokens*200019*element_bytes,
        'hypothetical_compact_kv_cache_bytes':tokens*8*2*2*64*element_bytes,
        'dense_attention_pair_count_per_head_per_layer':tokens*(tokens+1)//2,
        'supported_context_claim':False,
        'limitations':'Component arithmetic, not peak memory. KV cache is not implemented. Excludes weights, optimizer, gradients, temporary allocations, routing buffers and other activations.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens',type=int,default=1000000)
    parser.add_argument('--capacity',type=float,default=1.)
    parser.add_argument('--element-bytes',type=int,default=4)
    args=parser.parse_args()
    print(json.dumps(estimate(args.tokens,args.capacity,args.element_bytes),indent=2))
