"""Allocation arithmetic only; never loads a model or allocates token tensors."""
import argparse
import json


def estimate(tokens, capacity=1.0, element_bytes=4):
    if type(tokens) is not int or tokens < 1 or not 0 < capacity <= 1:
        raise ValueError('Positive token count and valid capacity required')
    if element_bytes not in (2,4): raise ValueError('Use two or four bytes per floating element')
    return {
        'tokens':tokens, 'batch_size':1, 'depth_capacity':capacity,
        'legacy_router_single_bool_matrix_bytes':tokens*tokens if capacity < 1 else 0,
        'legacy_full_language_logits_bytes':tokens*200019*element_bytes,
        'compact_kv_cache_bytes':tokens*8*2*2*64*element_bytes,
        'full_depth_padded_expert_one_hidden_tensor_bytes':tokens*4*2432*element_bytes,
        'full_depth_padded_expert_input_buffer_bytes':tokens*4*512*element_bytes,
        'dense_attention_pair_count_per_head_per_layer':tokens*(tokens+1)//2,
        'supported_context_claim':False,
        'limitations':'Batch-one retained baby-125m-cap4 shape. Component arithmetic, not peak memory. Prefix ranking, compact KV and last-only logits are implemented; legacy quadratic router/full-logit figures are not current unavoidable allocations. Padded expert figures describe depth 1 default dispatch. Excludes weights, optimizer, gradients, other simultaneous tensors and backend workspace. Operational context stages still stop at 8192.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens',type=int,default=1000000)
    parser.add_argument('--capacity',type=float,default=1.)
    parser.add_argument('--element-bytes',type=int,default=4)
    args=parser.parse_args()
    print(json.dumps(estimate(args.tokens,args.capacity,args.element_bytes),indent=2))
