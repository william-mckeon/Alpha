"""Read-only bounded donor probes under the existing exclusive GPU lock."""
import argparse
import gc
import json
import os
import sys
import time
import resource
from importlib.metadata import version
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcus3.config import read, authorize, deadline, check_live
from arcus3.donor import verify, load, digest
from arcus3.orchestration import invoke_messages
from baby_arcus.gpu_job_control import gpu_job

def run(args):
    authorize(read('/app/configs/arcus3/project.json'), 'inference')
    end = deadline(args.deadline)
    output = Path(args.output)
    if os.environ.get('ARCUS3_CONTROLLED_DOCKER') != '1' or not Path('/.dockerenv').exists():
        raise RuntimeError('Model probes require controlled Docker CUDA')
    if not 1 <= args.max_new_tokens <= 128:
        raise ValueError('Token budget must be 1..128')
    manifest = verify(args.donor)
    check_live(end, output)
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    from transformers import StoppingCriteria, StoppingCriteriaList
    class DeadlineStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            check_live(end, output)
            return False
    records = []
    precision_checks = []
    bf16_peaks = []
    prompts = ['Hi, how are you?', 'Arcus, can you please write me a simple Python for loop?']
    start = time.monotonic()
    with gpu_job():
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.cuda.set_per_process_memory_fraction(0.7)
        torch.manual_seed(0)
        for reload_index in range(2):
            check_live(end, output)
            model, tokenizer = load(args.donor)
            for prompt in prompts:
                check_live(end, output)
                messages = [{'role':'system', 'content':'You are a helpful AI assistant.'},
                            {'role':'user', 'content':prompt}]
                serialized = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(serialized, return_tensors='pt', add_special_tokens=False).to('cuda')
                if inputs.input_ids.shape[1] > 256:
                    raise ValueError('Input budget exceeded')
                torch.cuda.synchronize()
                begin = time.monotonic()
                with torch.inference_mode():
                    def generate(incoming):
                        if incoming != messages:
                            raise ValueError('Orchestration changed the message payload')
                        return model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                              use_cache=True, pad_token_id=tokenizer.eos_token_id,
                                              stopping_criteria=StoppingCriteriaList([DeadlineStop()]))
                    result = invoke_messages(generate, messages) if reload_index else generate(messages)
                    torch.cuda.synchronize()
                    generation_seconds = time.monotonic()-begin
                    ids = result[0, inputs.input_ids.shape[1]:].tolist()
                    full = model(**inputs, use_cache=False).logits[:, -1, :].float()
                    prefix = model(input_ids=inputs.input_ids[:, :-1],
                                   attention_mask=inputs.attention_mask[:, :-1], use_cache=True)
                    cached = model(input_ids=inputs.input_ids[:, -1:], attention_mask=inputs.attention_mask,
                                   past_key_values=prefix.past_key_values, use_cache=True).logits[:, -1, :].float()
                    difference = (full-cached).abs().max().item()
                    top_equal = full.argmax(-1).item() == cached.argmax(-1).item()
                    # BF16 raw logit deltas are diagnostic; independent FP32 checks below
                    # distinguish precision effects from a cache implementation failure.
                    if not top_equal:
                        raise RuntimeError('BF16 cached/full next-token choice differs')
                torch.cuda.synchronize()
                records.append({'reload_index':reload_index,
                    'execution':'langchain-runnable-in-langgraph' if reload_index else 'direct',
                    'messages':messages, 'serialized_prompt':serialized,
                    'input_ids':inputs.input_ids[0].tolist(), 'output_ids':ids,
                    'response':tokenizer.decode(ids, skip_special_tokens=True),
                    'truncated':len(ids)==args.max_new_tokens and ids[-1]!=tokenizer.eos_token_id,
                    'generation_seconds':generation_seconds,
                    'seconds_including_cache_check':time.monotonic()-begin,
                    'bf16_exceeds_original_diagnostic_limit':difference > 0.25,
                    'cache_max_abs_error':difference, 'cache_argmax_equal':top_equal})
                print(json.dumps(records[-1]), flush=True)
                del result, full, prefix, cached, inputs
            bf16_peaks.append(torch.cuda.max_memory_allocated())
            # Run only after BF16 generations. Do not cast back: that would also cast
            # original FP32 rotary buffers. The next pass reloads the pristine donor.
            model.float()
            for record in records[-2:]:
                check_live(end, output)
                inputs = tokenizer(record['serialized_prompt'], return_tensors='pt', add_special_tokens=False).to('cuda')
                with torch.inference_mode():
                    full = model(**inputs, use_cache=False).logits[:, -1, :]
                    prefix = model(input_ids=inputs.input_ids[:, :-1], attention_mask=inputs.attention_mask[:, :-1], use_cache=True)
                    cached = model(input_ids=inputs.input_ids[:, -1:], attention_mask=inputs.attention_mask,
                                   past_key_values=prefix.past_key_values, use_cache=True).logits[:, -1, :]
                    error = (full-cached).abs().max().item()
                    equal = full.argmax(-1).item() == cached.argmax(-1).item()
                    precision_checks.append({'reload_index':reload_index, 'input_ids':record['input_ids'],
                                             'dtype':'float32', 'max_abs_error':error, 'argmax_equal':equal})
                    if error > 0.001 or not equal:
                        raise RuntimeError(f'FP32 reference cache check failed: {error}')
                del inputs,full,prefix,cached
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        if any(records[i]['output_ids'] != records[i+2]['output_ids'] for i in range(2)):
            raise RuntimeError('Reload output mismatch')
        report = {'status':'passed', 'repo_id':manifest['repo_id'], 'revision':manifest['revision'],
            'donor_manifest_sha256':digest(Path(args.donor)/'manifest.json'), 'unique_parameters':1711376384,
            'dtype':'bfloat16', 'attention':'sdpa', 'greedy':True, 'max_new_tokens':args.max_new_tokens,
            'device':torch.cuda.get_device_name(), 'torch':torch.__version__,
            'dependencies':{name:version(name) for name in ('transformers','tokenizers','huggingface-hub','safetensors','accelerate','langchain','langchain-core','langgraph')},
            'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'bf16_peak_cuda_allocated':max(bf16_peaks),
            'precision_checks':precision_checks,
            'seconds':time.monotonic()-start, 'reload_token_parity':True, 'records':records,
            'limitations':['Two short prompts; not the full baseline suite or an 8k-context capability test.',
                          'Cache comparison covers next-token logits, not exhaustive long-sequence parity.']}
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print('Probe and reload verification passed', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--donor', default='/donor')
    parser.add_argument('--output', default='/output')
    parser.add_argument('--deadline', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        Path(args.output, 'error.json').write_text(json.dumps({'error':str(exc), 'type':type(exc).__name__}))
        raise
