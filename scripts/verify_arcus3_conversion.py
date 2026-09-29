"""Shared bounded parity measurements; invoked by the controlled conversion job."""
import torch


def measure(model, tokenizer):
    # Keep the batch convention explicit after tokenizer reload; padding_side is
    # an inference setting and need not be persisted by save_pretrained.
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='left'
    prompts=['Hi, how are you?', 'Write a simple Python for loop.',
             'The quick brown fox jumps over the lazy dog. '*8]
    result={}
    for index, text in enumerate(prompts):
        x=tokenizer(text,return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
        with torch.inference_mode():
            full=model(x,labels=x,use_cache=False)
            result[str(index)+'_logits']=full.logits.cpu()
            result[str(index)+'_loss']=full.loss.float().cpu()
            prefix=model(x[:,:-1],use_cache=True)
            cached=model(x[:,-1:],past_key_values=prefix.past_key_values,use_cache=True).logits
            result[str(index)+'_cached']=cached.cpu()
            result[str(index)+'_generated']=model.generate(x,max_new_tokens=16,do_sample=False,
                pad_token_id=tokenizer.eos_token_id).cpu()
    # A left-padded batch exercises attention masks without changing routing's
    # token-local semantics. Padding positions still execute an FFN, as in donor.
    batch=tokenizer(prompts[:2],return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
    with torch.inference_mode():result['masked_logits']=model(**batch,use_cache=False).logits.cpu()
    return result


def compare(before, after):
    rows={}
    for key,a in before.items():
        b=after[key]
        if a.shape!=b.shape:raise ValueError('Parity shape mismatch: '+key)
        exact=torch.equal(a,b)
        error=float((a.float()-b.float()).abs().max())
        rows[key]={'exact':exact,'max_abs':error}
    if not all(row['exact'] for row in rows.values()):raise ValueError('Initial conversion parity failed: '+str(rows))
    return rows
