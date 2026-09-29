"""Dense control adapters; pristine donor tensors remain frozen."""
def attach(model, settings):
    if any('.experts.' in name for name, _ in model.named_parameters()):
        raise ValueError('Expanded expert adapter training awaits Phase 6 qualification')
    from peft import LoraConfig, get_peft_model
    model=get_peft_model(model,LoraConfig(r=settings['rank'],lora_alpha=settings['alpha'],
        target_modules=settings['targets'],lora_dropout=0.0,bias='none',task_type='CAUSAL_LM'))
    trainable=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in name for name,p in trainable):
        raise ValueError('Only LoRA parameters may train')
    return model, sum(p.numel() for _,p in trainable)
