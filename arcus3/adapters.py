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


def attach_expanded(model, settings):
    import torch
    from torch import nn
    from arcus3.routing import SelectiveExperts
    import math
    class ExpertLoRA(nn.Module):
        def __init__(self, base):
            super().__init__();self.base=base
            self.lora_A=nn.Parameter(torch.empty(settings['rank'],base.in_features,device=base.weight.device,dtype=torch.float32))
            self.lora_B=nn.Parameter(torch.zeros(base.out_features,settings['rank'],device=base.weight.device,dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
            self.scale=settings['alpha']/settings['rank']
        def forward(self,x):
            delta=(x.float() @ self.lora_A.T @ self.lora_B.T)*self.scale
            return self.base(x)+delta.to(x.dtype)
    model.requires_grad_(False);count=0
    for module in list(model.modules()):
        if isinstance(module,SelectiveExperts):
            count+=1;module.router.float().requires_grad_(True)
            for expert in module.experts:
                for name in ('gate_proj','up_proj','down_proj'):
                    base=getattr(expert,name)
                    if not isinstance(base,nn.Linear):raise ValueError('Adapter already attached')
                    setattr(expert,name,ExpertLoRA(base))
    if not count:raise ValueError('Expanded model required')
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
