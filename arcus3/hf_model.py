"""Alpha 3.0 initialization architecture for explicit Transformers custom-code loading."""
from transformers import LlamaConfig,LlamaForCausalLM
from .routing import SelectiveExperts

class Alpha3Config(LlamaConfig):
    model_type='alpha3_selective_experts'
    def __init__(self,selected_layers=None,**kwargs):
        kwargs.pop('model_type',None)
        super().__init__(**kwargs)
        self.selected_layers=[3,7,11,15,19,23] if selected_layers is None else selected_layers
        if self.selected_layers!=[3,7,11,15,19,23]:raise ValueError('Unsupported Alpha 3.0 architecture')

class Alpha3ForCausalLM(LlamaForCausalLM):
    config_class=Alpha3Config
    def __init__(self,config):
        super().__init__(config)
        for index in config.selected_layers:
            self.model.layers[index].mlp=SelectiveExperts(self.model.layers[index].mlp)
