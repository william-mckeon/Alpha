"""Transformers custom-code loader for Alpha 3 model packages."""

from transformers import LlamaConfig, LlamaForCausalLM

from .routing import SelectiveExperts


class Alpha3Config(LlamaConfig):
    model_type = "alpha3_selective_experts"

    def __init__(
        self,
        selected_layers=None,
        depth_enabled=False,
        depth_capacity=1.0,
        added_parameters_dtype="inherit",
        arcus_release=None,
        **kwargs,
    ):
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)
        default_layers = [3, 7, 11, 15, 19, 23]
        self.selected_layers = default_layers if selected_layers is None else selected_layers
        if self.selected_layers != default_layers:
            raise ValueError("unsupported Alpha 3 architecture")
        if type(depth_enabled) is not bool:
            raise ValueError("depth_enabled must be boolean")
        if depth_enabled and depth_capacity != 1.0:
            raise ValueError("model packages require full depth")
        if added_parameters_dtype not in ("inherit", "float32"):
            raise ValueError("unsupported added-parameter dtype")
        self.depth_enabled = depth_enabled
        self.depth_capacity = depth_capacity
        self.added_parameters_dtype = added_parameters_dtype
        # Keep the on-disk field name used by existing Alpha 3 packages.
        self.arcus_release = {} if arcus_release is None else arcus_release


class Alpha3ForCausalLM(LlamaForCausalLM):
    config_class = Alpha3Config

    @classmethod
    def _load_pretrained_model(cls, model, *args, **kwargs):
        # Transformers 4.46 passes a legacy dtype into the loader. Let each
        # destination parameter preserve its constructed dtype instead.
        if model.config.added_parameters_dtype == "float32" and "dtype" in kwargs:
            kwargs["dtype"] = None
        return super()._load_pretrained_model(model, *args, **kwargs)

    def __init__(self, config):
        super().__init__(config)
        for index in config.selected_layers:
            self.model.layers[index].mlp = SelectiveExperts(
                self.model.layers[index].mlp
            )
            block = self.model.layers[index].mlp
            if config.depth_enabled:
                from .depth import FullDepthGate

                block.depth_gate = FullDepthGate(
                    block.router.in_features,
                    block.router.weight.device,
                    config.depth_capacity,
                )
            if config.added_parameters_dtype == "float32":
                block.experts[1].float()
                block.router.float()
                if config.depth_enabled:
                    block.depth_gate.float()
