import torch
from torch import nn

from copy import deepcopy
from collections import OrderedDict
from sys import stderr

# for type hint
from torch import Tensor


# https://www.zijianhu.com/post/pytorch/ema/
class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float):
        super().__init__()
        self.decay = decay

        self.model = model
        self.ema_model = deepcopy(self.model)

        for param in self.ema_model.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        ema_model_params = OrderedDict(self.ema_model.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == ema_model_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            ema_model_params[name].sub_((1. - self.decay) * (ema_model_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        ema_model_buffers = OrderedDict(self.ema_model.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == ema_model_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            ema_model_buffers[name].copy_(buffer)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.ema_model(inputs)

    def get_ema_model(self):
        return self.ema_model

    def get_model(self):
        return self.model
