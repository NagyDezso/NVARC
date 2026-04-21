import copy
import io

import torch
import torch.nn as nn


def _ema_inner_module(module: nn.Module) -> nn.Module:
    """Unwrap ``DataParallel`` and ``torch.compile`` so shadow keys match ``deepcopy(_orig_mod)``.

    Shadow must be keyed like ``_orig_mod.named_parameters()`` because ``ema_copy`` clones
    the inner module only (the compile wrapper cannot be ``deepcopy``'d).
    """
    if isinstance(module, nn.DataParallel):
        module = module.module
    return getattr(module, "_orig_mod", module)


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        m = _ema_inner_module(module)
        for name, param in m.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        m = _ema_inner_module(module)
        for name, param in m.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        m = _ema_inner_module(module)
        for name, param in m.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        inner = _ema_inner_module(module)
        try:
            module_copy = copy.deepcopy(inner)
        except RuntimeError:
            # e.g. ``weight_norm`` / other non-leaf state on ``inner``
            buf = io.BytesIO()
            torch.save(inner, buf)
            buf.seek(0)
            device = next(inner.parameters()).device
            try:
                module_copy = torch.load(buf, map_location=device, weights_only=False)
            except TypeError:
                module_copy = torch.load(buf, map_location=device)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

