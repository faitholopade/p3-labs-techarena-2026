"""Linear event calibration using the unchanged pooled training objective."""
import numpy as np
import torch
from torch import nn
from scipy.special import softmax
from calibration_model import normalized
import v33_training
import v45_training
from v45_compact_training import train as compact_method


class LinearCalibrator(nn.Module):
    def __init__(self, dim, width=0, architecture='linear'):
        super().__init__()
        assert width == 0 and architecture == 'linear'
        self.linear = nn.Linear(dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, offset):
        return self.linear(x).squeeze(-1) + offset


def _train(method, module, *args, **kwargs):
    original = module.Calibrator
    try:
        module.Calibrator = LinearCalibrator
        return method(*args, **kwargs)
    finally:
        module.Calibrator = original


def compact_train(*args, **kwargs):
    return _train(compact_method, v45_training, *args, **kwargs)


def dense_train(*args, **kwargs):
    return _train(v33_training.train, v33_training, *args, **kwargs)


@torch.no_grad()
def predict(spec, raw):
    assert spec['architecture'] == 'linear' and spec['width'] == 0
    torch.set_num_threads(3)
    model = LinearCalibrator(len(spec['selected']))
    model.load_state_dict(spec['state']); model.eval(); parts = []
    for start in range(0, len(raw), 40960):
        values = raw[start:start + 40960]
        offset = torch.from_numpy(np.asarray(values[:, spec['offset_column']], np.float32).copy())
        parts.append(model(normalized(values, spec), offset).numpy())
    logits = np.concatenate(parts).reshape(-1, 10)
    return softmax(np.c_[logits, np.zeros(len(logits))], axis=1)[:, :10].ravel()
