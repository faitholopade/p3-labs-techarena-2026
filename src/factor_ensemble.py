"""Rebuild the scalar V9 features affected by a preference-factor checkpoint."""
import numpy as np
from residual_model import load, features as residual_features
from posterior_inference import features as posterior_features
from uncertainty import logstd_summary,numpy_direction


def update_features(raw, events, model_path, posterior_samples=64,exact_direction=False):
    """Update columns 209:252 in place; all other chronological features stay fixed."""
    net = load(model_path)
    net.exact_direction=exact_direction
    residual, _ = residual_features(net, events)
    posterior, _ = posterior_features(net.state_dict(), events, samples=posterior_samples,exact_direction=exact_direction)
    state = net.state_dict()
    metadata = []
    for kind, indices in [('user', events['user']), ('item', events['items'])]:
        mean = state[kind + '.weight'].numpy()[indices]
        uncertainties = state[kind + '_logstd.weight'].numpy()[indices]
        logstd = logstd_summary(uncertainties)
        norm = np.linalg.norm(mean, axis=-1)
        certainty = norm / np.sqrt(norm ** 2 + 7 * np.exp(2 * logstd))
        if uncertainties.shape[-1]>1 or exact_direction:
            certainty=np.linalg.norm(numpy_direction(mean,np.exp(np.clip(uncertainties,-3,1)),exact=exact_direction),axis=-1)
        for values in [norm, logstd, certainty]:
            if values.ndim == 1:
                values = np.repeat(values[:, None], 10, axis=1)
            metadata.append(values.ravel())
    metadata.append(np.repeat(state['outside.weight'].numpy()[events['user'], 0], 10))
    raw[:, 209:252] = np.c_[residual, posterior, np.stack(metadata, axis=1)]
    return raw

