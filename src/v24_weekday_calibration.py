"""Compact event calibration on genuinely out-of-time weekday factor scores."""
import numpy as np
import lightgbm as lgb
from scipy.special import logit, expit
from v24_causal_propensity import features

PARAMS = {'objective': 'binary', 'metric': 'None', 'learning_rate': .025,
          'num_leaves': 7, 'min_data_in_leaf': 1000, 'lambda_l2': 100.,
          'max_bin': 127, 'feature_fraction': 1., 'bagging_fraction': 1.,
          'num_threads': 2, 'seed': 20261029, 'verbosity': -1,
          'force_col_wise': True, 'deterministic': True}
ROUNDS = 200


def build(query, state, factor_probability):
    raw, _, names = features(query, state)
    p = np.asarray(factor_probability, dtype=np.float64)
    if p.shape != (len(query['user']), 10) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('Invalid frozen factor probabilities.')
    mass = p.sum(1)
    if np.any((mass <= 0) | (mass >= 1)):
        raise ValueError('Factor event mass must be strictly between zero and one.')
    conditional = p / mass[:, None]
    entropy = -(conditional * np.log(np.maximum(conditional, 1e-12))).sum(1)
    geometry = np.column_stack([entropy, p.max(1), mass, np.square(p).sum(1)])
    names = names + ['factor_choice_entropy', 'factor_max_probability',
                     'factor_event_mass', 'factor_sum_squared_probability']
    return np.column_stack([raw, geometry]).astype(np.float32), logit(mass), names


def train(raw, offset, event_labels, names):
    if raw.shape[0] != len(offset) or len(offset) != len(event_labels):
        raise ValueError('Training feature/label alignment failed.')
    return lgb.train(PARAMS, lgb.Dataset(raw, label=event_labels,
                     init_score=offset, feature_name=names), num_boost_round=ROUNDS)


def predict(model, raw, offset, factor_probability):
    event = expit(model.predict(raw, raw_score=True, num_threads=2) + offset)
    p = np.asarray(factor_probability, dtype=np.float64)
    candidate = p / p.sum(1, keepdims=True) * event[:, None]
    if not np.isfinite(candidate).all() or np.any(candidate < 0) or np.any(candidate.sum(1) > 1):
        raise ValueError('Invalid recalibrated probabilities.')
    if not np.allclose(candidate.sum(1), event, rtol=1e-12, atol=1e-12):
        raise ValueError('Event mass was not preserved.')
    return candidate
