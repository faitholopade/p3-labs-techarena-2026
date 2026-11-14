"""Smoke-check learned slate permutation behavior and mixture algebra."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from calibration_model import Calibrator, normalized, predict
from daily_training import train
from screen_v34_interactions import pooled_head, changed_head, full_prediction


def main():
    names = json.loads(Path(sys.argv[1]).read_text())
    rng = np.random.default_rng(3401)
    x = rng.normal(size=(600, len(names))).astype(np.float32)
    y = np.zeros((60, 10), np.float32)
    y[np.arange(40), rng.integers(0, 10, 40)] = 1
    ds = [{'x': x, 'labels': y.ravel(), 'names': np.asarray(names)}]
    report = {}
    for architecture in ['cross', 'set']:
        ck = train(ds, [np.ones(60, bool)], seed=3401, epochs=3, width=32,
                   rank_weight=.5, architecture=architecture, threads=2,
                   offset_name='posterior1.0_mix1.0_utility')
        net = Calibrator(len(ck['selected']), 32, architecture)
        net.load_state_dict(ck['state']); net.eval()
        xx = normalized(x, ck); oo = torch.from_numpy(x[:, ck['offset_column']].copy())
        perm = rng.permutation(10)
        indices = (np.arange(60)[:, None] * 10 + perm).ravel()
        with torch.no_grad():
            original = net(xx, oo).numpy().reshape(60, 10)
            moved = net(xx[indices], oo[indices]).numpy().reshape(60, 10)
        error = float(np.max(np.abs(moved - original[:, perm])))
        assert error < 1e-5
        key = 'cross_out.weight' if architecture == 'cross' else 'event.3.weight'
        assert float(ck['state'][key].abs().max()) > 0
        prediction = predict(ck, x)
        assert np.isfinite(prediction).all() and prediction.min() >= 0
        assert prediction.reshape(60, 10).sum(1).max() <= 1 + 1e-8
        report[architecture] = {'learned_branch_nonzero': True, 'permutation_max_error': error,
                                'event_probabilities_valid': True, 'training_events': 60}
    data = [rng.dirichlet(np.ones(11), 12)[:, :10].ravel() for _ in range(5)]
    plan = json.loads((ROOT / 'results/v34_experiment_plan.json').read_text())
    errors = {}
    for recipe in plan['mixtures']:
        full = full_prediction(*data, recipe)
        pool = changed_head(*data[1:], recipe)
        parent_weight = .65625 * (1 - recipe['weight']) if recipe['mode'] == 'blend' else .65625
        error = float(np.max(np.abs(full - (parent_weight * data[0] + (1 - parent_weight) * pool))))
        assert error < 3e-16
        errors[recipe['mode']] = error
    report['normalized_head_formula_max_errors'] = errors
    report['scope'] = 'Tiny synthetic smoke checks, not validation AUC or a performance result.'
    Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
