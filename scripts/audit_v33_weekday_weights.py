"""Clarify the frozen head proxy versus V33's normalized pooled-head share."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['weekdays', 'controls', 'research', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    selected = json.loads((a.research / 'screen/report.json').read_text())['selected']
    assert selected['variant'] == 'small_memory' and selected['mode'] == 'blend' and selected['weight'] == .125
    old = np.load(a.controls / 'v27_predictions.npz')
    memory = np.load(a.controls / 'v29_weekday_predictions.npz')
    new = np.load(a.research / 'screen/small_memory/predictions.npz')
    rows = {}

    def compare(y, h, n):
        proxy = .875 * h + .125 * n
        normalized = (.21875 * h + .125 * n) / .34375
        base_auc = float(roc_auc_score(y, h))
        return {'baseline_head_auc': base_auc,
                'frozen_proxy_gain': float(roc_auc_score(y, proxy) - base_auc),
                'normalized_pool_gain': float(roc_auc_score(y, normalized) - base_auc),
                'normalized_pool_logloss_change': float(log_loss(y, normalized) - log_loss(y, h))}

    for label in ['thursday', 'friday']:
        al = np.load(a.weekdays / label / 'alignment.npz')
        pos = al['first_position']
        idx = (pos[:, None] * 10 + np.arange(10)).ravel()
        rows[label] = compare(al['labels'][idx], .5 * old[label] + .5 * memory[label], new[label])
    forward = np.load(a.research / 'screen/small_memory/forward/predictions.npz')
    rows['forward_friday'] = compare(forward['labels'], forward['baseline'], forward['new_head'])
    report = {'timing': 'Supplementary weight interpretation audit after the frozen screen and confirmations.',
              'frozen_proxy': '0.875*H+0.125*N, H=(base_head+memory_head)/2',
              'full_recipe': '0.875*V31+0.125*N = 0.65625*V20+0.21875*H+0.125*N',
              'normalized_pooled_head_share': '(0.21875*H+0.125*N)/0.34375',
              'results': rows, 'selection_rule_changed': False,
              'limits': 'Both checks are head-only diagnostics. Neither is complete V33/V31 weekday validation; the full recipe also changes the weight on V20.'}
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
