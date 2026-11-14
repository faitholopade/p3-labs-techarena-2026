"""Run the frozen V44 sampler comparison through the unchanged factor trainer."""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import refine_exposure_prior as trainer
import local_exposure_sampling as local
from rebuild_benchmark import sha, write, runtime_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['runtime', 'benchmark', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cohort', choices=['thursday', 'friday', 'earlier', 'later', 'full'], required=True)
    parser.add_argument('--variant', choices=['cluster_control', 'local75'], required=True)
    parser.add_argument('--plan', type=Path, default=REPO / 'results/v44_experiment_plan.json')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    cut = plan['cutoffs'][args.cohort]
    if args.cohort in ['thursday', 'friday']:
        factor = args.benchmark / f'inputs/factors/{args.cohort}.pt'
        basis = args.benchmark / f'inputs/bases/{cut}/exposure.npz'
    elif args.cohort == 'full':
        factor = runtime_path(args.runtime, 17) / 'models/factor_users_only.pt'
        basis = runtime_path(args.runtime, 11) / 'data/exposure_basis.npz'
    else:
        factor = runtime_path(args.runtime, 17) / f'validation/factors/{args.cohort}_factor_users_only.pt'
        basis = runtime_path(args.runtime, 13) / f'validation/source_states/{cut}/exposure.npz'
    root = runtime_path(args.runtime, 11)
    output = args.out / args.cohort / args.variant
    output.mkdir(parents=True, exist_ok=True)
    inputs = {'plan_sha256': sha(args.plan), 'factor_sha256': sha(factor), 'basis_sha256': sha(basis),
              'data_sha256': sha(root / 'data/train.npz'), 'cohort': args.cohort, 'variant': args.variant,
              'source_sha256': {p: sha(REPO / p) for p in ['src/local_exposure_sampling.py',
                  'src/refine_exposure_prior.py', 'src/residual_model.py', 'src/variational.py',
                  'scripts/train_v44_factor.py']}}
    if (output / 'completed.json').exists():
        report = json.loads((output / 'completed.json').read_text())
        assert report['inputs'] == inputs and sha(output / 'residual.pt') == report['model_sha256']
        print({'reused': str(output)}, flush=True)
        return
    if (output / 'residual.pt').exists():
        raise ValueError('Unverified existing factor; use a fresh output directory.')
    write(output / 'inputs.json', inputs)
    mixture = plan['variants'][args.variant]['local_mixture']
    trainer.pooled = lambda fit, ni, nu, cut, basis, cache: local.build(
        fit, ni, nu, cut, basis, cache, neighbors=plan['neighbors'])
    trainer.sample_pool = lambda pool, users, rng: local.sample(pool, users, rng, mixture)
    configuration = plan['training']
    sys.argv = ['refine_exposure_prior.py', '--root', str(root), '--init', str(factor), '--exposure', str(basis),
                '--cut', str(cut), '--out', str(output), '--pool-cache', str(args.out / 'pools'), '--antithetic']
    for name, value in configuration.items():
        sys.argv += ['--' + name.replace('_', '-'), str(value)]
    trainer.main()
    write(output / 'completed.json', {'inputs': inputs, 'model_sha256': sha(output / 'residual.pt'),
                                     'training_sha256': sha(output / 'training.json'),
                                     'local_mixture': mixture, 'portal_uploads': 0})


if __name__ == '__main__':
    main()
