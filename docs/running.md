# Running the code

## Checks that work from this checkout

Use Python 3.12:

```bash
python -m pip install -r requirements-tests.txt
python -m unittest discover -s tests -v
python scripts/check_source_archive.py
```

The tests call the original V142 `predict` function using small generated arrays
and exercise the original temporal split. The source check parses all Python,
checks JSON, and verifies the original source hashes. These are engineering
checks, not a new training experiment or competition evaluation.

## Historical entry points

The original layout is preserved: `src/`, `scripts/`, `experiments/`, `artifacts/`
and `runtime/`. The first four contain source or metadata here. Runtime data,
model checkpoints, fitted feature caches and submission predictions must be
supplied separately. Release specifications describe required historical locations and hashes.
The private artifact index is omitted. The retained `import_artifacts.py` helper
also needs that private `artifacts/manifest.json` before it can import bundles.

The following commands document the original entry points. They will not run
successfully without the required private inputs.

**V65 saved-model inference** expects the nine V11–V20 historical archives
(V11, V12, V13, V15, V16, V17, V18, V19 and V20), the later small models and
canonical dependency files named in the release specifications:

```bash
python src/infer_after_v63.py --runtime runtime --repo . \
  --spec artifacts/v65/spec.json --out work/v65.csv --threads 3
```

This is the original exact-hash entry point. It can stop at an ancestor CSV hash
when numerical library differences change the bytes. The private recovery used
separate compatibility copies and a final numerical check; those are not being
presented here as a full reproduction fix.

**V142 fixed-evaluation replay** expects these three files:

- `runtime/v142/production_model.npz`: catalog, mean, standard deviation and weights.
- `runtime/v142/production_inputs.npz`: candidate items, contexts and row IDs.
- `runtime/v142/production_baseline.npz`: saved V136 probabilities in matching order.

```bash
python experiments/v142/replay.py --out work/v142.csv
```

Original CSV SHA-256:
`23897eb33f1d288349838f34493608e52578551d697a17d9addc731af433a2dc`.
This command applies a correction to stored predictions for fixed rows. It does
not retrain V136 or produce a baseline for new rows.

**V141 correction fitting** is in `experiments/v141/research.py --stage run
--cohort N`. Its factor and pool dependencies are defined in
`experiments/v138/research.py`. These include private V119-era factor states,
training arrays, negative-sampling pools and V136 validation evidence. Review
the paths before running. Full original V108/V136 base-training recovery remains
incomplete; this repository does not fill those gaps with replacement code.

## Environment

`requirements.txt` lists the packages used by the historical source; it is not
an exact training lockfile. The 7 November saved-model recovery used Python 3.12,
NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, CPU PyTorch 2.5.1,
LightGBM 4.6.0 and Numba 0.62.1. Those versions record one verified inference
environment, not a promise of identical training results on every platform.

Some historical scripts load PyTorch checkpoints with `weights_only=False`.
Only use trusted original artifacts. The public checks do not load checkpoints.
