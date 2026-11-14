# P3 Labs — Huawei Ireland Tech Arena 2026

Our solution to the click-through prediction challenge. **P3 Labs was listed ninth among the teams advancing to Phase 2** in the organizer's announcement on 9 November 2026. Our best reported submission, V142, scored **0.832748 ROC AUC**.

Each event showed a user ten items. We had to predict the probability of a click on every item, allowing for the user to click none of them. The data included up to 20 previous clicks per user. The score compared predictions across all events, so we needed to learn both what someone liked and how likely they were to click at all.

We built user and item representations from click history and exposure data, then combined their scores with popularity, activity and context features. Calibration models turned those features into probabilities over the ten items and a no-click option. Later versions added smaller corrections: V63 adapted the model to a user's first query of the day, V65 adjusted the probability of any click, and V142 adjusted item preferences between weekdays and weekends while keeping the total click probability unchanged.

This repository contains the original model code, training and experiment scripts, inference code, and selected result records. The version numbers are the names used during the competition.

| Start here | What it contains |
| --- | --- |
| [Model guide](docs/model.md) | How the components fit together, with links to the implementation |
| [src/](src) | Original feature construction, models, training routines and inference chain |
| [scripts/](scripts) | Training runs, validation, model selection and submission packaging |
| [experiments/v141/research.py](experiments/v141/research.py) | Weekend preference fitting used for V142 |
| [experiments/v142/](experiments/v142) | V142 confirmation, submission construction and saved-prediction replay |
| [Results](docs/results.md) | Submitted scores, validation results and later experiments |
| [Running the code](docs/running.md) | Environment, archive requirements and reproduction status |

The original dataset, trained checkpoints and saved predictions are not included. The source can be inspected and the checks below run without them; reproducing a submission requires the corresponding private artifacts. Some later training dependencies remain incomplete in the recovered archive. The running guide identifies those gaps.

```bash
python -m pip install -r requirements-tests.txt
python -m unittest discover -s tests -v
python scripts/check_source_archive.py
```

These checks exercise the actual V142 prediction function, the historical validation split and source integrity. They do not train a substitute model or reproduce the leaderboard score. See [provenance](docs/provenance.md) for where the files came from. The project was developed with AI assistance.
