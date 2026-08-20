# Algorithmic Risk Profiling & Optimization

Vrushal Bagwe (25210728), Niraj Palve (25200843)

## Overview

Digital payment fraud detection faces three structural problems: fraud is rare
(extreme class imbalance), the real-world cost of a false positive vs. a false
negative varies widely, and genuine financial records are confidential. This
project addresses all three by:

1. **Simulating a realistic transaction environment** using the PaySim
   synthetic dataset (Lopez-Rojas et al., 2016), which mirrors the topology of
   real mobile-money networks without breaching privacy constraints.
2. **Probabilistic risk profiling** with a Bayesian model that outputs a full
   posterior distribution over fraud probability per transaction, rather than
   a binary flag (following Canillas et al., 2020), enabling a continuous
   *expected financial yield* score.
3. **Constrained resource optimization** via Integer Linear Programming
   (ILP), which takes those expected yields as an objective function and
   solves for the optimal deployment of a limited number of auditors/
   investigators, subject to real-world constraints such as capacity limits,
   sector boundaries, and mandatory random sampling (following the "audit
   games" framing of Blocki et al., 2013).

The end goal is a single pipeline that goes from raw transaction data to a
prescriptive, capacity-aware auditor assignment list — bridging predictive
data science and operations research rather than treating them as separate
stages.

## Project structure

```
risk-profiling-optimization/
├── data/
│   ├── raw/            # PaySim CSVs (gitignored — see Setup)
│   └── processed/      # cleaned / feature-engineered data
├── notebooks/           # exploratory analysis
├── src/
│   ├── data/            # loading, cleaning, feature engineering
│   ├── models/           # baseline classifiers + Bayesian model
│   ├── optimization/     # ILP formulation and solver
│   └── evaluation/       # metrics, financial-impact scoring
├── tests/                # unit tests
├── reports/figures/       # generated plots
└── requirements.txt
```

## Setup

```bash
git clone <repo-url>
cd risk-profiling-optimization
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
```

The last step installs this project itself in "editable" mode, so `src`
is importable as a package regardless of which directory you run a script
or `pytest` from (fixes `ModuleNotFoundError: No module named 'src'` if
you run a script directly, e.g. `python src/data/preprocessing.py`, rather
than as a module with `python -m src.data.preprocessing`).

### Data

Download the PaySim dataset (e.g. from Kaggle:
`ealaxi/paysim1`) and place the CSV in `data/raw/`. This folder is
gitignored — the raw dataset is never committed.

For pipeline development and testing without the full ~470MB file, generate
a small synthetic sample with the same schema:

```bash
python -m src.data.generate_sample
```

This writes `data/raw/sample_paysim.csv` (~20,000 rows). `src/data/load_data.py`
automatically prefers a real `PS_*.csv` if one is present in `data/raw/`, and
falls back to the sample otherwise.

### Running the EDA notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

See `reports/figures/` for the exported plots (class imbalance, transaction
type distribution, amount distribution, and origin-balance discrepancy).

### Preprocessing / feature engineering

```bash
python -m src.data.preprocessing
```

Runs `src/data/preprocessing.py` end to end: loads the PaySim data (real or
synthetic sample), engineers the balance-discrepancy features identified
during EDA, one-hot encodes transaction type, and writes a stratified
train/test split to `data/processed/train.csv` and `data/processed/test.csv`
(fraud rate is preserved in both partitions despite how rare it is).

### Baseline classifiers

```bash
python -m src.models.baseline
```

Trains a class-weighted Random Forest and a class-weighted Linear SVM on
`data/processed/train.csv` and evaluates both on `data/processed/test.csv`
(run preprocessing first). Reports precision, recall, F1, ROC AUC, and
AUPRC rather than accuracy, since accuracy is meaningless on data this
imbalanced. These serve as the deterministic-classifier comparison point
for the Bayesian risk-profiling model added in the next step.

### Bayesian risk-profiling model

```bash
python -m src.models.bayesian_model
```

Fits a Bayesian logistic regression via ADVI (variational inference —
automatically switches to minibatch ADVI when the training set exceeds
50,000 rows, so it scales to the full ~6.3M-row PaySim dataset without
needing a gradient over all rows every iteration). Rather than a single
fraud/not-fraud label, this produces a full posterior distribution over
each transaction's fraud probability — the posterior mean is a continuous
risk score, and its spread quantifies uncertainty. Results (precision/
recall/F1/ROC AUC/AUPRC at a 0.5 threshold, for comparison against the
baselines) and a calibration/reliability plot are saved to `reports/`.
The fitted model itself is saved to `models/` so later steps (expected
yield derivation, below) can reuse it without re-fitting.

### Expected financial yield

```bash
python -m src.models.expected_yield
```

Loads the fitted Bayesian model (saved to `models/` by the step above -
run `python -m src.models.bayesian_model` first) and, for every test
transaction, combines its posterior fraud probability with its dollar
amount into an **expected yield** score (`P(fraud) x amount`) - the
objective function the ILP optimizer maximizes in the next step. Saves a
ranked yield table to `data/processed/expected_yield.csv` and a capture-
rate curve (what % of total fraud would be caught by auditing the top-k
transactions ranked by expected yield) to `reports/figures/`.

### ILP auditor allocation

```bash
python -m src.optimization.ilp_optimizer
```

Solves an Integer Linear Program deciding which transactions to actually
audit, subject to real-world operational constraints named in the
literature review (an "audit games" formulation following Blocki et al.,
2013): a total investigator-time budget (transaction types are weighted
by relative review cost, not just counted), per-sector caps so audits
don't over-concentrate on one transaction type, and a mandatory random-
sampling quota forced in regardless of yield (mandatory rows are exempt
from the sector cap - it's meant to limit the optimizer's own
discretionary picks, not conflict with the sampling requirement). Uses
`scipy.optimize.milp` (HiGHS solver, linked directly into scipy - no
external solver executable or subprocess involved). Solves over a
bounded shortlist (top-N by
expected yield) plus a separately-drawn random sample for tractability,
since solving over the full multi-million-row population isn't feasible
for a MIP solver. Saves the allocation to
`data/processed/auditor_allocation.csv`, prints a comparison against a
naive unconstrained top-k baseline, and saves a sector-breakdown plot to
`reports/figures/`.

Also runs a **sensitivity analysis** across a range of `sector_cap_fraction`
values (0.1 through 1.0, plus unconstrained), quantifying the "price" of
the sectoral-boundary constraint — how much expected yield and actual
fraud dollars get traded away as the cap tightens. This matters
particularly on real PaySim data, where fraud only ever occurs in
`TRANSFER`/`CASH-OUT` transactions: a cap that forces capacity into other
sectors can cost real yield, since it's spending investigator time where
the model has nothing to find. Saves the table to
`reports/sector_cap_sensitivity.csv` and a plot to `reports/figures/`.

Finally, compares audit **composition** sector-by-sector between the
sensitivity sweep's best-performing cap (by actual fraud $ caught) and
the unconstrained case, to diagnose *why* one outperforms the other -
e.g. whether loosening the cap shifts audits toward a sector where the
model's own top picks turn out to be less often correct, even when the
model's posterior yield estimate says otherwise. This is what surfaces a
genuinely useful finding on real PaySim data: real fraud capture can
peak at a moderate cap (e.g. 0.5) rather than at "unconstrained," meaning
the sector cap acts as a hedge against model overconfidence rather than
a pure cost. Saves the table to
`reports/sector_composition_comparison.csv` and a plot to
`reports/figures/`.

### End-to-end benchmark

```bash
python -m src.evaluation.benchmark
```

Every prior evaluation validated one stage in isolation - this tests the
project's actual central claim directly: that a Bayesian posterior
specifically (not just any probability estimate) is what enables good
capacity-constrained allocation.

- **Score-source comparison**: runs the identical ILP formulation (same
  capacity, sector caps, mandatory sampling) driven by two different
  score sources - the Bayesian posterior mean vs. the Random Forest
  baseline's `predict_proba` - each also compared to its own naive top-k
  baseline, isolating whether the scoring model matters from whether
  constrained optimization matters at all.
- **Capacity sensitivity**: sweeps the auditor-capacity budget (only a
  single value was tested in every prior ILP run) to show how outcomes
  scale as resourcing changes.

Saves `reports/score_source_comparison.csv` and
`reports/capacity_sensitivity.csv`, plus matching plots in
`reports/figures/`.

## References

- Blocki, J., Christin, N., Datta, A., Procaccia, A. D., & Sinha, A. (2013).
  Audit games. *IJCAI*, 41–47.
- Canillas, M., Hasan, O., & Brunie, L. (2020). Supplier impersonation fraud
  detection using Bayesian inference. *IEEE BigComp*, 1–8.
- Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). PaySim: A financial
  mobile money simulator for fraud detection. *28th EMSS*, 249–255.

## Student Details
- Vrushal Bagwe (25210728)
- Niraj Palve (25200483)
