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

## References

- Blocki, J., Christin, N., Datta, A., Procaccia, A. D., & Sinha, A. (2013).
  Audit games. *IJCAI*, 41–47.
- Canillas, M., Hasan, O., & Brunie, L. (2020). Supplier impersonation fraud
  detection using Bayesian inference. *IEEE BigComp*, 1–8.
- Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). PaySim: A financial
  mobile money simulator for fraud detection. *28th EMSS*, 249–255.
