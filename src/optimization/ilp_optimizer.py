"""
ILP-based auditor allocation ("audit games" formulation).

Takes the ranked expected-yield table (previous commit) and solves an
Integer Linear Program deciding which transactions to actually audit,
subject to real-world operational constraints named in the literature
review (section 4, following Blocki et al., 2013's "audit games" framing):

  - a total investigator-time budget (not a raw headcount - different
    transaction types cost different amounts of investigator time to
    review)
  - per-sector (transaction-type) caps, so audits don't over-concentrate
    on a single transaction type
  - a mandatory random-sampling quota: some capacity is always spent on
    transactions the model didn't flag, as a baseline deterrence/coverage
    mechanism independent of the risk score

Solving over the full transaction population (millions of rows) is not
tractable for a MIP solver. Since transactions far outside the top of the
expected-yield ranking will essentially never be selected anyway, the ILP
is solved over a bounded shortlist (top N by expected yield) plus a
separately-drawn random sample for the mandatory-sampling requirement.
"""

import pathlib

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

DATA_PROCESSED_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"
YIELD_TABLE_PATH = DATA_PROCESSED_DIR / "expected_yield.csv"
ALLOCATION_PATH = DATA_PROCESSED_DIR / "auditor_allocation.csv"

REPORTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42

# Relative investigator-time cost by transaction type. TRANSFER/CASH-OUT
# are PaySim's only fraud-eligible types and typically require tracing
# funds to a destination account, so they're weighted as costing more
# investigator time than a simple PAYMENT or CASH-IN review.
DEFAULT_AUDIT_COST_BY_TYPE = {
    "CASH-IN": 0.5,
    "CASH-OUT": 1.0,
    "DEBIT": 0.5,
    "PAYMENT": 0.5,
    "TRANSFER": 1.5,
}


def build_candidate_pool(
    yield_table: pd.DataFrame,
    shortlist_size: int = 10_000,
    random_sample_size: int = 500,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Build a tractable candidate pool for the ILP: the top `shortlist_size`
    transactions by expected yield (already sorted descending - see
    build_yield_table in expected_yield.py), plus a separately-drawn
    random sample of `random_sample_size` transactions from the remaining
    population, marked "mandatory" so the ILP is forced to include them
    regardless of their yield.
    """
    yield_table = yield_table.reset_index(drop=True)
    shortlist = yield_table.head(shortlist_size).copy()
    shortlist["mandatory"] = False

    remainder = yield_table.iloc[shortlist_size:]
    rng = np.random.default_rng(random_state)
    random_sample_size = min(random_sample_size, len(remainder))
    if random_sample_size > 0:
        random_idx = rng.choice(remainder.index, size=random_sample_size, replace=False)
        random_sample = remainder.loc[random_idx].copy()
        random_sample["mandatory"] = True
        candidates = pd.concat([shortlist, random_sample], ignore_index=True)
    else:
        candidates = shortlist

    return candidates


def _infer_transaction_type(candidates: pd.DataFrame, type_columns_prefix: str = "type_") -> pd.Series:
    """Recover the transaction type from the one-hot columns produced by
    preprocessing (type_CASH-IN, type_CASH-OUT, ...)."""
    type_cols = [c for c in candidates.columns if c.startswith(type_columns_prefix)]
    if not type_cols:
        raise ValueError(
            "No one-hot 'type_*' columns found in candidates. Build the "
            "yield table from engineered features (see preprocessing.py)."
        )
    return candidates[type_cols].idxmax(axis=1).str.replace(type_columns_prefix, "", regex=False)


def solve_auditor_allocation(
    candidates: pd.DataFrame,
    capacity: float,
    sector_cap_fraction: float | None = 0.5,
    audit_cost_by_type: dict | None = None,
) -> pd.DataFrame:
    """
    Solve the ILP: which candidate transactions to audit, maximizing total
    expected yield, subject to:
      - sum(audit_cost_i * x_i) <= capacity          (investigator-time budget,
        across every audited transaction, mandatory or not)
      - sum(cost of NON-mandatory audits in sector s) <= sector_cap_fraction
        * capacity, per sector (sectoral boundaries, if sector_cap_fraction
        is given)
      - x_i == 1 for every transaction marked "mandatory" (the
        random-sampling quota - forced regardless of yield)
      - x_i binary

    Mandatory rows are deliberately excluded from the sector-cap
    accounting: the cap is meant to stop the *optimizer* from
    over-concentrating its discretionary picks in one transaction type,
    not to second-guess the mandatory-sampling requirement. Counting
    mandatory rows against the cap would make the problem infeasible
    whenever chance alone puts more mandatory rows in one sector than a
    tight cap allows - a modeling bug, not a real constraint conflict.

    If candidates doesn't already have a "_type"/"_audit_cost" column,
    type is inferred from one-hot 'type_*' columns and cost is looked up
    from audit_cost_by_type (defaults to DEFAULT_AUDIT_COST_BY_TYPE).

    Uses scipy.optimize.milp (HiGHS solver, linked directly into scipy -
    no external solver executable/subprocess involved) rather than an
    external MIP solver binary, which avoids a whole class of platform-
    specific issues (antivirus quarantine, missing runtime DLLs, PATH
    problems) that a bundled solver executable like CBC can run into,
    particularly on Windows.

    Returns `candidates` with an added "audited" 0/1 column.
    """
    audit_cost_by_type = audit_cost_by_type or DEFAULT_AUDIT_COST_BY_TYPE
    candidates = candidates.reset_index(drop=True).copy()

    candidates["_type"] = _infer_transaction_type(candidates)
    candidates["_audit_cost"] = candidates["_type"].map(audit_cost_by_type).fillna(1.0)

    n = len(candidates)
    costs = candidates["_audit_cost"].to_numpy()
    yields_ = candidates["expected_yield_mean"].to_numpy()
    mandatory_mask = (
        candidates["mandatory"].to_numpy(dtype=bool)
        if "mandatory" in candidates.columns
        else np.zeros(n, dtype=bool)
    )

    mandatory_cost = costs[mandatory_mask].sum()
    if mandatory_cost > capacity:
        raise ValueError(
            f"Mandatory random-sample transactions cost {mandatory_cost:.1f} in "
            f"total, which already exceeds the total capacity of {capacity}. "
            "Reduce random_sample_size in build_candidate_pool() or increase capacity."
        )

    # milp() minimizes c @ x by default; negate to maximize total yield.
    c = -yields_

    # Investigator-time budget: every audited transaction counts, mandatory
    # or not - this is a hard overall limit.
    rows = [costs]
    upper_bounds = [capacity]

    # Sectoral boundaries: only the optimizer's own discretionary picks in
    # a sector count against the cap - mandatory rows are exempt (see
    # docstring). Zero out mandatory rows' coefficients in each sector row.
    if sector_cap_fraction is not None:
        for _, group in candidates.groupby("_type"):
            row = np.zeros(n)
            discretionary = group.index[~mandatory_mask[group.index]]
            row[discretionary] = costs[discretionary]
            rows.append(row)
            upper_bounds.append(sector_cap_fraction * capacity)

    A = np.vstack(rows)
    constraints = LinearConstraint(A, lb=-np.inf, ub=np.array(upper_bounds))

    # Mandatory random sampling: force x_i == 1 via a tight lower bound
    # rather than an extra constraint row (equivalent, and cheaper).
    lb = np.zeros(n)
    ub = np.ones(n)
    lb[mandatory_mask] = 1.0
    bounds = Bounds(lb=lb, ub=ub)

    result = milp(
        c,
        constraints=constraints,
        bounds=bounds,
        integrality=np.ones(n),  # every variable must take an integer value (with bounds [0,1], this makes them binary)
    )

    if not result.success:
        raise RuntimeError(f"ILP did not solve to optimality: {result.message}")

    # result.x are the solved (near-)binary values; round for a clean 0/1 column.
    candidates["audited"] = np.round(result.x).astype(int)
    return candidates.drop(columns=["_type", "_audit_cost"])


def naive_topk_baseline(
    candidates: pd.DataFrame, capacity: float, audit_cost_by_type: dict | None = None
) -> pd.DataFrame:
    """
    The 'obvious' alternative to the ILP: ignore sector caps and mandatory
    sampling entirely, greedily take the highest-yield transactions until
    the time budget runs out (by yield alone, not yield-per-cost). Used as
    a comparison point for the ILP solution.

    Note this comparison can go either way: the ILP isn't just "the same
    yield with constraints bolted on" - because it optimizes total yield
    subject to the time-budget constraint directly, it naturally solves a
    knapsack problem and can favor transactions with a better yield-per-
    investigator-hour ratio than pure greedy-by-yield selection finds. In
    practice the ILP can match or exceed naive top-k on raw yield while
    also respecting sector caps and mandatory sampling - it isn't
    necessarily a tradeoff at all.
    """
    audit_cost_by_type = audit_cost_by_type or DEFAULT_AUDIT_COST_BY_TYPE
    candidates = candidates.sort_values("expected_yield_mean", ascending=False).reset_index(drop=True).copy()
    candidates["_type"] = _infer_transaction_type(candidates)
    candidates["_audit_cost"] = candidates["_type"].map(audit_cost_by_type).fillna(1.0)

    audited = np.zeros(len(candidates), dtype=int)
    used_capacity = 0.0
    for i, cost in enumerate(candidates["_audit_cost"]):
        if used_capacity + cost > capacity:
            continue
        audited[i] = 1
        used_capacity += cost

    candidates["audited"] = audited
    return candidates.drop(columns=["_type", "_audit_cost"])


def summarize_allocation(allocation: pd.DataFrame, label: str) -> dict:
    """Summary stats for one allocation (ILP or naive baseline): audits
    used, expected yield captured, and actual fraud cases/$ caught
    (using ground-truth isFraud, available here for evaluation purposes -
    a real deployment wouldn't know this in advance)."""
    audited = allocation[allocation["audited"] == 1]
    return {
        "label": label,
        "n_audited": int(len(audited)),
        "total_expected_yield": float(audited["expected_yield_mean"].sum()),
        "fraud_cases_caught": int(audited["isFraud"].sum()),
        "fraud_amount_caught": float(audited.loc[audited["isFraud"] == 1, "amount"].sum()),
    }


def plot_sector_breakdown(allocation: pd.DataFrame, title: str, save_path: pathlib.Path) -> None:
    """Bar chart of audited transaction counts by sector (transaction
    type) - a quick visual check that sector caps are doing their job."""
    import matplotlib.pyplot as plt

    audited = allocation[allocation["audited"] == 1].copy()
    audited["_type"] = _infer_transaction_type(audited)
    counts = audited["_type"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel("Transactions audited")
    ax.set_title(title)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def sector_cap_sensitivity(
    candidates: pd.DataFrame,
    capacity: float,
    cap_fractions: list,
    audit_cost_by_type: dict | None = None,
) -> pd.DataFrame:
    """
    Solve the ILP across a range of sector_cap_fraction values (including
    None for unconstrained) to quantify the 'price' of the sectoral-
    boundary constraint - how much expected yield and actual fraud $
    captured get traded away as the cap tightens.

    Whether this price is large or small is itself an empirical finding:
    if fraud is concentrated in only a few transaction types (as in real
    PaySim data, where only TRANSFER/CASH-OUT ever carry fraud), a tight
    cap forces capacity into sectors with near-zero yield, and the price
    can be substantial - directly demonstrating the tension the
    literature review names between a theoretically optimal risk score
    and constraints that make deployment "practically unmanageable" if
    poorly matched to the domain.
    """
    rows = []
    for cap in cap_fractions:
        allocation = solve_auditor_allocation(
            candidates, capacity=capacity, sector_cap_fraction=cap, audit_cost_by_type=audit_cost_by_type
        )
        label = f"cap={cap}" if cap is not None else "unconstrained"
        summary = summarize_allocation(allocation, label=label)
        summary["sector_cap_fraction"] = cap
        summary["precision"] = (
            summary["fraud_cases_caught"] / summary["n_audited"] if summary["n_audited"] else 0.0
        )
        rows.append(summary)
    return pd.DataFrame(rows)


def plot_sector_cap_sensitivity(
    sensitivity_df: pd.DataFrame, naive_summary: dict, save_path: pathlib.Path
) -> None:
    """
    Line chart: total expected yield and actual fraud $ caught (both in
    dollar terms, so directly comparable) as the sector cap tightens, with
    the naive unconstrained baseline as a horizontal reference line - the
    'price of the constraint' curve for the write-up.
    """
    import matplotlib.pyplot as plt

    df = sensitivity_df.copy()
    x_labels = [f"{c:.1f}" if pd.notna(c) else "None" for c in df["sector_cap_fraction"]]
    x = range(len(df))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, df["total_expected_yield"], marker="o", label="ILP total expected yield (posterior)")
    ax.plot(x, df["fraud_amount_caught"], marker="s", label="ILP actual fraud $ caught (ground truth)")
    ax.axhline(
        naive_summary["total_expected_yield"], color="gray", linestyle="--", linewidth=1,
        label="Naive top-k expected yield",
    )
    ax.axhline(
        naive_summary["fraud_amount_caught"], color="darkgray", linestyle=":", linewidth=1,
        label="Naive top-k fraud $ caught",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Sector cap fraction (max share of budget per transaction type)")
    ax.set_ylabel("Dollars ($)")
    ax.set_title("Price of the sectoral-boundary constraint")
    ax.legend(fontsize=8)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if not YIELD_TABLE_PATH.exists():
        raise FileNotFoundError(
            "No expected yield table found. Run `python -m src.models.expected_yield` first."
        )

    yield_table = pd.read_csv(YIELD_TABLE_PATH)
    n_fraud = int(yield_table["isFraud"].sum())
    total_fraud_amount = yield_table.loc[yield_table["isFraud"] == 1, "amount"].sum()
    print(f"Loaded {len(yield_table):,} scored transactions "
          f"({n_fraud:,} actual fraud, ${total_fraud_amount:,.2f} total fraud value)")

    candidates = build_candidate_pool(yield_table, shortlist_size=10_000, random_sample_size=500)
    print(f"Candidate pool for ILP: {len(candidates):,} transactions "
          f"({int(candidates['mandatory'].sum())} mandatory random)")

    capacity = 1000.0  # investigator-time budget, comparable to earlier k=1000 in the capture curve
    print(f"\nSolving ILP (capacity={capacity}, sector_cap_fraction=0.5)...")
    ilp_allocation = solve_auditor_allocation(candidates, capacity=capacity, sector_cap_fraction=0.5)

    naive_allocation = naive_topk_baseline(candidates, capacity=capacity)

    ilp_summary = summarize_allocation(ilp_allocation, "ILP (constrained)")
    naive_summary = summarize_allocation(naive_allocation, "Naive top-k (unconstrained)")

    print("\n" + pd.DataFrame([ilp_summary, naive_summary]).to_string(index=False))

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ilp_allocation.to_csv(ALLOCATION_PATH, index=False)
    print(f"\nSaved ILP allocation to {ALLOCATION_PATH}")

    plot_sector_breakdown(
        ilp_allocation,
        title="ILP Allocation — Audits by Transaction Type",
        save_path=FIGURES_DIR / "ilp_sector_breakdown.png",
    )
    print(f"Saved sector breakdown plot to {FIGURES_DIR / 'ilp_sector_breakdown.png'}")

    print("\nRunning sector_cap_fraction sensitivity analysis...")
    cap_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, None]
    sensitivity_df = sector_cap_sensitivity(candidates, capacity=capacity, cap_fractions=cap_fractions)
    print("\n" + sensitivity_df[
        ["label", "n_audited", "total_expected_yield", "fraud_cases_caught", "fraud_amount_caught", "precision"]
    ].to_string(index=False))

    SENSITIVITY_PATH = REPORTS_DIR / "sector_cap_sensitivity.csv"
    sensitivity_df.to_csv(SENSITIVITY_PATH, index=False)
    print(f"\nSaved sensitivity table to {SENSITIVITY_PATH}")

    plot_sector_cap_sensitivity(
        sensitivity_df, naive_summary, save_path=FIGURES_DIR / "sector_cap_sensitivity.png"
    )
    print(f"Saved sensitivity plot to {FIGURES_DIR / 'sector_cap_sensitivity.png'}")
