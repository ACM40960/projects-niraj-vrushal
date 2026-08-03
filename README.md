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

## References

- Blocki, J., Christin, N., Datta, A., Procaccia, A. D., & Sinha, A. (2013).
  Audit games. *IJCAI*, 41–47.
- Canillas, M., Hasan, O., & Brunie, L. (2020). Supplier impersonation fraud
  detection using Bayesian inference. *IEEE BigComp*, 1–8.
- Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). PaySim: A financial
  mobile money simulator for fraud detection. *28th EMSS*, 249–255.
