# perceptree

A Rust crate for a **coarse-partition / linear-leaf** binary classifier for
imbalanced, mixed numeric/categorical tabular data (RTB-style workloads).

The model splits the problem into the two things each model family is best at:

- a **coarse** gain-built tree finds a small partition of the input space
  (8–32 regions) — the only combinatorial step;
- each region's leaf is **not a constant** but a small **linear model over a
  sparse per-feature basis** `φ(x)`, giving smooth within-region response and
  graceful missing/unseen handling.

Globally it is a *piecewise-additive* function. Inference routes a row to a leaf
(a handful of comparisons) and does one sparse dot product, optimized for the
**single-row CPU path** (no batching). See [`SPEC.md`](SPEC.md) for the design.

## Status

Implements Phase 0–2 of the spec, with zero external dependencies:

- **Phase 0** — columnar loader + schema, temporal split, quantile binning,
  bounded-cat vocabs, leakage-safe **out-of-fold target encoders** for
  high-cardinality features, and the full eval suite (PR-AUC, lift, log-loss,
  ROC-AUC, reliability, single-row latency).
- **Phase 1** — coarse tree with constant-leaf gain, sparse basis `φ(x)`,
  per-leaf linear heads fit by **Newton/IRLS** with L2 (hand-rolled Cholesky),
  and a single-row scalar scorer.
- **Phase 2** — the **leaf-aware split criterion** (`bᵀA⁻¹b` over the reduced
  `[1, z̃]` basis), swappable with the constant-leaf gain via [`GainKind`].

Phase 3 (learned encodings, path-head folding, masking) and the LightGBM Python
harness are out of scope for this revision; calibration is left as a documented
hook.

## Module map

| module       | responsibility                                                        |
|--------------|-----------------------------------------------------------------------|
| `data`       | SoA columns, schema, temporal split, deterministic synthetic generator |
| `encoding`   | quantile knots, vocabs, OOF target encoders, sparse basis `φ(x)`      |
| `tree`       | histogram split-finding, pluggable gain, best-first growth            |
| `leaf_model` | per-leaf logistic fit by Newton/IRLS + L2                             |
| `infer`      | trained `Model`, raw + encoded single-row scorers                     |
| `train`      | end-to-end fit pipeline                                               |
| `eval`       | PR-AUC, lift, log-loss, ROC-AUC, reliability, latency                 |
| `linalg`     | SPD Cholesky solve                                                     |

## Quick start

```rust
use perceptree::{Config, GainKind, LeafKind};
use perceptree::data::make_synthetic;
use perceptree::train::train;
use perceptree::eval;

let ds = make_synthetic(120_000, 2024);
let (tr, va) = ds.temporal_split(0.75);

let mut cfg = Config::default();          // K=16, λ=1, linear leaves
cfg.gain = GainKind::LeafAware;           // Phase-2 split criterion
cfg.leaf_model = LeafKind::Linear;

let model = train(&tr, &cfg);
let preds = model.predict_dataset(&va);
println!("{:?}", eval::evaluate(&preds, &va.labels));

// Single-row hot path:
let row = perceptree::infer::Model::raw_row(&va, 0);
let p = model.predict_proba(&row);
```

## Run the demo and tests

```sh
cargo run --release --example end_to_end   # metrics + ablations + latency
cargo test                                 # unit + integration tests
```

The example reproduces the spec's two ablations: **linear vs constant leaf**
(isolating the linear-leaf contribution) and **leaf-aware vs constant gain**
across a sweep of `K` (justifying the Phase-2 criterion via leaf count at
matched metric).
