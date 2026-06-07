# Implementation Plan — Coarse-Partition / Linear-Leaf Tabular Classifier

## 1. Goal

Build a binary classifier for imbalanced, mixed numeric/categorical tabular data
(millions of rows) that:

- matches or beats a tuned gradient-boosted-tree baseline (LightGBM) on ranking
  metric (PR-AUC / lift) and calibrated log-loss, and
- is **strictly faster at single-row CPU inference** than that baseline, via a
  branchless scorer. **No batching** — optimize the single-row path only.

The core idea: split the problem into the two things each model family is best at.

- A **coarse** gain-built tree finds a small partition of the heterogeneous input
  space (8–32 regions). This is the only combinatorial step. Reuse the proven
  histogram split-finding machinery.
- Each region's leaf is **not a constant** — it is a small **linear model over a
  sparse per-feature basis** (a piecewise-additive function). This is what gives
  smooth within-region response, generalization near split boundaries, and
  graceful missing-value handling.

Globally the model is a *piecewise-additive* function: an additive model per
region, with cross-feature interactions captured by the partition itself.
Inference: route to a leaf (a handful of comparisons) + one sparse dot product.

## 2. Decisions (override if wrong)

- **Language: Rust.** Core builder, leaf fitting, and scorer all in Rust.
  Minimal deps; hand-roll the small linear algebra (≤ d×d Cholesky). The
  LightGBM baseline + metric comparison may live in a small Python harness on the
  same data; do not reimplement GBDT.
- **Phase 1 uses the standard constant-leaf gain** to first validate that
  "coarse partition + linear leaves" is competitive. The novel **leaf-aware gain
  criterion is Phase 2**, kept separable for clean ablation.
- **Leaf models fit by Newton/IRLS** (deterministic), not SGD, in v1. Learned
  shared encodings + gradient training are Phase 3, only if metrics demand it.
- High-cardinality unbounded categoricals are handled by **leakage-safe target
  encoding** into continuous features (see §4), not one-hot or many-vs-many.

## 3. Loss, gradients, imbalance

Binary log-loss on raw score `s` (log-odds), `p = sigmoid(s)`, label `y ∈ {0,1}`,
per-row class weight `c_i` (weight the positive class for imbalance):

- gradient `g_i = c_i (p_i - y_i)`
- hessian  `h_i = c_i p_i (1 - p_i)`

Base score `s_0` = log-odds of the weighted positive rate (so global `Σ g_i ≈ 0`).
For a single-tree model all stats are computed at `s_0`.

Evaluate with **PR-AUC and lift** (not just ROC-AUC), plus log-loss and a
reliability curve. RTB use needs calibrated probabilities — the output must be a
clean probability the caller can recalibrate (leave a hook for an optional
isotonic calibration layer; not implemented here).

## 4. Feature classes and encoding

Benchmark schema: **10 numeric + 10 bounded categorical + 5 high-cardinality
unbounded categorical = 25 features.** Three distinct treatments, all producing
entries of one shared **sparse** basis vector `φ(x)` (record per-feature
offset/length once; most entries are zero per row).

**Numeric (×10).** Quantile knots per feature (config `K`, default 16). Basis =
piecewise-linear interpolation weights over the bracketing knots → 2 nonzeros per
feature. Reserved component for missing. Splits = thresholds on the raw value
with a learned default direction for missing.

**Bounded categorical (×10).** Known finite vocab. Basis = one-hot (1 nonzero per
feature); reserved ids for missing and unseen. Splits = many-vs-many grouping:
sort categories by `G/H`, evaluate contiguous partitions.

**High-cardinality unbounded categorical (×5).** Cannot one-hot or enumerate;
unseen values occur at eval. Convert each to **continuous derived features**:
- `enc(k)` = empirical-Bayes smoothed target signal for category `k`:
  `enc(k) = (Σ_{i∈k} c_i y_i + α·prior) / (Σ_{i∈k} c_i + α)`, shrunk toward the
  global/parent rate by `α` (config). Optionally a second derived feature
  `log(1 + count_k)` (rare-vs-common often matters in RTB).
- **Leakage is the critical footgun.** Compute `enc` **out-of-fold**
  (cross-fitting) or via an expanding/ordered scheme so a row's own label never
  informs its own encoding. Without this, near-unique categories (ids) memorize
  and the split criterion overfits badly. This must be in place from Phase 1.
- Downstream, treat the derived feature exactly like a numeric (knots, basis,
  threshold splits). **Unseen category → prior**, then default direction. Graceful
  by construction.
- Phase 3 alternative: replace target encoding with learned **hashed embeddings**
  (hashing trick for the unbounded vocab + reserved unseen row), trained jointly;
  target encoding remains a strong init/baseline.

Per-row basis sparsity: ≈ `10·2 + 10·1 + 5·1 ≈ 35` active entries. The leaf model
is a gather-accumulate over these, not a dense matvec.

## 5. Module layout (`src/`)

- `data` — columnar / struct-of-arrays loading (parquet + csv), feature schema
  (numeric / bounded-cat / highcard-cat), train/valid temporal split.
- `encoding` — quantile knots; categorical vocabs with missing/unseen ids;
  **out-of-fold target encoders** for high-card features; sparse basis expansion
  `φ(x)` with a fixed layout map.
- `tree` — histogram accumulation, split-finding, growth control (stop at K
  leaves or gain threshold), per-split default direction for missing, categorical
  many-vs-many grouping. Pluggable gain criterion (constant-leaf / leaf-aware).
- `leaf_model` — per-leaf linear head fit by Newton/IRLS with L2; init to leaf
  base-rate log-odds; (Phase 3) path-head folding to a single per-leaf weight.
- `infer` — single-row scorer: high-card encode lookups → route → sparse basis →
  gather-accumulate over the leaf weights → sigmoid.
- `train` — orchestration, class weights, fit pipeline.
- `eval` — PR-AUC, lift, log-loss, reliability; single-row latency (p50/p99);
  comparison vs the LightGBM baseline.

## 6. Data structures

```
NumericFeature  { knots: Vec<f32>, basis_off: u32, basis_len: u32 }
BoundedCat      { vocab: HashMap<Cat,u32>, missing_id, unseen_id, basis_off, basis_len }
HighCardCat     { encoder: HashMap<Cat,f32> /* OOF target enc */, prior: f32,
                  count: HashMap<Cat,u32>, /* then knots like a numeric */ knots: Vec<f32>,
                  basis_off, basis_len }

Node  { feature: u32, kind: Numeric{thr:f32} | Categorical{set:BitSet},
        default_left: bool, left: NodeRef, right: NodeRef }
Leaf  { idx: Vec<u32>, w: Vec<f32>, bias: f32 }   // sparse weights over active basis slots
Tree  { nodes, leaves, basis_layout, encoders }
```

The high-card `encoder` maps are the only per-row lookups in the hot path; keep
them compact (sorted arrays + binary search, or a flat hashmap).

## 7. Phase 0 — scaffolding + baseline (gate: numbers exist)

- Columnar SoA loader; temporal train/valid split.
- Quantile binning; categorical vocabs; out-of-fold target encoders for the 5
  high-card features.
- `eval` metrics implemented and unit-tested on toy data.
- Record the **LightGBM baseline**: metric + single-row p50/p99 latency on the
  same split. Everything else is measured against these.

## 8. Phase 1 — core hypothesis (gate: match baseline metric at lower latency)

1. **Coarse tree, standard gain.** Histogram split-finding, constant-leaf score
   per side `score_const(G,H) = G²/(H+λ)`, gain `= score_L + score_R − score_P`.
   Grow best-first; stop at `K` leaves (default 16). Pick `default_left` per
   split by trying both missing directions and keeping the higher gain.
2. **Sparse basis** `φ(x)` per §4.
3. **Leaf fit.** Per leaf, fit linear `w` over the leaf's active basis by
   Newton/IRLS on class-weighted log-loss with L2 (`A = Σ h φφᵀ + λI`,
   `b = Σ g φ`; solve via Cholesky; few iterations). Init so the leaf predicts its
   base-rate log-odds — it starts at the coarse-tree prediction and only learns
   the within-region slope.
4. **Scorer** (`infer`): scalar reference first — correctness over speed.
5. **Compare** vs Phase 0; sweep `K`; ablate **linear leaf vs constant leaf** to
   isolate the linear-leaf contribution.

**Go/no-go:** does coarse-tree + linear leaves reach the baseline metric? If it
needs many leaves to get there, that is the signal Phase 2 is needed.

## 9. Phase 2 — leaf-aware split criterion (the novel piece)

Score splits knowing the leaf will be linear, so a split that only captures a
linear trend in its own feature earns nothing → a coarser, better-matched tree.

Per candidate split on feature value `z`, reduced basis `φ = [1, z̃]` per side:
- accumulate per bin: `Σh, Σh·z̃, Σh·z̃², Σg, Σg·z̃`
- prefix-sum across sorted bins → left side `A` (2×2) `+ λI` and `b` (2-vec);
  right = parent − left.
- side score `= bᵀ A⁻¹ b`; leaf-aware gain `= score_L + score_R − score_P_linear`.

Keep behind the same trait as Phase 1's gain so they're swappable. The **actual
leaf model is still the full sparse fit of Phase 1** — the reduced basis only
scores splits. Ablate: leaf count chosen by leaf-aware vs constant gain at matched
metric, and latency at that count. This is the experiment that justifies the
approach.

## 10. Phase 3 — expressiveness + robustness (only if metric still gapped)

- **Shared learned encoding.** Replace fixed basis with learned per-feature
  vectors (numeric: interpolated learned knot-vectors; categorical: embeddings;
  high-card: hashed embeddings), trained jointly with leaf heads on the frozen
  partition. Shared across leaves so all data informs the representation.
- **Coarse-to-fine path heads.** Head at each node on the path, summed
  root→leaf; **fold** to one per-leaf weight at inference (sum of ancestor heads,
  since summed linear heads are linear). Train structured, serve flat. Add a
  fold-equivalence test.
- **Missing/unseen robustness.** Train with random feature masking (replace a
  feature's basis with its missing component). Optional global additive backbone
  summed with the tree's region correction for maximal graceful degradation.

Dial to expose: linear leaf (folds, fastest) vs one nonlinearity in the head
(no fold, one small net per row).

## 11. Phase 4 — single-row inference hardening (gate: beat baseline p99)

No batching, so the target is one row, end to end:
1. **High-card encode**: 5 compact map lookups (sorted-array + binary search or
   flat hashmap). These are the only non-arithmetic steps; keep maps small.
2. **Route**: coarse tree, ≤ ~5 deep → a handful of predicated compares, no
   data-dependent branch; default direction on missing.
3. **Sparse basis**: compute the ≈35 active `(index, value)` entries directly.
4. **Score**: gather the leaf's weights at those indices, accumulate, add bias,
   sigmoid. The active leaf's weight vector is small (≈ a couple KB) → L1.
- Optional int8 quantization of leaf weights only if floating-point misses p99.
- Re-measure single-row p50/p99 vs Phase 0.

## 12. Testing

- Gain accumulation vs brute force on small data, both criteria.
- **Target-encoder leakage test**: encodings of a held-out fold are independent of
  held-out labels; near-unique categories do not collapse to their own label.
- Leaf fit: IRLS converges; matches a reference solver on a small leaf.
- Fold equivalence (Phase 3): path-sum heads == folded per-leaf weights, exactly.
- Scorer: optimized/quantized path matches the scalar reference within tolerance;
  unseen high-card category routes via prior + default direction.
- End-to-end metric on a held-out temporal split, regression-tested across phases.

## 13. Config surface

`num_leaves` (K), L2 `λ`, knots per numeric, knots per high-card encoded feature,
target-encoding smoothing `α`, class weight, gain criterion {constant |
leaf-aware}, leaf model {constant | linear | linear+nonlinearity}, masking
probability (Phase 3).