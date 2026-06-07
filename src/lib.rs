//! # perceptree
//!
//! A binary classifier for imbalanced, mixed numeric/categorical tabular data.
//!
//! The model is a **coarse partition with linear leaves**: a small gain-built
//! tree (8–32 regions) splits the heterogeneous input space, and each leaf is a
//! small linear model over a sparse per-feature basis `φ(x)` — a
//! piecewise-additive function per region. Inference routes a row to a leaf (a
//! handful of comparisons) then does one sparse dot product.
//!
//! See `SPEC.md` for the full design. Module map:
//! - [`data`]      — columnar loading, schema, temporal split.
//! - [`encoding`]  — quantile knots, vocabs, out-of-fold target encoders,
//!                   sparse basis expansion `φ(x)`.
//! - [`tree`]      — histogram split-finding with a pluggable gain criterion
//!                   (constant-leaf / leaf-aware) and best-first growth.
//! - [`leaf_model`]— per-leaf linear head fit by Newton/IRLS with L2.
//! - [`infer`]     — the trained [`infer::Model`] and single-row scorer.
//! - [`train`]     — orchestration of the full fit pipeline.
//! - [`eval`]      — PR-AUC, lift, log-loss, reliability, latency.

pub mod data;
pub mod encoding;
pub mod eval;
pub mod infer;
pub mod leaf_model;
pub mod linalg;
pub mod train;
pub mod tree;

/// Which split-scoring criterion the tree uses (Phase 1 vs Phase 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GainKind {
    /// Standard constant-leaf gain `G²/(H+λ)`.
    Constant,
    /// Leaf-aware gain: scores a split assuming the leaf will fit a line in the
    /// split feature, so a pure linear trend earns nothing.
    LeafAware,
}

/// What sits in each leaf.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafKind {
    /// A single constant log-odds correction (ablation baseline).
    Constant,
    /// A linear model over the leaf's active sparse basis (the real model).
    Linear,
}

/// All tunable knobs. Defaults match the spec's recommended starting point.
#[derive(Clone, Debug)]
pub struct Config {
    /// `K`: stop growing at this many leaves.
    pub num_leaves: usize,
    /// L2 regularization `λ`, shared by split scores and leaf fits.
    pub lambda: f32,
    /// Quantile knots per numeric feature.
    pub knots: usize,
    /// Quantile knots per high-card encoded feature.
    pub hc_knots: usize,
    /// Target-encoding smoothing `α` (shrink toward the prior).
    pub alpha: f32,
    /// Number of out-of-fold folds for leakage-safe target encoding.
    pub n_folds: usize,
    /// Class weight applied to positive rows (imbalance handling).
    pub pos_weight: f32,
    /// Split-scoring criterion.
    pub gain: GainKind,
    /// Leaf model family.
    pub leaf_model: LeafKind,
    /// Newton/IRLS iterations for each leaf fit.
    pub irls_iters: usize,
    /// Minimum rows required on each side of a split.
    pub min_leaf_samples: usize,
    /// Whether high-card features also emit a `log(1+count)` basis block.
    pub hc_use_count: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            num_leaves: 16,
            lambda: 1.0,
            knots: 16,
            hc_knots: 16,
            alpha: 20.0,
            n_folds: 5,
            pos_weight: 1.0,
            gain: GainKind::Constant,
            leaf_model: LeafKind::Linear,
            irls_iters: 8,
            min_leaf_samples: 50,
            hc_use_count: true,
        }
    }
}
