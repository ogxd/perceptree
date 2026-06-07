//! End-to-end regression tests on a held-out temporal split, plus the
//! spec-mandated robustness checks (unseen high-card routing, leaf-aware gain
//! yielding no-worse-at-fewer-leaves behaviour, scorer-path equivalence).

use perceptree::data::{make_synthetic, Column, MISSING_CAT};
use perceptree::eval;
use perceptree::infer::{Model, RawValue};
use perceptree::train::train;
use perceptree::{Config, GainKind, LeafKind};

fn split() -> (perceptree::data::Dataset, perceptree::data::Dataset) {
    make_synthetic(60_000, 99).temporal_split(0.75)
}

#[test]
fn linear_leaf_at_least_matches_constant_leaf() {
    let (tr, va) = split();
    let mut cfg = Config::default();
    cfg.num_leaves = 16;

    cfg.leaf_model = LeafKind::Constant;
    let m_const = train(&tr, &cfg);
    let p_const = m_const.predict_dataset(&va);
    let pr_const = eval::pr_auc(&p_const, &va.labels);

    cfg.leaf_model = LeafKind::Linear;
    let m_lin = train(&tr, &cfg);
    let p_lin = m_lin.predict_dataset(&va);
    let pr_lin = eval::pr_auc(&p_lin, &va.labels);

    // Linear leaves should not be worse than constant leaves at equal K
    // (the within-region slope is the whole point).
    assert!(
        pr_lin >= pr_const - 0.01,
        "linear PR-AUC {pr_lin} fell below constant {pr_const}"
    );
}

#[test]
fn leaf_aware_competitive_at_fewer_leaves() {
    let (tr, va) = split();

    let mut c = Config::default();
    c.leaf_model = LeafKind::Linear;

    // Constant gain at K=16.
    c.gain = GainKind::Constant;
    c.num_leaves = 16;
    let big = train(&tr, &c);
    let pr_big = eval::pr_auc(&big.predict_dataset(&va), &va.labels);

    // Leaf-aware gain at K=8 should remain competitive (within a margin),
    // demonstrating a coarser, better-matched partition.
    c.gain = GainKind::LeafAware;
    c.num_leaves = 8;
    let small = train(&tr, &c);
    let pr_small = eval::pr_auc(&small.predict_dataset(&va), &va.labels);

    assert!(
        pr_small > pr_big - 0.05,
        "leaf-aware K=8 PR-AUC {pr_small} not competitive with constant K=16 {pr_big}"
    );
}

#[test]
fn unseen_highcard_routes_via_prior_and_default() {
    let (tr, _va) = split();
    let cfg = Config::default();
    let model = train(&tr, &cfg);

    // Build a raw row whose high-card features carry never-before-seen ids and
    // whose numerics are missing — must score finitely via prior + default dir.
    let mut row: Vec<RawValue> = Vec::new();
    for c in &tr.columns {
        match c {
            Column::Numeric(_) => row.push(RawValue::Num(f32::NAN)),
            Column::Categorical(_) => row.push(RawValue::Cat(i64::MAX - 1)), // unseen
        }
    }
    // Override one categorical to explicitly missing for good measure.
    row[10] = RawValue::Cat(MISSING_CAT);

    let p = model.predict_proba(&row);
    assert!(p.is_finite() && p > 0.0 && p < 1.0, "degraded score not finite: {p}");
}

#[test]
fn scorer_paths_match_within_tolerance() {
    let (tr, va) = split();
    let cfg = Config::default();
    let model = train(&tr, &cfg);
    let m = model.encoders.transform(&va);
    for r in 0..va.n_rows().min(1000) {
        let a = model.predict_proba_encoded(&m, r);
        let b = model.predict_proba(&Model::raw_row(&va, r));
        assert!((a - b).abs() < 1e-5, "row {r}: {a} vs {b}");
    }
}

#[test]
fn end_to_end_metric_is_strong() {
    let (tr, va) = split();
    let mut cfg = Config::default();
    cfg.num_leaves = 16;
    let model = train(&tr, &cfg);
    let preds = model.predict_dataset(&va);
    let metrics = eval::evaluate(&preds, &va.labels);
    assert!(metrics.roc_auc > 0.70, "roc {}", metrics.roc_auc);
    assert!(metrics.lift_top10pct > 2.0, "lift {}", metrics.lift_top10pct);
}
