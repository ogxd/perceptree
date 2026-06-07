//! End-to-end demo on synthetic data: train the coarse-partition / linear-leaf
//! model, report ranking + calibration metrics and single-row latency, and run
//! the two ablations the spec calls for:
//!   - linear leaf vs constant leaf (isolates the linear-leaf contribution),
//!   - leaf-aware gain vs constant gain across a sweep of `K`.

use perceptree::data::make_synthetic;
use perceptree::eval;
use perceptree::train::train;
use perceptree::{Config, GainKind, LeafKind};

fn run(label: &str, cfg: &Config, tr: &perceptree::data::Dataset, va: &perceptree::data::Dataset) {
    let model = train(tr, cfg);
    let preds = model.predict_dataset(va);
    let m = eval::evaluate(&preds, &va.labels);
    let lat = eval::single_row_latency(&model, va, 20_000);
    println!(
        "{label:<34} leaves={:>2} | PR-AUC {:.4}  ROC-AUC {:.4}  logloss {:.4}  lift@1% {:>5.2}  lift@10% {:>5.2} | p50 {:>5}ns p99 {:>6}ns",
        model.leaves.len(),
        m.pr_auc,
        m.roc_auc,
        m.log_loss,
        m.lift_top1pct,
        m.lift_top10pct,
        lat.p50_ns,
        lat.p99_ns,
    );
}

fn main() {
    let ds = make_synthetic(120_000, 2024);
    let (tr, va) = ds.temporal_split(0.75);
    let base_rate =
        va.labels.iter().map(|&y| y as f64).sum::<f64>() / va.n_rows() as f64;
    println!(
        "synthetic: {} train / {} valid rows, valid positive rate {:.3}\n",
        tr.n_rows(),
        va.n_rows(),
        base_rate
    );

    println!("== ablation: leaf model (K=16, constant gain) ==");
    let mut cfg = Config::default();
    cfg.num_leaves = 16;
    cfg.leaf_model = LeafKind::Constant;
    run("constant leaf", &cfg, &tr, &va);
    cfg.leaf_model = LeafKind::Linear;
    run("linear leaf", &cfg, &tr, &va);

    println!("\n== sweep K, gain criterion: constant vs leaf-aware (linear leaves) ==");
    for &k in &[4usize, 8, 16, 24, 32] {
        for gain in [GainKind::Constant, GainKind::LeafAware] {
            let mut c = Config::default();
            c.num_leaves = k;
            c.gain = gain;
            c.leaf_model = LeafKind::Linear;
            let tag = match gain {
                GainKind::Constant => "constant-gain",
                GainKind::LeafAware => "leaf-aware-gain",
            };
            run(&format!("K={k:<2} {tag}"), &c, &tr, &va);
        }
    }
}
