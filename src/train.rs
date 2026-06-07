//! Training orchestration: class weights → encoders → base score → gradient
//! statistics at `s_0` → coarse tree → per-leaf linear fits → [`Model`].

use crate::data::Dataset;
use crate::encoding::Encoders;
use crate::infer::Model;
use crate::leaf_model::fit_leaf;
use crate::tree::grow;
use crate::Config;

/// Fit a full model on `train`.
pub fn train(train: &Dataset, cfg: &Config) -> Model {
    // Apply class weights for imbalance (positives up-weighted by pos_weight).
    let mut ds = train.clone();
    ds.set_class_weights(cfg.pos_weight);

    // Encoders + the leakage-safe (out-of-fold) training matrix.
    let (encoders, matrix) = Encoders::fit(&ds, cfg);

    // Base score s_0 = log-odds of the weighted positive rate, so Σ g_i ≈ 0.
    let mut wy = 0.0f64;
    let mut wsum = 0.0f64;
    for (&y, &w) in ds.labels.iter().zip(&ds.weights) {
        wy += w as f64 * y as f64;
        wsum += w as f64;
    }
    let rate = (wy / wsum).clamp(1e-6, 1.0 - 1e-6);
    let s0 = (rate / (1.0 - rate)).ln() as f32;
    let p0 = rate as f32;

    // Single-tree model: all gradient/hessian stats computed once at s_0.
    let n = ds.n_rows();
    let mut g = vec![0.0f32; n];
    let mut h = vec![0.0f32; n];
    for i in 0..n {
        let c = ds.weights[i];
        g[i] = c * (p0 - ds.labels[i] as f32);
        h[i] = c * p0 * (1.0 - p0);
    }

    // Coarse partition.
    let tree = grow(&encoders, &matrix, &g, &h, cfg);

    // Per-leaf linear heads (fit on the same out-of-fold training matrix).
    let leaves = tree
        .leaf_rows
        .iter()
        .map(|rows| fit_leaf(rows, &encoders, &matrix, &ds.labels, &ds.weights, s0, cfg))
        .collect();

    Model { encoders, nodes: tree.nodes, root: tree.root, leaves, s0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::make_synthetic;
    use crate::eval;
    use crate::infer::Model;

    #[test]
    fn trains_and_beats_constant_baseline() {
        let ds = make_synthetic(40_000, 7);
        let (tr, va) = ds.temporal_split(0.75);
        let mut cfg = Config::default();
        cfg.num_leaves = 16;
        let model = train(&tr, &cfg);

        let preds = model.predict_dataset(&va);
        let pr_auc = eval::pr_auc(&preds, &va.labels);
        // A trivial constant predictor has PR-AUC ≈ positive rate.
        let base_rate =
            va.labels.iter().map(|&y| y as f64).sum::<f64>() / va.n_rows() as f64;
        assert!(
            pr_auc > base_rate * 1.5,
            "PR-AUC {pr_auc} not clearly above base rate {base_rate}"
        );
        // ROC-AUC should be well above chance on this learnable target.
        let roc = eval::roc_auc(&preds, &va.labels);
        assert!(roc > 0.7, "ROC-AUC too low: {roc}");
    }

    #[test]
    fn raw_and_encoded_paths_agree() {
        let ds = make_synthetic(8_000, 3);
        let (tr, va) = ds.temporal_split(0.8);
        let cfg = Config::default();
        let model = train(&tr, &cfg);

        let m = model.encoders.transform(&va);
        for r in 0..va.n_rows().min(500) {
            let enc_p = model.predict_proba_encoded(&m, r);
            let raw = Model::raw_row(&va, r);
            let raw_p = model.predict_proba(&raw);
            assert!(
                (enc_p - raw_p).abs() < 1e-5,
                "row {r}: encoded {enc_p} vs raw {raw_p}"
            );
        }
    }
}
