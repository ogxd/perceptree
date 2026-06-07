//! Evaluation: PR-AUC (average precision), lift, log-loss, ROC-AUC, a
//! reliability curve, and single-row latency. RTB use needs calibrated
//! probabilities, so log-loss and reliability sit alongside the ranking metrics.

use crate::data::Dataset;
use crate::infer::Model;
use std::time::Instant;

#[inline]
fn clamp_p(p: f32) -> f64 {
    (p as f64).clamp(1e-7, 1.0 - 1e-7)
}

/// Mean class-weighted-agnostic binary log-loss.
pub fn log_loss(preds: &[f32], labels: &[u8]) -> f64 {
    assert_eq!(preds.len(), labels.len());
    let mut s = 0.0;
    for (&p, &y) in preds.iter().zip(labels) {
        let p = clamp_p(p);
        s += if y == 1 { -p.ln() } else { -(1.0 - p).ln() };
    }
    s / preds.len() as f64
}

/// Sort row indices by descending prediction (stable, deterministic ties).
fn order_desc(preds: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..preds.len()).collect();
    idx.sort_by(|&a, &b| {
        preds[b]
            .partial_cmp(&preds[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx
}

/// ROC-AUC via the Mann–Whitney U statistic with tie-averaged ranks.
pub fn roc_auc(preds: &[f32], labels: &[u8]) -> f64 {
    let n = preds.len();
    let n_pos: usize = labels.iter().map(|&y| y as usize).sum();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }
    // Ascending order for rank assignment.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| preds[a].partial_cmp(&preds[b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut rank_sum_pos = 0.0f64;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && preds[idx[j + 1]] == preds[idx[i]] {
            j += 1;
        }
        // Tie group [i, j]; average rank (1-based).
        let avg_rank = ((i + 1) + (j + 1)) as f64 / 2.0;
        for &row in &idx[i..=j] {
            if labels[row] == 1 {
                rank_sum_pos += avg_rank;
            }
        }
        i = j + 1;
    }
    let u = rank_sum_pos - (n_pos * (n_pos + 1)) as f64 / 2.0;
    u / (n_pos as f64 * n_neg as f64)
}

/// Average precision (area under the precision–recall curve), the PR-AUC.
pub fn pr_auc(preds: &[f32], labels: &[u8]) -> f64 {
    let n_pos: usize = labels.iter().map(|&y| y as usize).sum();
    if n_pos == 0 {
        return 0.0;
    }
    let order = order_desc(preds);
    let mut tp = 0.0f64;
    let mut fp = 0.0f64;
    let mut prev_recall = 0.0f64;
    let mut ap = 0.0f64;
    let mut i = 0;
    let n = order.len();
    while i < n {
        // Advance over all rows tied at this score before reading precision.
        let score = preds[order[i]];
        while i < n && preds[order[i]] == score {
            if labels[order[i]] == 1 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            i += 1;
        }
        let recall = tp / n_pos as f64;
        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 1.0 };
        ap += (recall - prev_recall) * precision;
        prev_recall = recall;
    }
    ap
}

/// Lift at the top `frac` of scored rows: positive rate in the top slice over
/// the overall positive rate. `frac` in (0,1].
pub fn lift_at(preds: &[f32], labels: &[u8], frac: f64) -> f64 {
    let n = preds.len();
    let k = ((n as f64 * frac).round() as usize).clamp(1, n);
    let order = order_desc(preds);
    let top_pos: usize = order[..k].iter().map(|&r| labels[r] as usize).sum();
    let overall: f64 = labels.iter().map(|&y| y as f64).sum::<f64>() / n as f64;
    if overall <= 0.0 {
        return 0.0;
    }
    (top_pos as f64 / k as f64) / overall
}

/// A reliability-curve bin: predicted vs observed positive rate.
#[derive(Clone, Copy, Debug)]
pub struct ReliabilityBin {
    pub mean_pred: f64,
    pub mean_label: f64,
    pub count: usize,
}

/// Equal-width reliability bins over `[0,1]`.
pub fn reliability(preds: &[f32], labels: &[u8], bins: usize) -> Vec<ReliabilityBin> {
    let bins = bins.max(1);
    let mut sp = vec![0.0f64; bins];
    let mut sy = vec![0.0f64; bins];
    let mut cnt = vec![0usize; bins];
    for (&p, &y) in preds.iter().zip(labels) {
        let b = ((p as f64 * bins as f64) as usize).min(bins - 1);
        sp[b] += p as f64;
        sy[b] += y as f64;
        cnt[b] += 1;
    }
    (0..bins)
        .map(|b| {
            let c = cnt[b].max(1) as f64;
            ReliabilityBin {
                mean_pred: sp[b] / c,
                mean_label: sy[b] / c,
                count: cnt[b],
            }
        })
        .collect()
}

/// A bundle of the headline metrics on one split.
#[derive(Clone, Debug)]
pub struct Metrics {
    pub pr_auc: f64,
    pub roc_auc: f64,
    pub log_loss: f64,
    pub lift_top1pct: f64,
    pub lift_top10pct: f64,
}

pub fn evaluate(preds: &[f32], labels: &[u8]) -> Metrics {
    Metrics {
        pr_auc: pr_auc(preds, labels),
        roc_auc: roc_auc(preds, labels),
        log_loss: log_loss(preds, labels),
        lift_top1pct: lift_at(preds, labels, 0.01),
        lift_top10pct: lift_at(preds, labels, 0.10),
    }
}

/// Single-row inference latency over the raw path.
#[derive(Clone, Copy, Debug)]
pub struct Latency {
    pub p50_ns: u128,
    pub p99_ns: u128,
    pub mean_ns: u128,
}

/// Measure single-row latency by scoring `ds` rows through the raw path. A
/// warmup pass primes caches; the accumulator defeats dead-code elimination.
pub fn single_row_latency(model: &Model, ds: &Dataset, max_rows: usize) -> Latency {
    let n = ds.n_rows().min(max_rows);
    let rows: Vec<Vec<crate::infer::RawValue>> =
        (0..n).map(|r| Model::raw_row(ds, r)).collect();

    let mut sink = 0.0f32;
    for row in &rows {
        sink += model.predict_proba(row);
    }

    let mut times = Vec::with_capacity(n);
    for row in &rows {
        let t = Instant::now();
        sink += model.predict_proba(row);
        times.push(t.elapsed().as_nanos());
    }
    std::hint::black_box(sink);

    times.sort_unstable();
    let pct = |q: f64| times[((times.len() as f64 * q) as usize).min(times.len() - 1)];
    let mean = times.iter().sum::<u128>() / times.len() as u128;
    Latency { p50_ns: pct(0.50), p99_ns: pct(0.99), mean_ns: mean }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_ranking_scores_one() {
        let preds = [0.1, 0.4, 0.35, 0.8];
        let labels = [0u8, 0, 1, 1];
        // pred 0.8->y1, 0.4->y0, 0.35->y1, 0.1->y0
        assert!(roc_auc(&preds, &labels) > 0.7);
        let preds2 = [0.9f32, 0.8, 0.2, 0.1];
        let labels2 = [1u8, 1, 0, 0];
        assert!((roc_auc(&preds2, &labels2) - 1.0).abs() < 1e-9);
        assert!((pr_auc(&preds2, &labels2) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pr_auc_floor_is_base_rate() {
        // Constant predictions → AP equals the positive rate.
        let preds = vec![0.5f32; 1000];
        let labels: Vec<u8> = (0..1000).map(|i| (i % 5 == 0) as u8).collect(); // 20%
        let ap = pr_auc(&preds, &labels);
        assert!((ap - 0.2).abs() < 1e-6, "ap={ap}");
    }

    #[test]
    fn log_loss_zero_for_perfect() {
        let preds = [0.999999f32, 0.000001];
        let labels = [1u8, 0];
        assert!(log_loss(&preds, &labels) < 1e-4);
    }

    #[test]
    fn lift_above_one_when_ranking_helps() {
        let mut preds = Vec::new();
        let mut labels = Vec::new();
        for i in 0..1000 {
            let pos = i < 100; // top 100 are positive
            preds.push(if pos { 0.9 } else { 0.1 });
            labels.push(pos as u8);
        }
        assert!(lift_at(&preds, &labels, 0.1) > 5.0);
    }
}
