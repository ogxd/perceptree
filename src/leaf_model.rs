//! Per-leaf linear head, fit by Newton/IRLS on the class-weighted log-loss
//! with L2.
//!
//! For a leaf the model is `s = s_0 + bias + Σ_j φ_j(x) · w_j`, where `φ` is the
//! leaf's active slice of the sparse basis. The fit solves, each Newton step,
//! `A δ = b` with `A = Σ_i h_i x_i x_iᵀ + λI` and `b = Σ_i g_i x_i + λ w`
//! (intercept unregularized), via Cholesky. Weights are initialized so the leaf
//! starts at its base-rate log-odds — it only learns the within-region slope.

use crate::encoding::{expand_row, EncodedMatrix, Encoders};
use crate::linalg::{cholesky_solve, SymMatrix};
use crate::{Config, LeafKind};
use std::collections::HashMap;

/// A fitted leaf: sparse weights over active basis slots plus a bias, on top of
/// the global base score `s_0`.
#[derive(Clone, Debug)]
pub struct Leaf {
    pub idx: Vec<u32>,
    pub w: Vec<f32>,
    pub bias: f32,
}

impl Leaf {
    /// Score contribution of this leaf for a sparse basis row (excludes `s_0`).
    /// `phi` need not be sorted; indices not in `idx` contribute zero.
    pub fn dot(&self, phi: &[(u32, f32)]) -> f32 {
        let mut acc = self.bias;
        // Linear scan of the (small) weight vector; idx is leaf-local.
        for &(gi, val) in phi {
            // Binary search since idx is built sorted.
            if let Ok(pos) = self.idx.binary_search(&gi) {
                acc += self.w[pos] * val;
            }
        }
        acc
    }
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Fit one leaf over `rows` (indices into `matrix`). `s0` is the global base
/// score. Returns a [`Leaf`] with `idx` sorted ascending.
pub fn fit_leaf(
    rows: &[u32],
    encoders: &Encoders,
    matrix: &EncodedMatrix,
    labels: &[u8],
    weights: &[f32],
    s0: f32,
    cfg: &Config,
) -> Leaf {
    // Leaf base-rate log-odds → initial bias (relative to s0).
    let mut wy = 0.0f64;
    let mut wsum = 0.0f64;
    for &r in rows {
        let r = r as usize;
        wy += weights[r] as f64 * labels[r] as f64;
        wsum += weights[r] as f64;
    }
    let rate = if wsum > 0.0 { (wy / wsum).clamp(1e-6, 1.0 - 1e-6) } else { 0.5 };
    let init_bias = (rate / (1.0 - rate)).ln() - s0 as f64;

    if cfg.leaf_model == LeafKind::Constant {
        return Leaf { idx: Vec::new(), w: Vec::new(), bias: init_bias as f32 };
    }

    // Collect the active basis indices over the leaf, assign local slots
    // (slot 0 is the intercept).
    let mut global_to_local: HashMap<u32, usize> = HashMap::new();
    let mut active: Vec<u32> = Vec::new();
    let mut phi = Vec::new();
    // Precompute each row's sparse basis in local coordinates.
    let mut rows_local: Vec<Vec<(usize, f32)>> = Vec::with_capacity(rows.len());
    for &r in rows {
        expand_row(encoders, matrix, r as usize, &mut phi);
        let mut entries = Vec::with_capacity(phi.len());
        for &(gidx, val) in &phi {
            let local = *global_to_local.entry(gidx).or_insert_with(|| {
                active.push(gidx);
                active.len() // local slot = position+1 (0 reserved for intercept)
            });
            entries.push((local, val));
        }
        rows_local.push(entries);
    }

    let d = active.len() + 1; // + intercept
    let lam = cfg.lambda as f64;

    let mut w = vec![0.0f64; d];
    w[0] = init_bias;

    for _ in 0..cfg.irls_iters {
        let mut a = SymMatrix::zeros(d);
        let mut b = vec![0.0f64; d];

        for (k, &r) in rows.iter().enumerate() {
            let r = r as usize;
            let entries = &rows_local[k];
            // s = s0 + bias + Σ w·φ
            let mut s = s0 as f64 + w[0];
            for &(local, val) in entries {
                s += w[local] * val as f64;
            }
            let p = sigmoid(s);
            let c = weights[r] as f64;
            let gi = c * (p - labels[r] as f64);
            let hi = c * p * (1.0 - p);

            // x = [1, φ...]; accumulate A += hi x xᵀ, b += gi x.
            // Intercept row/col.
            a.add(0, 0, hi);
            b[0] += gi;
            for &(li, vi) in entries {
                let xi = vi as f64;
                a.add(li, 0, hi * xi);
                a.add(0, li, hi * xi);
                b[li] += gi * xi;
                for &(lj, vj) in entries {
                    a.add(li, lj, hi * xi * vj as f64);
                }
            }
        }

        // L2 on every weight except the intercept.
        for i in 1..d {
            a.add(i, i, lam);
            b[i] += lam * w[i];
        }

        match cholesky_solve(a, &b) {
            Some(delta) => {
                for i in 0..d {
                    w[i] -= delta[i];
                }
            }
            None => break, // not PD (shouldn't happen with λ>0); keep current w
        }
    }

    // Emit with indices sorted so the scorer can binary-search.
    let mut pairs: Vec<(u32, f32)> =
        active.iter().enumerate().map(|(i, &g)| (g, w[i + 1] as f32)).collect();
    pairs.sort_by_key(|&(g, _)| g);
    let idx = pairs.iter().map(|&(g, _)| g).collect();
    let wv = pairs.iter().map(|&(_, v)| v).collect();
    Leaf { idx, w: wv, bias: w[0] as f32 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Column, Dataset, FeatureKind, Rng, Schema};

    /// On a single leaf with one numeric feature carrying a clean linear
    /// log-odds slope, IRLS should recover a positive slope and converge.
    #[test]
    fn irls_recovers_linear_signal() {
        let n = 4000;
        let mut rng = Rng::new(5);
        let mut col = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..n {
            let x = rng.next_f32() * 2.0 - 1.0; // [-1,1]
            col.push(x);
            let s = 2.0 * x; // true slope
            let p = 1.0 / (1.0 + (-s).exp());
            labels.push((rng.next_f32() < p) as u8);
        }
        let schema = Schema::new(vec!["x".into()], vec![FeatureKind::Numeric]);
        let ds = Dataset::new(schema, vec![Column::Numeric(col)], labels.clone());
        let mut cfg = Config::default();
        cfg.lambda = 0.01;
        cfg.irls_iters = 25;
        let (enc, m) = crate::encoding::Encoders::fit(&ds, &cfg);

        let rate = labels.iter().map(|&y| y as f64).sum::<f64>() / n as f64;
        let s0 = (rate / (1.0 - rate)).ln() as f32;
        let rows: Vec<u32> = (0..n as u32).collect();
        let leaf = fit_leaf(&rows, &enc, &m, &labels, &ds.weights, s0, &cfg);

        // Score at x=+1 should exceed score at x=-1 (monotone increasing).
        let mut phi_hi = Vec::new();
        crate::encoding::push_numeric(
            match &enc.features[0] {
                crate::encoding::FeatureEncoder::Numeric(e) => &e.knots,
                _ => unreachable!(),
            },
            0,
            0.95,
            &mut phi_hi,
        );
        let mut phi_lo = Vec::new();
        crate::encoding::push_numeric(
            match &enc.features[0] {
                crate::encoding::FeatureEncoder::Numeric(e) => &e.knots,
                _ => unreachable!(),
            },
            0,
            -0.95,
            &mut phi_lo,
        );
        let s_hi = s0 + leaf.dot(&phi_hi);
        let s_lo = s0 + leaf.dot(&phi_lo);
        assert!(s_hi > s_lo + 0.5, "expected increasing: s_lo={s_lo} s_hi={s_hi}");
    }

    #[test]
    fn constant_leaf_predicts_base_rate() {
        let n = 1000;
        let labels: Vec<u8> = (0..n).map(|i| (i % 4 == 0) as u8).collect(); // 25% rate
        let col = vec![0.0f32; n];
        let schema = Schema::new(vec!["x".into()], vec![FeatureKind::Numeric]);
        let ds = Dataset::new(schema, vec![Column::Numeric(col)], labels.clone());
        let mut cfg = Config::default();
        cfg.leaf_model = LeafKind::Constant;
        let (enc, m) = crate::encoding::Encoders::fit(&ds, &cfg);
        let s0 = 0.0f32; // pretend global base is 0
        let rows: Vec<u32> = (0..n as u32).collect();
        let leaf = fit_leaf(&rows, &enc, &m, &labels, &ds.weights, s0, &cfg);
        let p = 1.0 / (1.0 + (-(s0 + leaf.bias) as f64).exp());
        assert!((p - 0.25).abs() < 0.02, "p={p}");
    }
}
