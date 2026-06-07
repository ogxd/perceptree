//! Feature encoding and the sparse basis `φ(x)`.
//!
//! Three feature treatments, all producing entries of one shared sparse basis:
//! - **Numeric**: quantile knots + piecewise-linear interpolation (2 nonzeros)
//!   with a reserved missing component.
//! - **Bounded categorical**: one-hot over a finite vocab, with reserved
//!   missing and unseen ids.
//! - **High-cardinality categorical**: leakage-safe **out-of-fold** target
//!   encoding into a continuous value, then treated like a numeric (knots +
//!   piecewise-linear basis), plus an optional `log(1+count)` block.
//!
//! [`Encoders::fit`] returns both the inference-time encoders (full-data maps)
//! and the **training** [`EncodedMatrix`] whose high-card columns use the
//! out-of-fold encoding so a row's own label never informs its own features.

use crate::data::{Column, Dataset, FeatureKind, MISSING_CAT};
use crate::Config;
use std::collections::HashMap;

/// Per-feature encoder; carries its slice of the global basis layout.
#[derive(Clone, Debug)]
pub enum FeatureEncoder {
    Numeric(NumericEnc),
    BoundedCat(BoundedCatEnc),
    HighCard(HighCardEnc),
}

#[derive(Clone, Debug)]
pub struct NumericEnc {
    pub knots: Vec<f32>,
    pub basis_off: u32,
}

#[derive(Clone, Debug)]
pub struct BoundedCatEnc {
    /// Raw category code → dense id in `0..n_vocab`.
    pub vocab: HashMap<i64, u32>,
    pub n_vocab: u32,
    pub basis_off: u32,
}

impl BoundedCatEnc {
    #[inline]
    pub fn missing_id(&self) -> u32 {
        self.n_vocab
    }
    #[inline]
    pub fn unseen_id(&self) -> u32 {
        self.n_vocab + 1
    }
    /// Total number of one-hot slots (vocab + missing + unseen).
    #[inline]
    pub fn n_slots(&self) -> u32 {
        self.n_vocab + 2
    }
    /// Map a raw code to its dense routing/basis id.
    #[inline]
    pub fn code_of(&self, raw: i64) -> u32 {
        if raw == MISSING_CAT {
            self.missing_id()
        } else {
            *self.vocab.get(&raw).unwrap_or(&self.unseen_id())
        }
    }
}

#[derive(Clone, Debug)]
pub struct HighCardEnc {
    /// Full-data smoothed target encoding (inference path).
    pub enc: HashMap<i64, f32>,
    /// Full-data category counts.
    pub count: HashMap<i64, u32>,
    /// Global rate the encoding shrinks toward; also the unseen value.
    pub prior: f32,
    pub enc_knots: Vec<f32>,
    pub enc_off: u32,
    pub cnt_knots: Vec<f32>,
    pub cnt_off: u32,
    pub use_count: bool,
}

impl HighCardEnc {
    /// Encoded value for a raw category (unseen/missing → prior).
    #[inline]
    pub fn enc_of(&self, raw: i64) -> f32 {
        if raw == MISSING_CAT {
            self.prior
        } else {
            *self.enc.get(&raw).unwrap_or(&self.prior)
        }
    }
    #[inline]
    pub fn count_of(&self, raw: i64) -> f32 {
        if raw == MISSING_CAT {
            0.0
        } else {
            (*self.count.get(&raw).unwrap_or(&0) as f32).ln_1p()
        }
    }
}

impl FeatureEncoder {
    /// Number of basis slots this feature owns.
    pub fn basis_len(&self) -> u32 {
        match self {
            FeatureEncoder::Numeric(e) => e.knots.len() as u32 + 1,
            FeatureEncoder::BoundedCat(e) => e.n_slots(),
            FeatureEncoder::HighCard(e) => {
                let a = e.enc_knots.len() as u32 + 1;
                let b = if e.use_count { e.cnt_knots.len() as u32 + 1 } else { 0 };
                a + b
            }
        }
    }
}

/// All feature encoders plus the total basis dimension.
#[derive(Clone, Debug)]
pub struct Encoders {
    pub features: Vec<FeatureEncoder>,
    pub basis_dim: u32,
}

/// One encoded feature column, ready for routing and basis expansion.
/// Identical layout for train (out-of-fold high-card) and eval (full maps).
#[derive(Clone, Debug)]
pub enum EncCol {
    /// Numeric raw values; `f32::NAN` marks missing.
    Num(Vec<f32>),
    /// Bounded categorical dense codes (includes missing/unseen ids).
    Cat(Vec<u32>),
    /// High-card: target-encoded value (finite) and `log(1+count)`.
    HighCard { enc: Vec<f32>, cnt: Vec<f32> },
}

/// Column-major encoded matrix, one [`EncCol`] per schema feature.
#[derive(Clone, Debug)]
pub struct EncodedMatrix {
    pub cols: Vec<EncCol>,
    pub n_rows: usize,
}

/// `K` quantile knots spanning the non-missing values (min..max inclusive),
/// strictly increasing after dedup. Returns at least one knot.
fn quantile_knots(values: &[f32], k: usize) -> Vec<f32> {
    let mut v: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return vec![0.0];
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = k.max(2);
    let n = v.len();
    let mut knots = Vec::with_capacity(k);
    for i in 0..k {
        let q = i as f64 / (k as f64 - 1.0);
        let idx = ((q * (n as f64 - 1.0)).round() as usize).min(n - 1);
        let val = v[idx];
        if knots.last().map_or(true, |&last: &f32| val > last) {
            knots.push(val);
        }
    }
    if knots.is_empty() {
        knots.push(v[0]);
    }
    knots
}

/// Weighted positive rate `Σ c_i y_i / Σ c_i` over the rows.
fn weighted_rate(labels: &[u8], weights: &[f32]) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&y, &w) in labels.iter().zip(weights) {
        num += (w as f64) * (y as f64);
        den += w as f64;
    }
    if den > 0.0 {
        (num / den) as f32
    } else {
        0.0
    }
}

/// Smoothed target encoding for one set of (code, label, weight) rows.
/// `enc(k) = (Σ c·y + α·prior) / (Σ c + α)`.
fn target_enc_map(
    codes: &[i64],
    labels: &[u8],
    weights: &[f32],
    rows: impl Iterator<Item = usize>,
    alpha: f32,
    prior: f32,
) -> (HashMap<i64, f32>, HashMap<i64, u32>) {
    let mut wy: HashMap<i64, f64> = HashMap::new();
    let mut wsum: HashMap<i64, f64> = HashMap::new();
    let mut cnt: HashMap<i64, u32> = HashMap::new();
    for i in rows {
        let c = codes[i];
        if c == MISSING_CAT {
            continue;
        }
        *wy.entry(c).or_insert(0.0) += (weights[i] as f64) * (labels[i] as f64);
        *wsum.entry(c).or_insert(0.0) += weights[i] as f64;
        *cnt.entry(c).or_insert(0) += 1;
    }
    let a = alpha as f64;
    let mut enc = HashMap::with_capacity(wsum.len());
    for (&k, &s) in &wsum {
        let v = (wy[&k] + a * prior as f64) / (s + a);
        enc.insert(k, v as f32);
    }
    (enc, cnt)
}

impl Encoders {
    /// Fit all encoders on the training data and produce the **training**
    /// encoded matrix (high-card columns use out-of-fold encoding).
    pub fn fit(train: &Dataset, cfg: &Config) -> (Encoders, EncodedMatrix) {
        let labels = &train.labels;
        let weights = &train.weights;
        let n = train.n_rows();
        let prior = weighted_rate(labels, weights);

        let mut features = Vec::with_capacity(train.schema.n_features());
        let mut cols: Vec<EncCol> = Vec::with_capacity(train.schema.n_features());
        let mut off: u32 = 0;

        for (fi, kind) in train.schema.kinds.iter().enumerate() {
            match kind {
                FeatureKind::Numeric => {
                    let raw = match &train.columns[fi] {
                        Column::Numeric(v) => v,
                        _ => panic!("schema/column kind mismatch at {fi}"),
                    };
                    let knots = quantile_knots(raw, cfg.knots);
                    let enc = NumericEnc { knots, basis_off: off };
                    off += FeatureEncoder::Numeric(enc.clone()).basis_len();
                    features.push(FeatureEncoder::Numeric(enc));
                    cols.push(EncCol::Num(raw.clone()));
                }
                FeatureKind::BoundedCat => {
                    let raw = match &train.columns[fi] {
                        Column::Categorical(v) => v,
                        _ => panic!("schema/column kind mismatch at {fi}"),
                    };
                    let mut vocab: HashMap<i64, u32> = HashMap::new();
                    for &c in raw {
                        if c != MISSING_CAT && !vocab.contains_key(&c) {
                            let id = vocab.len() as u32;
                            vocab.insert(c, id);
                        }
                    }
                    let enc = BoundedCatEnc { n_vocab: vocab.len() as u32, vocab, basis_off: off };
                    off += enc.n_slots();
                    let codes: Vec<u32> = raw.iter().map(|&c| enc.code_of(c)).collect();
                    features.push(FeatureEncoder::BoundedCat(enc));
                    cols.push(EncCol::Cat(codes));
                }
                FeatureKind::HighCardCat => {
                    let raw = match &train.columns[fi] {
                        Column::Categorical(v) => v,
                        _ => panic!("schema/column kind mismatch at {fi}"),
                    };
                    // Full-data maps for inference.
                    let (enc_full, count_full) = target_enc_map(
                        raw, labels, weights, 0..n, cfg.alpha, prior,
                    );

                    // Out-of-fold encoding for the training matrix (leakage-safe).
                    let nf = cfg.n_folds.max(2);
                    let mut oof_enc = vec![prior; n];
                    let mut oof_cnt = vec![0.0f32; n];
                    for f in 0..nf {
                        let train_rows = (0..n).filter(|i| i % nf != f);
                        let (m, c) = target_enc_map(
                            raw, labels, weights, train_rows, cfg.alpha, prior,
                        );
                        for i in (f..n).step_by(nf) {
                            let code = raw[i];
                            if code != MISSING_CAT {
                                oof_enc[i] = *m.get(&code).unwrap_or(&prior);
                                oof_cnt[i] = (*c.get(&code).unwrap_or(&0) as f32).ln_1p();
                            }
                        }
                    }

                    // Knots from the full-data per-row distributions.
                    let full_enc_vals: Vec<f32> =
                        raw.iter().map(|&c| *enc_full.get(&c).unwrap_or(&prior)).collect();
                    let full_cnt_vals: Vec<f32> = raw
                        .iter()
                        .map(|&c| (*count_full.get(&c).unwrap_or(&0) as f32).ln_1p())
                        .collect();
                    let enc_knots = quantile_knots(&full_enc_vals, cfg.hc_knots);
                    let cnt_knots = quantile_knots(&full_cnt_vals, cfg.hc_knots);

                    let enc_off = off;
                    off += enc_knots.len() as u32 + 1;
                    let cnt_off = off;
                    if cfg.hc_use_count {
                        off += cnt_knots.len() as u32 + 1;
                    }
                    let henc = HighCardEnc {
                        enc: enc_full,
                        count: count_full,
                        prior,
                        enc_knots,
                        enc_off,
                        cnt_knots,
                        cnt_off,
                        use_count: cfg.hc_use_count,
                    };
                    features.push(FeatureEncoder::HighCard(henc));
                    cols.push(EncCol::HighCard { enc: oof_enc, cnt: oof_cnt });
                }
            }
        }

        let encoders = Encoders { features, basis_dim: off };
        let matrix = EncodedMatrix { cols, n_rows: n };
        (encoders, matrix)
    }

    /// Encode an arbitrary dataset for inference/eval using the full-data maps.
    pub fn transform(&self, ds: &Dataset) -> EncodedMatrix {
        let n = ds.n_rows();
        let mut cols = Vec::with_capacity(self.features.len());
        for (fi, fe) in self.features.iter().enumerate() {
            match fe {
                FeatureEncoder::Numeric(_) => {
                    let raw = match &ds.columns[fi] {
                        Column::Numeric(v) => v.clone(),
                        _ => panic!("kind mismatch"),
                    };
                    cols.push(EncCol::Num(raw));
                }
                FeatureEncoder::BoundedCat(e) => {
                    let raw = match &ds.columns[fi] {
                        Column::Categorical(v) => v,
                        _ => panic!("kind mismatch"),
                    };
                    cols.push(EncCol::Cat(raw.iter().map(|&c| e.code_of(c)).collect()));
                }
                FeatureEncoder::HighCard(e) => {
                    let raw = match &ds.columns[fi] {
                        Column::Categorical(v) => v,
                        _ => panic!("kind mismatch"),
                    };
                    let enc = raw.iter().map(|&c| e.enc_of(c)).collect();
                    let cnt = raw.iter().map(|&c| e.count_of(c)).collect();
                    cols.push(EncCol::HighCard { enc, cnt });
                }
            }
        }
        EncodedMatrix { cols, n_rows: n }
    }
}

/// Push the piecewise-linear basis entries for one numeric value.
/// Layout: `off..off+K` are knot weights, `off+K` is the reserved missing slot.
#[inline]
pub fn push_numeric(knots: &[f32], off: u32, v: f32, out: &mut Vec<(u32, f32)>) {
    let k = knots.len() as u32;
    if v.is_nan() {
        out.push((off + k, 1.0));
        return;
    }
    let n = knots.len();
    if n == 1 || v <= knots[0] {
        out.push((off, 1.0));
        return;
    }
    if v >= knots[n - 1] {
        out.push((off + (n as u32 - 1), 1.0));
        return;
    }
    let j = knots.partition_point(|&kk| kk <= v) - 1; // knots[j] <= v < knots[j+1]
    let lo = knots[j];
    let hi = knots[j + 1];
    let t = (v - lo) / (hi - lo);
    out.push((off + j as u32, 1.0 - t));
    out.push((off + (j as u32 + 1), t));
}

/// Expand one row of an [`EncodedMatrix`] into sparse basis entries.
pub fn expand_row(
    encoders: &Encoders,
    matrix: &EncodedMatrix,
    row: usize,
    out: &mut Vec<(u32, f32)>,
) {
    out.clear();
    for (fe, col) in encoders.features.iter().zip(&matrix.cols) {
        match (fe, col) {
            (FeatureEncoder::Numeric(e), EncCol::Num(v)) => {
                push_numeric(&e.knots, e.basis_off, v[row], out);
            }
            (FeatureEncoder::BoundedCat(e), EncCol::Cat(codes)) => {
                out.push((e.basis_off + codes[row], 1.0));
            }
            (FeatureEncoder::HighCard(e), EncCol::HighCard { enc, cnt }) => {
                push_numeric(&e.enc_knots, e.enc_off, enc[row], out);
                if e.use_count {
                    push_numeric(&e.cnt_knots, e.cnt_off, cnt[row], out);
                }
            }
            _ => panic!("encoder/column kind mismatch"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::make_synthetic;

    #[test]
    fn knots_strictly_increasing() {
        let v: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let k = quantile_knots(&v, 16);
        for w in k.windows(2) {
            assert!(w[1] > w[0], "knots not strictly increasing: {:?}", k);
        }
    }

    #[test]
    fn numeric_basis_partition_of_unity() {
        let knots = vec![0.0, 1.0, 2.0, 3.0];
        let mut out = Vec::new();
        push_numeric(&knots, 0, 1.25, &mut out);
        let sum: f32 = out.iter().map(|&(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // 1.25 brackets knots[1]=1 and knots[2]=2, weight 0.75 / 0.25.
        assert_eq!(out.len(), 2);
        assert!((out[0].1 - 0.75).abs() < 1e-6 && out[0].0 == 1);
        assert!((out[1].1 - 0.25).abs() < 1e-6 && out[1].0 == 2);
    }

    #[test]
    fn numeric_missing_goes_to_reserved_slot() {
        let knots = vec![0.0, 1.0, 2.0];
        let mut out = Vec::new();
        push_numeric(&knots, 10, f32::NAN, &mut out);
        assert_eq!(out, vec![(13, 1.0)]); // off=10 + K=3
    }

    #[test]
    fn target_encoder_is_out_of_fold_for_unique_ids() {
        // Each row a unique category → in-fold encoding would equal its own
        // label; out-of-fold encoding must fall back to the prior instead.
        let mut cfg = Config::default();
        cfg.alpha = 0.0; // no smoothing: makes leakage maximally visible
        cfg.n_folds = 5;
        let n = 500;
        let mut codes = Vec::new();
        let mut labels = Vec::new();
        for i in 0..n {
            codes.push(i as i64); // unique id per row
            labels.push((i % 2) as u8);
        }
        let weights = vec![1.0f32; n];
        let prior = weighted_rate(&labels, &weights);

        let nf = cfg.n_folds;
        let mut oof = vec![prior; n];
        for f in 0..nf {
            let train_rows = (0..n).filter(|i| i % nf != f);
            let (m, _) = target_enc_map(&codes, &labels, &weights, train_rows, cfg.alpha, prior);
            for i in (f..n).step_by(nf) {
                oof[i] = *m.get(&codes[i]).unwrap_or(&prior);
            }
        }
        // Every category is unique → never present out-of-fold → all prior.
        for (i, &e) in oof.iter().enumerate() {
            assert!(
                (e - prior).abs() < 1e-6,
                "row {i} leaked: enc={e} prior={prior}"
            );
        }
    }

    #[test]
    fn fit_transform_consistent_dims() {
        let ds = make_synthetic(3000, 9);
        let cfg = Config::default();
        let (enc, train_m) = Encoders::fit(&ds, &cfg);
        let eval_m = enc.transform(&ds);
        assert_eq!(train_m.cols.len(), eval_m.cols.len());
        assert!(enc.basis_dim > 0);

        let mut buf = Vec::new();
        expand_row(&enc, &eval_m, 0, &mut buf);
        // Every active basis index is within the declared dimension.
        for &(idx, _) in &buf {
            assert!(idx < enc.basis_dim, "idx {idx} >= dim {}", enc.basis_dim);
        }
    }
}
