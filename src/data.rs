//! Columnar (struct-of-arrays) data loading and the feature schema.
//!
//! Raw values use sentinel encodings for missing: `f32::NAN` for numeric
//! columns and [`MISSING_CAT`] for categorical columns. The dataset carries an
//! implicit row order; the temporal train/valid split is "first `frac` rows are
//! train", so callers should pass rows pre-sorted by time.

/// Sentinel category code marking a missing categorical value.
pub const MISSING_CAT: i64 = i64::MIN;

/// How a raw feature column is treated by the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureKind {
    /// Continuous value; quantile knots + piecewise-linear basis.
    Numeric,
    /// Known finite vocabulary; one-hot basis, many-vs-many splits.
    BoundedCat,
    /// Unbounded / high-cardinality; out-of-fold target encoded to a numeric.
    HighCardCat,
}

/// Names + kinds for every feature column, in column order.
#[derive(Clone, Debug)]
pub struct Schema {
    pub names: Vec<String>,
    pub kinds: Vec<FeatureKind>,
}

impl Schema {
    pub fn new(names: Vec<String>, kinds: Vec<FeatureKind>) -> Self {
        assert_eq!(names.len(), kinds.len());
        Schema { names, kinds }
    }
    pub fn n_features(&self) -> usize {
        self.kinds.len()
    }
}

/// One raw feature column.
#[derive(Clone, Debug)]
pub enum Column {
    /// `f32::NAN` marks missing.
    Numeric(Vec<f32>),
    /// [`MISSING_CAT`] marks missing.
    Categorical(Vec<i64>),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Column::Numeric(v) => v.len(),
            Column::Categorical(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A fully materialized dataset: aligned feature columns, labels, weights.
#[derive(Clone, Debug)]
pub struct Dataset {
    pub schema: Schema,
    pub columns: Vec<Column>,
    pub labels: Vec<u8>,
    /// Per-row class weight `c_i` (1.0 if unset; positives often up-weighted).
    pub weights: Vec<f32>,
}

impl Dataset {
    pub fn new(schema: Schema, columns: Vec<Column>, labels: Vec<u8>) -> Self {
        let n = labels.len();
        assert_eq!(columns.len(), schema.n_features());
        for c in &columns {
            assert_eq!(c.len(), n, "all columns must have equal length");
        }
        Dataset { schema, columns, labels, weights: vec![1.0; n] }
    }

    pub fn n_rows(&self) -> usize {
        self.labels.len()
    }

    /// Set per-row weights so positives carry `pos_weight` and negatives 1.0.
    pub fn set_class_weights(&mut self, pos_weight: f32) {
        for (w, &y) in self.weights.iter_mut().zip(&self.labels) {
            *w = if y == 1 { pos_weight } else { 1.0 };
        }
    }

    /// Temporal split: first `train_frac` of rows → train, remainder → valid.
    pub fn temporal_split(&self, train_frac: f64) -> (Dataset, Dataset) {
        let n = self.n_rows();
        let n_train = ((n as f64) * train_frac).round() as usize;
        (self.slice(0, n_train), self.slice(n_train, n))
    }

    fn slice(&self, lo: usize, hi: usize) -> Dataset {
        let columns = self
            .columns
            .iter()
            .map(|c| match c {
                Column::Numeric(v) => Column::Numeric(v[lo..hi].to_vec()),
                Column::Categorical(v) => Column::Categorical(v[lo..hi].to_vec()),
            })
            .collect();
        Dataset {
            schema: self.schema.clone(),
            columns,
            labels: self.labels[lo..hi].to_vec(),
            weights: self.weights[lo..hi].to_vec(),
        }
    }
}

/// A tiny deterministic xorshift RNG so the synthetic generator and tests are
/// fully reproducible without pulling in a dependency.
#[derive(Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed | 1 }
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    /// Uniform in [0, 1).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Approx standard normal via sum of 6 uniforms (Irwin–Hall, centered).
    pub fn next_normal(&mut self) -> f32 {
        let mut s = 0.0f32;
        for _ in 0..6 {
            s += self.next_f32();
        }
        (s - 3.0) * std::f32::consts::SQRT_2
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// Generate a synthetic imbalanced benchmark dataset matching the spec schema:
/// 10 numeric + 10 bounded categorical + 5 high-cardinality categorical.
///
/// The target depends on a nonlinear function of two numerics, an interaction
/// between a numeric and a bounded category, and a per-id signal carried by one
/// high-card feature — exactly the structure a coarse partition + linear leaves
/// should capture. Some values are randomly set missing.
pub fn make_synthetic(n_rows: usize, seed: u64) -> Dataset {
    let n_num = 10;
    let n_bcat = 10;
    let n_hcat = 5;
    let bcat_vocab = 8u64;
    let hcat_vocab = 5000u64;

    let mut rng = Rng::new(seed);
    let mut names = Vec::new();
    let mut kinds = Vec::new();
    for i in 0..n_num {
        names.push(format!("num_{i}"));
        kinds.push(FeatureKind::Numeric);
    }
    for i in 0..n_bcat {
        names.push(format!("bcat_{i}"));
        kinds.push(FeatureKind::BoundedCat);
    }
    for i in 0..n_hcat {
        names.push(format!("hcat_{i}"));
        kinds.push(FeatureKind::HighCardCat);
    }
    let schema = Schema::new(names, kinds);

    let mut num_cols: Vec<Vec<f32>> = vec![Vec::with_capacity(n_rows); n_num];
    let mut bcat_cols: Vec<Vec<i64>> = vec![Vec::with_capacity(n_rows); n_bcat];
    let mut hcat_cols: Vec<Vec<i64>> = vec![Vec::with_capacity(n_rows); n_hcat];
    let mut labels = Vec::with_capacity(n_rows);

    // Latent per-id risk for the first high-card feature (stable signal).
    let id_risk: Vec<f32> = (0..hcat_vocab)
        .map(|_| rng.next_normal() * 0.9)
        .collect();

    for _ in 0..n_rows {
        let num: Vec<f32> = (0..n_num).map(|_| rng.next_normal()).collect();
        let bcat: Vec<u64> = (0..n_bcat).map(|_| rng.below(bcat_vocab)).collect();
        let hcat: Vec<u64> = (0..n_hcat).map(|_| rng.below(hcat_vocab)).collect();

        // True log-odds: nonlinear + interaction + per-id signal, low base rate.
        let mut s = -3.2f32;
        s += 0.8 * (num[0] * num[0] - 1.0); // nonlinear in num_0
        s += 1.1 * num[1] * (if bcat[0] % 2 == 0 { 1.0 } else { -1.0 }); // interaction
        s += 0.6 * num[2];
        s += if bcat[1] < 2 { 1.3 } else { 0.0 }; // categorical step
        s += id_risk[hcat[0] as usize]; // high-card per-id signal
        let p = 1.0 / (1.0 + (-s).exp());
        let y = if rng.next_f32() < p { 1u8 } else { 0u8 };
        labels.push(y);

        for (j, &v) in num.iter().enumerate() {
            // ~3% missing on numeric columns.
            let stored = if rng.next_f32() < 0.03 { f32::NAN } else { v };
            num_cols[j].push(stored);
        }
        for (j, &v) in bcat.iter().enumerate() {
            let stored = if rng.next_f32() < 0.02 { MISSING_CAT } else { v as i64 };
            bcat_cols[j].push(stored);
        }
        for (j, &v) in hcat.iter().enumerate() {
            hcat_cols[j].push(v as i64);
        }
    }

    let mut columns = Vec::new();
    for c in num_cols {
        columns.push(Column::Numeric(c));
    }
    for c in bcat_cols {
        columns.push(Column::Categorical(c));
    }
    for c in hcat_cols {
        columns.push(Column::Categorical(c));
    }
    Dataset::new(schema, columns, labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_shapes_and_imbalance() {
        let ds = make_synthetic(20_000, 42);
        assert_eq!(ds.n_rows(), 20_000);
        assert_eq!(ds.columns.len(), 25);
        let pos: usize = ds.labels.iter().map(|&y| y as usize).sum();
        let rate = pos as f64 / ds.n_rows() as f64;
        // Imbalanced but not degenerate.
        assert!(rate > 0.01 && rate < 0.30, "rate={rate}");
    }

    #[test]
    fn temporal_split_sizes() {
        let ds = make_synthetic(1000, 1);
        let (tr, va) = ds.temporal_split(0.8);
        assert_eq!(tr.n_rows(), 800);
        assert_eq!(va.n_rows(), 200);
    }

    #[test]
    fn rng_is_deterministic() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        assert_eq!(a.next_u64(), b.next_u64());
    }
}
