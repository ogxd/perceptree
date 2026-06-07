//! The trained [`Model`] and single-row scorer.
//!
//! Inference is: encode high-card lookups → route through the coarse tree (a
//! handful of predicated compares) → compute the sparse basis → gather the
//! active leaf's weights and accumulate → sigmoid. There is a raw single-row
//! path ([`Model::predict_proba`]) and an encoded-matrix path used by training
//! and evaluation ([`Model::predict_proba_encoded`]).

use crate::data::{Column, Dataset};
use crate::encoding::{
    expand_row, push_numeric, EncCol, EncodedMatrix, Encoders, FeatureEncoder,
};
use crate::leaf_model::Leaf;
use crate::tree::{NodeKind, Split};

/// A single raw feature value for the single-row inference path.
#[derive(Clone, Copy, Debug)]
pub enum RawValue {
    /// Numeric (`f32::NAN` = missing).
    Num(f32),
    /// Categorical raw code ([`MISSING_CAT`] = missing).
    Cat(i64),
}

/// The trained model: encoders + routing tree + leaf heads + base score.
#[derive(Clone, Debug)]
pub struct Model {
    pub encoders: Encoders,
    pub nodes: Vec<NodeKind>,
    pub root: usize,
    pub leaves: Vec<Leaf>,
    /// Global base log-odds `s_0`.
    pub s0: f32,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Model {
    /// Build the sparse basis entries for a raw row.
    fn expand_raw(&self, row: &[RawValue], out: &mut Vec<(u32, f32)>) {
        out.clear();
        for (fi, fe) in self.encoders.features.iter().enumerate() {
            match (fe, row[fi]) {
                (FeatureEncoder::Numeric(e), RawValue::Num(v)) => {
                    push_numeric(&e.knots, e.basis_off, v, out);
                }
                (FeatureEncoder::BoundedCat(e), RawValue::Cat(c)) => {
                    out.push((e.basis_off + e.code_of(c), 1.0));
                }
                (FeatureEncoder::HighCard(e), RawValue::Cat(c)) => {
                    push_numeric(&e.enc_knots, e.enc_off, e.enc_of(c), out);
                    if e.use_count {
                        push_numeric(&e.cnt_knots, e.cnt_off, e.count_of(c), out);
                    }
                }
                _ => panic!("encoder/value kind mismatch at feature {fi}"),
            }
        }
    }

    /// The routing value of feature `fi` for a raw row: `Ok(numeric)` for
    /// numeric / high-card features, `Err(code)` for bounded categoricals.
    #[inline]
    fn route_value_raw(&self, fi: usize, row: &[RawValue]) -> Result<f32, u32> {
        match (&self.encoders.features[fi], row[fi]) {
            (FeatureEncoder::Numeric(_), RawValue::Num(v)) => Ok(v),
            (FeatureEncoder::HighCard(e), RawValue::Cat(c)) => Ok(e.enc_of(c)),
            (FeatureEncoder::BoundedCat(e), RawValue::Cat(c)) => Err(e.code_of(c)),
            _ => panic!("encoder/value kind mismatch at feature {fi}"),
        }
    }

    /// Walk the routing tree to a leaf id, given a routing accessor.
    #[inline]
    fn route<F>(&self, mut value: F) -> usize
    where
        F: FnMut(usize) -> Result<f32, u32>,
    {
        let mut node = self.root;
        loop {
            match &self.nodes[node] {
                NodeKind::Leaf(id) => return *id,
                NodeKind::Internal { feature, split, default_left, left, right } => {
                    let go_left = match split {
                        Split::Numeric { thr } => match value(*feature as usize) {
                            Ok(v) => {
                                if v.is_nan() {
                                    *default_left
                                } else {
                                    v < *thr
                                }
                            }
                            Err(_) => *default_left,
                        },
                        Split::Categorical { left_set } => match value(*feature as usize) {
                            Err(code) => left_set.contains(&code),
                            Ok(_) => *default_left,
                        },
                    };
                    node = if go_left { *left } else { *right };
                }
            }
        }
    }

    /// Single-row probability from raw feature values (the hot path).
    pub fn predict_proba(&self, row: &[RawValue]) -> f32 {
        let leaf_id = self.route(|fi| self.route_value_raw(fi, row));
        let mut phi = Vec::with_capacity(40);
        self.expand_raw(row, &mut phi);
        let leaf = &self.leaves[leaf_id];
        sigmoid(self.s0 + leaf.dot(&phi))
    }

    /// Routing value of feature `fi` for a row of an [`EncodedMatrix`].
    #[inline]
    fn route_value_encoded(&self, fi: usize, m: &EncodedMatrix, row: usize) -> Result<f32, u32> {
        match &m.cols[fi] {
            EncCol::Num(v) => Ok(v[row]),
            EncCol::HighCard { enc, .. } => Ok(enc[row]),
            EncCol::Cat(c) => Err(c[row]),
        }
    }

    /// Leaf id for a row of an encoded matrix.
    pub fn route_leaf_encoded(&self, m: &EncodedMatrix, row: usize) -> usize {
        self.route(|fi| self.route_value_encoded(fi, m, row))
    }

    /// Probability for a row of an encoded matrix (batch eval path).
    pub fn predict_proba_encoded(&self, m: &EncodedMatrix, row: usize) -> f32 {
        let leaf_id = self.route_leaf_encoded(m, row);
        let mut phi = Vec::with_capacity(40);
        expand_row(&self.encoders, m, row, &mut phi);
        sigmoid(self.s0 + self.leaves[leaf_id].dot(&phi))
    }

    /// Score every row of a dataset (encodes via the full-data maps first).
    pub fn predict_dataset(&self, ds: &Dataset) -> Vec<f32> {
        let m = self.encoders.transform(ds);
        (0..m.n_rows).map(|r| self.predict_proba_encoded(&m, r)).collect()
    }

    /// Extract row `r` of a dataset as a `RawValue` vector for the raw path.
    pub fn raw_row(ds: &Dataset, r: usize) -> Vec<RawValue> {
        ds.columns
            .iter()
            .map(|c| match c {
                Column::Numeric(v) => RawValue::Num(v[r]),
                Column::Categorical(v) => RawValue::Cat(v[r]),
            })
            .collect()
    }
}
