//! Coarse partition: histogram split-finding, a pluggable gain criterion
//! (constant-leaf / leaf-aware), and best-first growth to `K` leaves.
//!
//! Splits are found on **routing values** (raw numeric, target-encoded
//! high-card, dense categorical codes), not on the basis. All gradient/hessian
//! statistics are computed once at the constant base score `s_0`, since this is
//! a single-tree model.

use crate::encoding::{EncCol, EncodedMatrix, Encoders, FeatureEncoder};
use crate::{Config, GainKind};

/// How an internal node splits.
#[derive(Clone, Debug)]
pub enum Split {
    /// Route left when `value < thr` (missing uses `default_left`).
    Numeric { thr: f32 },
    /// Route left when the dense code is in `left_set`.
    Categorical { left_set: Vec<u32> },
}

/// A node in the routing tree (flat arena, children by index).
#[derive(Clone, Debug)]
pub enum NodeKind {
    Internal {
        feature: u32,
        split: Split,
        default_left: bool,
        left: usize,
        right: usize,
    },
    Leaf(usize),
}

/// Result of growth: the routing structure plus the rows landing in each leaf.
pub struct GrownTree {
    pub nodes: Vec<NodeKind>,
    pub root: usize,
    pub leaf_rows: Vec<Vec<u32>>,
}

/// Constant-leaf split score `G²/(H+λ)`.
#[inline]
fn score_const(g: f64, h: f64, lam: f64) -> f64 {
    g * g / (h + lam)
}

/// Leaf-aware (linear) side score `bᵀ A⁻¹ b` for the 2×2 reduced basis
/// `[1, z̃]`, with `A = [[Σh,Σhz],[Σhz,Σhz²]] + λI`, `b = [Σg, Σgz]`.
#[inline]
fn score_linear(h: f64, hz: f64, hzz: f64, g: f64, gz: f64, lam: f64) -> f64 {
    let a = h + lam;
    let b = hz;
    let d = hzz + lam;
    let det = a * d - b * b;
    if det <= 1e-12 {
        return score_const(g, h, lam);
    }
    let x0 = (d * g - b * gz) / det;
    let x1 = (-b * g + a * gz) / det;
    g * x0 + gz * x1
}

/// A candidate split for a node; row partition is recomputed on commit.
struct Candidate {
    gain: f32,
    feature: u32,
    split: Split,
    default_left: bool,
}

/// Per-bin moments for numeric split-finding.
#[derive(Clone, Copy, Default)]
struct Bin {
    g: f64,
    h: f64,
    gz: f64,
    hz: f64,
    hzz: f64,
    n: u64,
}

/// Normalize a routing value into `[0,1]` over the knot span for the leaf-aware
/// reduced basis.
#[inline]
fn znorm(knots: &[f32], v: f32) -> f64 {
    let lo = knots[0];
    let hi = knots[knots.len() - 1];
    if hi <= lo {
        return 0.0;
    }
    (((v - lo) / (hi - lo)) as f64).clamp(0.0, 1.0)
}

/// Best split on one numeric/high-card column over `rows`.
fn best_numeric(
    rows: &[u32],
    vals: &[f32],
    knots: &[f32],
    g: &[f32],
    h: &[f32],
    cfg: &Config,
    parent_score: f64,
) -> Option<Candidate> {
    let k = knots.len();
    let nbins = k + 1;
    let mut bins = vec![Bin::default(); nbins];
    let mut miss = Bin::default();

    for &r in rows {
        let r = r as usize;
        let v = vals[r];
        let gi = g[r] as f64;
        let hi = h[r] as f64;
        if v.is_nan() {
            miss.g += gi;
            miss.h += hi;
            miss.n += 1;
            continue;
        }
        let bin = knots.partition_point(|&kk| kk <= v); // 0..=k
        let z = znorm(knots, v);
        let b = &mut bins[bin];
        b.g += gi;
        b.h += hi;
        b.gz += gi * z;
        b.hz += hi * z;
        b.hzz += hi * z * z;
        b.n += 1;
    }

    // Totals over non-missing bins.
    let mut tot = Bin::default();
    for b in &bins {
        tot.g += b.g;
        tot.h += b.h;
        tot.gz += b.gz;
        tot.hz += b.hz;
        tot.hzz += b.hzz;
        tot.n += b.n;
    }

    let lam = cfg.lambda as f64;
    let leaf_aware = cfg.gain == GainKind::LeafAware;
    let min = cfg.min_leaf_samples as u64;

    let mut best: Option<Candidate> = None;
    let mut acc = Bin::default(); // running left side over bins 0..=t

    // Threshold at knots[t]: left = bins 0..=t, right = bins t+1..=k.
    for t in 0..k {
        let b = &bins[t];
        acc.g += b.g;
        acc.h += b.h;
        acc.gz += b.gz;
        acc.hz += b.hz;
        acc.hzz += b.hzz;
        acc.n += b.n;

        // right = totals - left.
        let rg = tot.g - acc.g;
        let rh = tot.h - acc.h;
        let rgz = tot.gz - acc.gz;
        let rhz = tot.hz - acc.hz;
        let rhzz = tot.hzz - acc.hzz;
        let rn = tot.n - acc.n;

        // Try both directions for missing rows.
        for default_left in [true, false] {
            let (ln, rn2) = if default_left {
                (acc.n + miss.n, rn)
            } else {
                (acc.n, rn + miss.n)
            };
            if ln < min || rn2 < min {
                continue;
            }
            let (lg, lh, lgz, lhz, lhzz, rg2, rh2) = if default_left {
                (
                    acc.g + miss.g,
                    acc.h + miss.h,
                    acc.gz,
                    acc.hz,
                    acc.hzz,
                    rg,
                    rh,
                )
            } else {
                (acc.g, acc.h, acc.gz, acc.hz, acc.hzz, rg + miss.g, rh + miss.h)
            };
            let (sl, sr) = if leaf_aware {
                (
                    score_linear(lh, lhz, lhzz, lg, lgz, lam),
                    score_linear(rh2, rhz, rhzz, rg2, rgz, lam),
                )
            } else {
                (score_const(lg, lh, lam), score_const(rg2, rh2, lam))
            };
            let gain = (sl + sr - parent_score) as f32;
            if best.as_ref().map_or(true, |c| gain > c.gain) {
                best = Some(Candidate {
                    gain,
                    feature: 0, // filled by caller
                    split: Split::Numeric { thr: knots[t] },
                    default_left,
                });
            }
        }
    }
    best
}

/// Best many-vs-many split on a bounded categorical column over `rows`.
/// Categories are sorted by `G/H` and a contiguous prefix is taken as the left
/// group. Always scored with the constant-leaf criterion.
fn best_categorical(
    rows: &[u32],
    codes: &[u32],
    g: &[f32],
    h: &[f32],
    cfg: &Config,
    parent_score: f64,
) -> Option<Candidate> {
    use std::collections::HashMap;
    let mut stat: HashMap<u32, (f64, f64, u64)> = HashMap::new();
    let mut tot_g = 0.0;
    let mut tot_h = 0.0;
    for &r in rows {
        let r = r as usize;
        let e = stat.entry(codes[r]).or_insert((0.0, 0.0, 0));
        e.0 += g[r] as f64;
        e.1 += h[r] as f64;
        e.2 += 1;
        tot_g += g[r] as f64;
        tot_h += h[r] as f64;
    }
    if stat.len() < 2 {
        return None;
    }
    let lam = cfg.lambda as f64;
    let min = cfg.min_leaf_samples as u64;

    let mut order: Vec<(u32, f64, f64, u64)> = stat
        .iter()
        .map(|(&c, &(g, h, n))| (c, g, h, n))
        .collect();
    order.sort_by(|a, b| {
        let ra = a.1 / (a.2 as f64).max(1.0);
        let rb = b.1 / (b.2 as f64).max(1.0);
        ra.partial_cmp(&rb).unwrap()
    });

    let mut best: Option<Candidate> = None;
    let mut lg = 0.0;
    let mut lh = 0.0;
    let mut ln = 0u64;
    let mut left_set: Vec<u32> = Vec::new();
    for i in 0..order.len() - 1 {
        let (c, gg, hh, nn) = order[i];
        lg += gg;
        lh += hh;
        ln += nn;
        left_set.push(c);
        let rn = (rows.len() as u64) - ln;
        if ln < min || rn < min {
            continue;
        }
        let sl = score_const(lg, lh, lam);
        let sr = score_const(tot_g - lg, tot_h - lh, lam);
        let gain = (sl + sr - parent_score) as f32;
        if best.as_ref().map_or(true, |c| gain > c.gain) {
            best = Some(Candidate {
                gain,
                feature: 0,
                split: Split::Categorical { left_set: left_set.clone() },
                default_left: true,
            });
        }
    }
    best
}

/// Parent score for the constant or leaf-aware criterion over `rows`,
/// per feature kind, so `gain = sL + sR − sP` is comparable across features.
fn parent_score_for(
    rows: &[u32],
    feat: &FeatureEncoder,
    col: &EncCol,
    g: &[f32],
    h: &[f32],
    cfg: &Config,
) -> f64 {
    let lam = cfg.lambda as f64;

    // Numeric / high-card under the leaf-aware criterion need the full reduced
    // moments so the parent is scored on the same basis as the children.
    if cfg.gain == GainKind::LeafAware {
        let knots: Option<(&[f32], &[f32])> = match (feat, col) {
            (FeatureEncoder::Numeric(e), EncCol::Num(vals)) => Some((&e.knots, vals)),
            (FeatureEncoder::HighCard(e), EncCol::HighCard { enc, .. }) => Some((&e.enc_knots, enc)),
            _ => None,
        };
        if let Some((knots, vals)) = knots {
            let (mut sg, mut sh, mut sgz, mut shz, mut shzz) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for &r in rows {
                let r = r as usize;
                let v = vals[r];
                let gi = g[r] as f64;
                let hi = h[r] as f64;
                let z = if v.is_nan() { 0.0 } else { znorm(knots, v) };
                sg += gi;
                sh += hi;
                sgz += gi * z;
                shz += hi * z;
                shzz += hi * z * z;
            }
            return score_linear(sh, shz, shzz, sg, sgz, lam);
        }
    }

    // Constant criterion (and all categorical features): G²/(H+λ).
    let (mut sg, mut sh) = (0.0, 0.0);
    for &r in rows {
        sg += g[r as usize] as f64;
        sh += h[r as usize] as f64;
    }
    score_const(sg, sh, lam)
}

/// Find the best split for a node across all features.
fn find_best_split(
    rows: &[u32],
    encoders: &Encoders,
    matrix: &EncodedMatrix,
    g: &[f32],
    h: &[f32],
    cfg: &Config,
) -> Option<Candidate> {
    let mut best: Option<Candidate> = None;
    for (fi, (feat, col)) in encoders.features.iter().zip(&matrix.cols).enumerate() {
        let parent = parent_score_for(rows, feat, col, g, h, cfg);
        let cand = match (feat, col) {
            (FeatureEncoder::Numeric(e), EncCol::Num(vals)) => {
                best_numeric(rows, vals, &e.knots, g, h, cfg, parent)
            }
            (FeatureEncoder::HighCard(e), EncCol::HighCard { enc, .. }) => {
                best_numeric(rows, enc, &e.enc_knots, g, h, cfg, parent)
            }
            (FeatureEncoder::BoundedCat(_), EncCol::Cat(codes)) => {
                best_categorical(rows, codes, g, h, cfg, parent)
            }
            _ => None,
        };
        if let Some(mut c) = cand {
            c.feature = fi as u32;
            if c.gain > 0.0 && best.as_ref().map_or(true, |b| c.gain > b.gain) {
                best = Some(c);
            }
        }
    }
    best
}

/// Route a node's rows into (left, right) for a committed split.
fn partition(
    rows: &[u32],
    feature: usize,
    split: &Split,
    default_left: bool,
    matrix: &EncodedMatrix,
) -> (Vec<u32>, Vec<u32>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    match (split, &matrix.cols[feature]) {
        (Split::Numeric { thr }, EncCol::Num(v)) => {
            for &r in rows {
                let val = v[r as usize];
                let go_left = if val.is_nan() { default_left } else { val < *thr };
                if go_left {
                    left.push(r);
                } else {
                    right.push(r);
                }
            }
        }
        (Split::Numeric { thr }, EncCol::HighCard { enc, .. }) => {
            for &r in rows {
                let val = enc[r as usize];
                let go_left = if val.is_nan() { default_left } else { val < *thr };
                if go_left {
                    left.push(r);
                } else {
                    right.push(r);
                }
            }
        }
        (Split::Categorical { left_set }, EncCol::Cat(codes)) => {
            for &r in rows {
                if left_set.contains(&codes[r as usize]) {
                    left.push(r);
                } else {
                    right.push(r);
                }
            }
        }
        _ => panic!("split/column kind mismatch"),
    }
    (left, right)
}

/// An open (not-yet-split) leaf and its best candidate split.
struct Open {
    node: usize,
    rows: Vec<u32>,
    cand: Option<Candidate>,
}

/// Grow a coarse tree best-first to at most `cfg.num_leaves` leaves.
pub fn grow(
    encoders: &Encoders,
    matrix: &EncodedMatrix,
    g: &[f32],
    h: &[f32],
    cfg: &Config,
) -> GrownTree {
    let n = matrix.n_rows;
    let all: Vec<u32> = (0..n as u32).collect();
    let mut nodes: Vec<NodeKind> = vec![NodeKind::Leaf(usize::MAX)];
    let root = 0;

    let root_cand = find_best_split(&all, encoders, matrix, g, h, cfg);
    let mut open = vec![Open { node: root, rows: all, cand: root_cand }];
    let mut n_leaves = 1;

    while n_leaves < cfg.num_leaves {
        // Pick the open leaf with the highest-gain candidate.
        let pick = open
            .iter()
            .enumerate()
            .filter_map(|(i, o)| o.cand.as_ref().map(|c| (i, c.gain)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let Some((idx, gain)) = pick else { break };
        if gain <= 0.0 {
            break;
        }

        let o = open.swap_remove(idx);
        let cand = o.cand.unwrap();
        let (lrows, rrows) =
            partition(&o.rows, cand.feature as usize, &cand.split, cand.default_left, matrix);

        let l = nodes.len();
        nodes.push(NodeKind::Leaf(usize::MAX));
        let r = nodes.len();
        nodes.push(NodeKind::Leaf(usize::MAX));
        nodes[o.node] = NodeKind::Internal {
            feature: cand.feature,
            split: cand.split,
            default_left: cand.default_left,
            left: l,
            right: r,
        };

        let lc = find_best_split(&lrows, encoders, matrix, g, h, cfg);
        let rc = find_best_split(&rrows, encoders, matrix, g, h, cfg);
        open.push(Open { node: l, rows: lrows, cand: lc });
        open.push(Open { node: r, rows: rrows, cand: rc });
        n_leaves += 1;
    }

    // Assign leaf ids to the remaining open leaves.
    let mut leaf_rows = Vec::with_capacity(open.len());
    for (leaf_id, o) in open.into_iter().enumerate() {
        nodes[o.node] = NodeKind::Leaf(leaf_id);
        leaf_rows.push(o.rows);
    }
    GrownTree { nodes, root, leaf_rows }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Column, Dataset, FeatureKind, Schema};
    use crate::encoding::Encoders;

    /// Brute-force best constant-gain numeric threshold over candidate knots,
    /// compared against the histogram split-finder.
    #[test]
    fn numeric_gain_matches_brute_force() {
        // One numeric feature, a clean threshold effect at ~0.
        let n = 2000;
        let mut col = Vec::new();
        let mut labels = Vec::new();
        let mut rng = crate::data::Rng::new(3);
        for _ in 0..n {
            let x = rng.next_normal();
            col.push(x);
            labels.push(if x > 0.0 { (rng.next_f32() < 0.7) as u8 } else { (rng.next_f32() < 0.1) as u8 });
        }
        let schema = Schema::new(vec!["x".into()], vec![FeatureKind::Numeric]);
        let ds = Dataset::new(schema, vec![Column::Numeric(col.clone())], labels.clone());
        let mut cfg = Config::default();
        cfg.min_leaf_samples = 1;
        cfg.gain = GainKind::Constant;
        let (enc, _m) = Encoders::fit(&ds, &cfg);

        // Stats at base score.
        let rate = labels.iter().map(|&y| y as f64).sum::<f64>() / n as f64;
        let p0 = rate as f32;
        let g: Vec<f32> = labels.iter().map(|&y| p0 - y as f32).collect();
        let h: Vec<f32> = vec![p0 * (1.0 - p0); n];

        let rows: Vec<u32> = (0..n as u32).collect();
        let knots = match &enc.features[0] {
            FeatureEncoder::Numeric(e) => e.knots.clone(),
            _ => unreachable!(),
        };
        let parent = score_const(
            g.iter().map(|&x| x as f64).sum(),
            h.iter().map(|&x| x as f64).sum(),
            cfg.lambda as f64,
        );
        let found = best_numeric(&rows, &col, &knots, &g, &h, &cfg, parent).unwrap();

        // Brute force over the same candidate thresholds.
        let mut brute = f64::MIN;
        for t in 0..knots.len() {
            let thr = knots[t];
            let (mut lg, mut lh, mut rg, mut rh) = (0.0, 0.0, 0.0, 0.0);
            for i in 0..n {
                if col[i] < thr {
                    lg += g[i] as f64;
                    lh += h[i] as f64;
                } else {
                    rg += g[i] as f64;
                    rh += h[i] as f64;
                }
            }
            let gain = score_const(lg, lh, cfg.lambda as f64)
                + score_const(rg, rh, cfg.lambda as f64)
                - parent;
            if gain > brute {
                brute = gain;
            }
        }
        assert!(
            (found.gain as f64 - brute).abs() < 1e-3,
            "histogram gain {} vs brute {}",
            found.gain,
            brute
        );
    }

    #[test]
    fn grows_to_requested_leaves() {
        let ds = crate::data::make_synthetic(8000, 11);
        let mut cfg = Config::default();
        cfg.num_leaves = 8;
        let (enc, matrix) = Encoders::fit(&ds, &cfg);
        let rate = ds.labels.iter().map(|&y| y as f64).sum::<f64>() / ds.n_rows() as f64;
        let p0 = rate as f32;
        let g: Vec<f32> = ds.labels.iter().map(|&y| p0 - y as f32).collect();
        let h: Vec<f32> = vec![p0 * (1.0 - p0); ds.n_rows()];
        let tree = grow(&enc, &matrix, &g, &h, &cfg);
        assert!(tree.leaf_rows.len() <= 8 && tree.leaf_rows.len() >= 2);
        // All rows accounted for exactly once.
        let total: usize = tree.leaf_rows.iter().map(|r| r.len()).sum();
        assert_eq!(total, ds.n_rows());
    }
}
