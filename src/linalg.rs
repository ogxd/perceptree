//! Minimal dense linear algebra: symmetric positive-(semi)definite solves via
//! Cholesky. Used by the leaf-model Newton/IRLS fit (up to d×d) and by the
//! leaf-aware split criterion (2×2). Everything is `f64` internally for
//! numerical stability; callers cast results to `f32`.

/// A dense symmetric matrix stored row-major (full storage, `n*n`).
#[derive(Clone, Debug)]
pub struct SymMatrix {
    pub n: usize,
    pub data: Vec<f64>,
}

impl SymMatrix {
    pub fn zeros(n: usize) -> Self {
        SymMatrix { n, data: vec![0.0; n * n] }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    #[inline]
    pub fn add(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.n + j] += v;
    }

    /// Add `v` to the diagonal (L2 regularization `+ λI`).
    pub fn add_diag(&mut self, v: f64) {
        for i in 0..self.n {
            self.data[i * self.n + i] += v;
        }
    }
}

/// Solve `A x = b` for symmetric positive-definite `A` via Cholesky
/// (`A = L Lᵀ`). Returns `None` if `A` is not numerically PD (caller should
/// have added enough L2 to guarantee PD). `A` is consumed as scratch.
pub fn cholesky_solve(mut a: SymMatrix, b: &[f64]) -> Option<Vec<f64>> {
    let n = a.n;
    debug_assert_eq!(b.len(), n);

    // In-place Cholesky factorization; lower triangle of `a.data` becomes L.
    for j in 0..n {
        let mut diag = a.get(j, j);
        for k in 0..j {
            let ljk = a.data[j * n + k];
            diag -= ljk * ljk;
        }
        if diag <= 0.0 || !diag.is_finite() {
            return None;
        }
        let ljj = diag.sqrt();
        a.data[j * n + j] = ljj;
        for i in (j + 1)..n {
            let mut s = a.get(i, j);
            for k in 0..j {
                s -= a.data[i * n + k] * a.data[j * n + k];
            }
            a.data[i * n + j] = s / ljj;
        }
    }

    // Forward solve L y = b.
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a.data[i * n + k] * y[k];
        }
        y[i] = s / a.data[i * n + i];
    }

    // Back solve Lᵀ x = y.
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= a.data[k * n + i] * x[k];
        }
        x[i] = s / a.data[i * n + i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_2x2() {
        // [[4,1],[1,3]] x = [1,2] -> x = [1/11, 7/11]
        let mut a = SymMatrix::zeros(2);
        a.add(0, 0, 4.0);
        a.add(0, 1, 1.0);
        a.add(1, 0, 1.0);
        a.add(1, 1, 3.0);
        let x = cholesky_solve(a, &[1.0, 2.0]).unwrap();
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-9);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-9);
    }

    #[test]
    fn matches_identity() {
        let mut a = SymMatrix::zeros(3);
        a.add_diag(2.0);
        let x = cholesky_solve(a, &[2.0, 4.0, 6.0]).unwrap();
        for (xi, expect) in x.iter().zip([1.0, 2.0, 3.0]) {
            assert!((xi - expect).abs() < 1e-9);
        }
    }

    #[test]
    fn rejects_non_pd() {
        let mut a = SymMatrix::zeros(2);
        a.add(0, 0, 1.0);
        a.add(0, 1, 2.0);
        a.add(1, 0, 2.0);
        a.add(1, 1, 1.0); // indefinite
        assert!(cholesky_solve(a, &[1.0, 1.0]).is_none());
    }
}
