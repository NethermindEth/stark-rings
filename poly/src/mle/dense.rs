use core::ops::IndexMut;

use ark_ff::Zero;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    cfg_iter, log2,
    ops::{Add, AddAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign},
    vec::*,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use stark_rings_linalg::SparseMatrix;

use super::{swap_bits, MultilinearExtension};
use stark_rings::Ring;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, CanonicalDeserialize, CanonicalSerialize)]
pub struct DenseMultilinearExtension<Rn: Ring> {
    /// The evaluation over {0,1}^`num_vars`.
    pub evaluations: Vec<Rn>,
    /// Number of variables.
    pub num_vars: usize,
    /// Extended length (= 2^num_vars).
    pub elen: usize,
    /// Zero element for OOB access.
    zero: Rn,
}

/// Representation of a dense multilinear extension (MLE).
impl<R: Ring> DenseMultilinearExtension<R> {
    /// Create a [`DenseMultilinearExtension`] from the input slice containing all evaluations.
    ///
    /// Only the evaluations until the last non-zero element are kept.
    pub fn from_evaluations_slice(num_vars: usize, evaluations: &[R]) -> Self {
        let elen = 1 << num_vars;
        // Length till last non-zero element
        let nzl = evaluations
            .iter()
            .enumerate()
            .rev()
            .find(|e| !e.1.is_zero())
            .map(|(i, _)| i + 1)
            .unwrap_or(0);
        let mut vec = Vec::with_capacity(nzl);
        vec.extend(evaluations);

        Self {
            num_vars,
            evaluations: vec,
            zero: R::zero(),
            elen,
        }
    }

    /// Create a [`DenseMultilinearExtension`] from the input vector containing all evaluations.
    ///
    /// The input vector is then truncated and shrunk to the last non-zero element.
    pub fn from_evaluations_vec(num_vars: usize, evaluations: Vec<R>) -> Self {
        let elen = 1 << num_vars;

        let mut dmle = Self {
            num_vars,
            evaluations,
            zero: R::zero(),
            elen,
        };
        dmle.truncate_lnze();

        dmle
    }

    /// Create a [`DenseMultilinearExtension`] from the input vector containing all evaluations.
    ///
    /// The input vector is then resized to 2^num_vars.
    pub fn from_evaluations_vec_padded(num_vars: usize, mut evaluations: Vec<R>) -> Self {
        let elen = 1 << num_vars;
        evaluations.resize(elen, R::zero());

        Self {
            num_vars,
            evaluations,
            zero: R::zero(),
            elen,
        }
    }

    /// Truncates and shrinks the evaluations vector to the last non-zero element.
    pub fn truncate_lnze(&mut self) {
        // Length till last non-zero element
        let nzl = self
            .evaluations
            .iter()
            .enumerate()
            .rev()
            .find(|e| !e.1.is_zero())
            .map(|(i, _)| i + 1)
            .unwrap_or(0);
        self.evaluations.truncate(nzl);
        self.evaluations.shrink_to_fit();
    }

    pub fn evaluate(&self, point: &[R]) -> Option<R> {
        if point.len() == self.num_vars {
            Some(self.fixed_variables(point)[0])
        } else {
            None
        }
    }

    /// Returns the dense MLE from the given matrix, without modifying the original matrix.
    pub fn from_matrix(matrix: &SparseMatrix<R>) -> Self {
        let n_vars: usize = (log2(matrix.nrows()) + log2(matrix.ncols())) as usize; // n_vars = s + s'

        // Matrices might need to get padded before turned into an MLE
        let padded_rows = matrix.nrows.next_power_of_two();
        let padded_cols = matrix.ncols.next_power_of_two();

        // build dense vector representing the sparse padded matrix
        let mut v: Vec<R> = vec![R::zero(); padded_rows * padded_cols];

        for (row_i, row) in matrix.coeffs.iter().enumerate() {
            for (val, col_i) in row {
                v[(padded_cols * row_i) + *col_i] = *val;
            }
        }

        // convert the dense vector into a mle
        Self::from_evaluations_vec(n_vars, v)
    }

    pub fn relabel_in_place(&mut self, mut a: usize, mut b: usize, k: usize) {
        // enforce order of a and b
        if a > b {
            ark_std::mem::swap(&mut a, &mut b);
        }
        if a == b || k == 0 {
            return;
        }
        assert!(b + k <= self.num_vars, "invalid relabel argument");
        assert!(a + k <= b, "overlapped swap window is not allowed");
        for i in 0..self.evaluations.len() {
            let j = swap_bits(i, a, b, k);
            if i < j {
                self.evaluations.swap(i, j);
            }
        }
    }
}

impl<R: Ring> MultilinearExtension<R> for DenseMultilinearExtension<R> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn rand<Rn: rand::Rng>(num_vars: usize, rng: &mut Rn) -> Self {
        Self::from_evaluations_vec(num_vars, (0..1 << num_vars).map(|_| R::rand(rng)).collect())
    }

    fn relabel(&self, a: usize, b: usize, k: usize) -> Self {
        let mut copy = self.clone();
        copy.relabel_in_place(a, b, k);
        copy
    }

    fn fix_variables(&mut self, partial_point: &[R]) {
        assert!(
            partial_point.len() <= self.num_vars,
            "too many partial points"
        );

        let nv = self.num_vars;
        let dim = partial_point.len();

        if !self.evaluations.is_empty() {
            for i in 1..dim + 1 {
                let r = partial_point[i - 1];
                for b in 0..1 << (nv - i) {
                    let left = self[b << 1];
                    let right = self[(b << 1) + 1];
                    let a = right - left;
                    if !a.is_zero() {
                        self[b] = left + r * a;
                    } else {
                        self[b] = left;
                    };
                }
            }
        }

        self.elen = 1 << (nv - dim);
        self.evaluations.truncate(self.elen);
        self.num_vars = nv - dim;
    }

    fn fixed_variables(&self, partial_point: &[R]) -> Self {
        let mut res = self.clone();
        res.fix_variables(partial_point);
        res
    }

    fn to_evaluations(&self) -> Vec<R> {
        self.evaluations.clone()
    }
}

impl<R: Ring> Zero for DenseMultilinearExtension<R> {
    fn zero() -> Self {
        Self {
            num_vars: 0,
            evaluations: vec![],
            zero: R::zero(),
            elen: 0,
        }
    }

    fn is_zero(&self) -> bool {
        self.num_vars == 0 && self.evaluations.is_empty()
    }
}

impl<R: Ring> Neg for DenseMultilinearExtension<R> {
    type Output = Self;

    fn neg(mut self) -> Self {
        cfg_iter_mut!(self.evaluations).for_each(|a| *a = a.neg());

        self
    }
}

impl<'a, R: Ring> AddAssign<&'a DenseMultilinearExtension<R>> for DenseMultilinearExtension<R> {
    fn add_assign(&mut self, rhs: &'a Self) {
        if self.is_zero() {
            *self = rhs.clone();
            return;
        }

        if rhs.is_zero() {
            return;
        }

        assert_eq!(
            self.num_vars, rhs.num_vars,
            "trying to add two dense MLEs with different numbers of variables"
        );

        if self.evaluations.len() < rhs.evaluations.len() {
            self.evaluations.resize(rhs.evaluations.len(), R::zero());
        }
        cfg_iter_mut!(self.evaluations)
            .zip(cfg_iter!(rhs.evaluations))
            .for_each(|(a, b)| a.add_assign(b));
    }
}

impl<R: Ring> AddAssign for DenseMultilinearExtension<R> {
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs);
    }
}

impl<R: Ring> Add for DenseMultilinearExtension<R> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let mut res = self;
        res += &rhs;
        res
    }
}

impl<'a, R: Ring> Add<&'a DenseMultilinearExtension<R>> for DenseMultilinearExtension<R> {
    type Output = Self;

    fn add(self, rhs: &'a Self) -> Self {
        let mut res = self;
        res += rhs;
        res
    }
}

impl<R: Ring> AddAssign<(R, &DenseMultilinearExtension<R>)> for DenseMultilinearExtension<R>
where
    R: Copy + ark_std::ops::AddAssign,
{
    fn add_assign(&mut self, (r, other): (R, &Self)) {
        if self.is_zero() {
            *self = other.clone();

            cfg_iter_mut!(self.evaluations).for_each(|a| a.mul_assign(&r));

            return;
        }

        if other.is_zero() {
            return;
        }

        assert_eq!(
            self.num_vars, other.num_vars,
            "trying to add two dense MLEs with different numbers of variables"
        );

        if self.evaluations.len() < other.evaluations.len() {
            self.evaluations.resize(other.evaluations.len(), R::zero());
        }
        cfg_iter_mut!(self.evaluations)
            .zip(cfg_iter!(other.evaluations))
            .for_each(|(a, b)| a.add_assign(r * b));
    }
}

impl<'a, R: Ring> SubAssign<&'a DenseMultilinearExtension<R>> for DenseMultilinearExtension<R> {
    fn sub_assign(&mut self, rhs: &'a Self) {
        if self.is_zero() {
            *self = rhs.clone().neg();
            return;
        }

        if rhs.is_zero() {
            return;
        }

        assert_eq!(
            self.num_vars, rhs.num_vars,
            "trying to subtract two dense MLEs with different numbers of variables"
        );

        if self.evaluations.len() < rhs.evaluations.len() {
            self.evaluations.resize(rhs.evaluations.len(), R::zero());
        }
        cfg_iter_mut!(self.evaluations)
            .zip(cfg_iter!(rhs.evaluations))
            .for_each(|(a, b)| a.sub_assign(b));
    }
}

impl<R: Ring> SubAssign for DenseMultilinearExtension<R> {
    fn sub_assign(&mut self, other: Self) {
        self.sub_assign(&other);
    }
}

impl<R: Ring> Sub for DenseMultilinearExtension<R> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut res = self;
        res -= &other;
        res
    }
}

impl<'a, R: Ring> Sub<&'a DenseMultilinearExtension<R>> for DenseMultilinearExtension<R> {
    type Output = Self;

    fn sub(self, rhs: &'a Self) -> Self {
        let mut res = self;
        res -= rhs;
        res
    }
}

impl<R: Ring> MulAssign<R> for DenseMultilinearExtension<R> {
    fn mul_assign(&mut self, rhs: R) {
        self.evaluations.iter_mut().for_each(|x| *x *= rhs);
    }
}

impl<R: Ring> Mul<R> for DenseMultilinearExtension<R> {
    type Output = Self;

    fn mul(self, rhs: R) -> Self {
        let mut res = self;
        res *= rhs;
        res
    }
}

impl<R: Ring> Add<R> for DenseMultilinearExtension<R> {
    type Output = Self;

    fn add(self, rhs: R) -> Self {
        let mut res = self;
        res.evaluations.iter_mut().for_each(|x| *x += rhs);

        res
    }
}

impl<R: Ring> Index<usize> for DenseMultilinearExtension<R> {
    type Output = R;

    fn index(&self, index: usize) -> &Self::Output {
        if index < self.evaluations.len() {
            &self.evaluations[index]
        } else {
            &self.zero
        }
    }
}

impl<R: Ring> IndexMut<usize> for DenseMultilinearExtension<R> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index < self.evaluations.len() {
            &mut self.evaluations[index]
        } else {
            self.evaluations.resize(self.elen, R::zero());
            &mut self.evaluations[index]
        }
    }
}
