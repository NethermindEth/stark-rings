use core::ops::IndexMut;

use ark_ff::Zero;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    borrow::ToOwned,
    cfg_iter, log2,
    ops::{Add, AddAssign, Index, Mul, MulAssign, Neg, Sub, SubAssign},
    vec::*,
};
use lattirust_linear_algebra::SparseMatrix;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{swap_bits, MultilinearExtension};
use lattirust_ring::Ring;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, CanonicalDeserialize, CanonicalSerialize)]
pub struct DenseMultilinearExtension<Rn: Ring> {
    /// The evaluation over {0,1}^`num_vars`
    pub evaluations: Vec<Rn>,
    /// Number of variables
    pub num_vars: usize,
}

impl<R: Ring> DenseMultilinearExtension<R> {
    pub fn from_evaluations_slice(num_vars: usize, evaluations: &[R]) -> Self {
        Self::from_evaluations_vec(num_vars, evaluations.to_vec())
    }

    pub fn evaluate(&self, point: &[R]) -> Option<R> {
        if point.len() == self.num_vars {
            Some(self.fixed_variables(point)[0])
        } else {
            None
        }
    }

    pub fn from_evaluations_vec(num_vars: usize, evaluations: Vec<R>) -> Self {
        // assert that the number of variables matches the size of evaluations
        assert_eq!(
            evaluations.len(),
            1 << num_vars,
            "The size of evaluations should be 2^num_vars."
        );

        Self {
            num_vars,
            evaluations,
        }
    }

    /// Returns the dense MLE from the given matrix, without modifying the original matrix.
    pub fn from_matrix(matrix: &SparseMatrix<R>) -> Self {
        let n_vars: usize = (log2(matrix.nrows()) + log2(matrix.ncols())) as usize; // n_vars = s + s'

        // Matrices might need to get padded before turned into an MLE
        let padded_rows = matrix.n_rows.next_power_of_two();
        let padded_cols = matrix.n_cols.next_power_of_two();

        // build dense vector representing the sparse padded matrix
        let mut v: Vec<R> = vec![R::zero(); padded_rows * padded_cols];

        for (row_i, row) in matrix.coeffs.iter().enumerate() {
            for (val, col_i) in row {
                v[(padded_cols * row_i) + *col_i] = *val;
            }
        }

        // convert the dense vector into a mle
        Self::from_slice(n_vars, &v)
    }

    /// Takes n_vars and a dense slice and returns its dense MLE.
    pub fn from_slice(n_vars: usize, v: &[R]) -> Self {
        let v_padded: Vec<R> = if v.len() < (1 << n_vars) {
            // pad to 2^n_vars
            [
                v.to_owned(),
                ark_std::iter::repeat(R::zero())
                    .take((1 << n_vars) - v.len())
                    .collect(),
            ]
            .concat()
        } else {
            v.to_owned()
        };
        DenseMultilinearExtension::<R>::from_evaluations_vec(n_vars, v_padded)
    }

    /// Takes n_vars and a dense vector and returns its dense MLE.
    pub fn from_vec(n_vars: usize, mut v: Vec<R>) -> Self {
        if v.len() < (1 << n_vars) {
            v.resize(1 << n_vars, R::zero());
        }
        DenseMultilinearExtension::<R>::from_evaluations_vec(n_vars, v)
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

        let poly = &mut self.evaluations;
        let nv = self.num_vars;
        let dim = partial_point.len();

        for i in 1..dim + 1 {
            let r = partial_point[i - 1];
            for b in 0..1 << (nv - i) {
                let left = poly[b << 1];
                let right = poly[(b << 1) + 1];
                let a = right - left;
                if !a.is_zero() {
                    poly[b] = left + r * a;
                } else {
                    poly[b] = left;
                };
            }
        }

        self.evaluations.truncate(1 << (nv - dim));
        self.num_vars = nv - dim;
    }

    fn fixed_variables(&self, partial_point: &[R]) -> Self {
        let mut res = self.clone();
        res.fix_variables(partial_point);
        res
    }

    fn to_evaluations(&self) -> Vec<R> {
        self.evaluations.to_vec()
    }
}

impl<R: Ring> Zero for DenseMultilinearExtension<R> {
    fn zero() -> Self {
        Self {
            num_vars: 0,
            evaluations: vec![R::zero()],
        }
    }

    fn is_zero(&self) -> bool {
        self.num_vars == 0 && self.evaluations[0].is_zero()
    }
}

impl<R: Ring> Add for DenseMultilinearExtension<R> {
    type Output = DenseMultilinearExtension<R>;

    fn add(self, other: DenseMultilinearExtension<R>) -> Self {
        &self + &other
    }
}

impl<'a, R: Ring> Add<&'a DenseMultilinearExtension<R>> for &DenseMultilinearExtension<R> {
    type Output = DenseMultilinearExtension<R>;

    fn add(self, rhs: &'a DenseMultilinearExtension<R>) -> Self::Output {
        if rhs.is_zero() {
            return self.clone();
        }

        if self.is_zero() {
            return rhs.clone();
        }

        assert_eq!(
            self.num_vars, rhs.num_vars,
            "trying to add two dense MLEs with different numbers of variables"
        );

        let result: Vec<R> = cfg_iter!(self.evaluations)
            .zip(cfg_iter!(rhs.evaluations))
            .map(|(a, b)| *a + b)
            .collect();

        Self::Output::from_evaluations_vec(self.num_vars, result)
    }
}

impl<R: Ring> AddAssign for DenseMultilinearExtension<R> {
    fn add_assign(&mut self, rhs: DenseMultilinearExtension<R>) {
        self.add_assign(&rhs);
    }
}

impl<'a, R: Ring> AddAssign<&'a DenseMultilinearExtension<R>> for DenseMultilinearExtension<R> {
    fn add_assign(&mut self, rhs: &'a DenseMultilinearExtension<R>) {
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

        cfg_iter_mut!(self.evaluations)
            .zip(cfg_iter!(rhs.evaluations))
            .for_each(|(a, b)| a.add_assign(b));
    }
}

impl<R: Ring> AddAssign<(R, &DenseMultilinearExtension<R>)> for DenseMultilinearExtension<R>
where
    R: Copy + ark_std::ops::AddAssign,
{
    fn add_assign(&mut self, (r, other): (R, &DenseMultilinearExtension<R>)) {
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

        cfg_iter_mut!(self.evaluations)
            .zip(cfg_iter!(other.evaluations))
            .for_each(|(a, b)| a.add_assign(r * b));
    }
}

impl<R: Ring> Neg for DenseMultilinearExtension<R> {
    type Output = DenseMultilinearExtension<R>;

    fn neg(mut self) -> Self {
        cfg_iter_mut!(self.evaluations).for_each(|a| *a = a.neg());

        self
    }
}

impl<R: Ring> Sub for DenseMultilinearExtension<R> {
    type Output = DenseMultilinearExtension<R>;

    fn sub(self, other: DenseMultilinearExtension<R>) -> Self {
        &self - &other
    }
}

impl<'a, R: Ring> Sub<&'a DenseMultilinearExtension<R>> for &DenseMultilinearExtension<R> {
    type Output = DenseMultilinearExtension<R>;

    fn sub(self, rhs: &'a DenseMultilinearExtension<R>) -> Self::Output {
        if rhs.is_zero() {
            return self.clone();
        }

        if self.is_zero() {
            return rhs.clone().neg();
        }

        assert_eq!(
            self.num_vars, rhs.num_vars,
            "trying to subtract two dense MLEs with different numbers of variables"
        );

        let result: Vec<R> = cfg_iter!(self.evaluations)
            .zip(cfg_iter!(rhs.evaluations))
            .map(|(a, b)| *a - b)
            .collect();

        Self::Output::from_evaluations_vec(self.num_vars, result)
    }
}

impl<R: Ring> SubAssign for DenseMultilinearExtension<R> {
    fn sub_assign(&mut self, other: DenseMultilinearExtension<R>) {
        self.sub_assign(&other);
    }
}

impl<'a, R: Ring> SubAssign<&'a DenseMultilinearExtension<R>> for DenseMultilinearExtension<R> {
    fn sub_assign(&mut self, rhs: &'a DenseMultilinearExtension<R>) {
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

        cfg_iter_mut!(self.evaluations)
            .zip(cfg_iter!(rhs.evaluations))
            .for_each(|(a, b)| a.sub_assign(b));
    }
}

impl<R: Ring> Mul<R> for DenseMultilinearExtension<R> {
    type Output = DenseMultilinearExtension<R>;

    fn mul(mut self, rhs: R) -> DenseMultilinearExtension<R> {
        self.evaluations.iter_mut().for_each(|x| *x *= rhs);

        self
    }
}

impl<R: Ring> MulAssign<R> for DenseMultilinearExtension<R> {
    fn mul_assign(&mut self, rhs: R) {
        self.evaluations.iter_mut().for_each(|x| *x *= rhs);
    }
}

impl<R: Ring> Sub<R> for DenseMultilinearExtension<R> {
    type Output = DenseMultilinearExtension<R>;

    fn sub(mut self, rhs: R) -> DenseMultilinearExtension<R> {
        self.evaluations.iter_mut().for_each(|x| *x -= rhs);

        self
    }
}

impl<R: Ring> Add<R> for DenseMultilinearExtension<R> {
    type Output = DenseMultilinearExtension<R>;

    fn add(mut self, rhs: R) -> DenseMultilinearExtension<R> {
        self.evaluations.iter_mut().for_each(|x| *x += rhs);

        self
    }
}

impl<R: Ring> Index<usize> for DenseMultilinearExtension<R> {
    type Output = R;

    fn index(&self, index: usize) -> &Self::Output {
        &self.evaluations[index]
    }
}

impl<R: Ring> IndexMut<usize> for DenseMultilinearExtension<R> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.evaluations[index]
    }
}
