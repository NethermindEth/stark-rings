use ark_std::{
    cfg_chunks, cfg_chunks_mut, cfg_into_iter, cfg_iter, cfg_iter_mut,
    iter::Sum,
    ops::{AddAssign, Mul, MulAssign},
    vec::*,
    One, Zero,
};
use num_traits::Signed;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::Ring;
use convertible_ring::ConvertibleRing;
use stark_rings_linalg::{ops::rounded_div, Matrix, SparseMatrix, SymmetricMatrix};
pub mod convertible_ring;
mod fq_convertible;
pub(crate) mod representatives;

/// Ring decomposition
pub trait Decompose: Ring {
    fn decompose_to(&self, b: u128, out: &mut [Self]);
    fn decompose(&self, b: u128, padding_size: usize) -> Vec<Self> {
        let mut out = vec![Self::zero(); padding_size];
        self.decompose_to(b, &mut out);
        out
    }
}

/// Decompose to split vectors
pub trait DecomposeToVec {
    type Element;
    fn decompose_to_vec(&self, b: u128, padding_size: usize) -> Vec<Self::Element>;
}

/// Decompose, creating an extended version
pub trait GadgetDecompose {
    type Output;
    fn gadget_decompose(&self, b: u128, padding_size: usize) -> Self::Output;
}

/// Recompose
pub trait GadgetRecompose {
    type Output;
    fn gadget_recompose(&self, b: u128, padding_size: usize) -> Self::Output;
}

/// Returns the balanced decomposition of a slice as a Vec of Vecs.
///
/// # Arguments
/// * `v`: input element
/// * `b`: basis for the decomposition, must be even
/// * `padding_size`: indicates whether the output should be padded with zeros
///   to a specified length `k` if `padding_size` is `Some(k)`, or if it should
///   be padded to the largest decomposition length required for `v` if
///   `padding_size` is `None`
///
/// # Output
/// Returns `d`, the decomposition in basis `b` as a Vec of size `decomp_size`,
/// i.e., $\texttt{v}\[i\] = \sum_{j \in \[k\]} \texttt{b}^j \texttt{d}\[j\]$
/// and $|\texttt{d}\[j\]| \leq \left\lfloor\frac{\texttt{b}}{2}\right\rfloor$.
pub fn decompose_balanced_in_place<Z: ConvertibleRing>(v: &Z, b: u128, out: &mut [Z]) {
    assert!(
        !b.is_zero() && !b.is_one(),
        "cannot decompose in basis 0 or 1"
    );
    // TODO: not sure if this really necessary, but having b be even allow for more
    // efficient divisions/remainders
    assert_eq!(b % 2, 0, "decomposition basis must be even");

    let mut curr = Into::<Z::SignedInt>::into(*v);
    let mut current_i = 0;
    let b = b as i128;
    let b_half_floor = b.div_euclid(2);
    loop {
        let rem = curr.clone() % b; // rem = curr % b is in [-(b-1), (b-1)]

        // Ensure digit is in [-b/2, b/2]
        if rem.abs() <= b_half_floor.into() {
            out[current_i] = Into::<Z>::into(rem);
            curr /= b; // Rust integer division rounds towards zero
        } else {
            // The next element in the decomposition is sign(rem) * (|rem| - b)
            if rem < 0.into() {
                out[current_i] = Into::<Z>::into(rem.clone() + b);
            } else {
                out[current_i] = Into::<Z>::into(rem.clone() - b);
            }
            let carry = rounded_div(rem, b); // Round toward nearest integer, not towards 0
            curr = (curr / b) + carry;
        }

        current_i += 1;

        if curr.is_zero() {
            break;
        }
    }

    for out_tail_elem in out[current_i..].iter_mut() {
        *out_tail_elem = Z::zero();
    }
}

pub fn recompose<A, B>(v: &[A], b: B) -> A
where
    A: Zero + for<'a> MulAssign<&'a B> + for<'a> AddAssign<&'a A>,
{
    let mut result = A::zero();

    for v_i in v.iter().rev() {
        result *= &b;
        result += v_i;
    }

    result
}

impl<R: Decompose> DecomposeToVec for &[R] {
    type Element = Vec<R>;

    /// Returns the balanced decomposition of a slice as a Vec of Vecs.
    ///
    /// # Arguments
    /// * `v`: input slice, of length `l`
    /// * `b`: basis for the decomposition, must be even
    /// * `padding_size`: indicates whether the output should be padded with
    ///   zeros to a specified length `k` if `padding_size` is `Some(k)`, or if
    ///   it should be padded to the largest decomposition length required for
    ///   `v` if `padding_size` is `None`
    ///
    /// # Output
    /// Returns `d` the decomposition in basis `b` as a Vec of size
    /// `decomp_size`, with each item being a Vec of length `l`, i.e.,
    fn decompose_to_vec(&self, b: u128, padding_size: usize) -> Vec<Vec<R>> {
        cfg_iter!(self)
            .map(|v_i| v_i.decompose(b, padding_size))
            .collect()
    }
}

impl<R: Decompose> DecomposeToVec for Vec<R> {
    type Element = Vec<R>;

    /// Returns the balanced decomposition of a slice as a Vec of Vecs.
    ///
    /// # Arguments
    /// * `v`: input slice, of length `l`
    /// * `b`: basis for the decomposition, must be even
    /// * `padding_size`: indicates whether the output should be padded with
    ///   zeros to a specified length `k` if `padding_size` is `Some(k)`, or if
    ///   it should be padded to the largest decomposition length required for
    ///   `v` if `padding_size` is `None`
    ///
    /// # Output
    /// Returns `d` the decomposition in basis `b` as a Vec of size
    /// `decomp_size`, with each item being a Vec of length `l`, i.e.,
    fn decompose_to_vec(&self, b: u128, padding_size: usize) -> Vec<Vec<R>> {
        self.as_slice().decompose_to_vec(b, padding_size)
    }
}

impl<R: Decompose> GadgetDecompose for &[R] {
    type Output = Vec<R>;

    fn gadget_decompose(&self, b: u128, padding_size: usize) -> Vec<R> {
        let mut out = vec![R::zero(); padding_size * self.len()];

        cfg_chunks_mut!(&mut out, padding_size)
            .zip(cfg_iter!(self))
            .for_each(|(chunk_mut, v)| v.decompose_to(b, chunk_mut));

        out
    }
}

impl<R: Ring> GadgetRecompose for &[R] {
    type Output = Vec<R>;

    fn gadget_recompose(&self, b: u128, padding_size: usize) -> Vec<R> {
        let mut out = vec![R::zero(); self.len() / padding_size];
        let b = R::from(b);

        cfg_chunks!(self, padding_size)
            .zip(cfg_iter_mut!(out))
            .for_each(|(chunk, v)| *v = recompose(chunk, b));

        out
    }
}

impl<R: Decompose> GadgetDecompose for Vec<R> {
    type Output = Vec<R>;

    fn gadget_decompose(&self, b: u128, padding_size: usize) -> Vec<R> {
        self.as_slice().gadget_decompose(b, padding_size)
    }
}

impl<R: Ring> GadgetRecompose for Vec<R> {
    type Output = Vec<R>;

    fn gadget_recompose(&self, b: u128, padding_size: usize) -> Vec<R> {
        self.as_slice().gadget_recompose(b, padding_size)
    }
}

impl<R: Decompose> GadgetDecompose for &[(R, usize)] {
    type Output = Vec<(R, usize)>;

    fn gadget_decompose(&self, b: u128, padding_size: usize) -> Vec<(R, usize)> {
        let mut out = vec![(R::zero(), 0usize); self.len() * padding_size];

        cfg_chunks_mut!(out, padding_size)
            .zip(cfg_iter!(self))
            .for_each(|(chunk, (r, index))| {
                let decomposed = r.decompose(b, padding_size);

                for (i, &r) in decomposed.iter().enumerate() {
                    let new_index = index * padding_size + i;
                    chunk[i] = (r, new_index);
                }
            });

        // Maintain full sparsity
        out.retain(|&(r, _)| r != R::zero());

        out
    }
}

impl<R: Ring> GadgetRecompose for &[(R, usize)] {
    type Output = Vec<(R, usize)>;

    fn gadget_recompose(&self, b: u128, padding_size: usize) -> Vec<(R, usize)> {
        let mut chunks: Vec<&[(R, usize)]> = Vec::with_capacity(self.len() / padding_size);
        let mut chunk_start = 0;

        for i in 1..self.len() {
            let index = self[i].1 / padding_size;

            let prev_index = self[i - 1].1 / padding_size;
            if index != prev_index {
                chunks.push(&self[chunk_start..i]);
                chunk_start = i;
            }
        }

        if chunk_start < self.len() {
            chunks.push(&self[chunk_start..]);
        }

        let b = R::from(b);
        cfg_iter!(chunks)
            .map(|chunk| {
                let index = chunk[0].1 / padding_size;
                let mut decomposed = vec![R::zero(); padding_size];
                chunk
                    .iter()
                    .for_each(|(elem, i)| decomposed[i % padding_size] = *elem);
                let recomposed = recompose(&decomposed, b);
                (recomposed, index)
            })
            .collect()
    }
}

impl<R: Ring> GadgetRecompose for Vec<(R, usize)> {
    type Output = Vec<(R, usize)>;

    fn gadget_recompose(&self, b: u128, padding_size: usize) -> Vec<(R, usize)> {
        self.as_slice().gadget_recompose(b, padding_size)
    }
}

impl<R: Decompose> GadgetDecompose for Matrix<R> {
    type Output = Matrix<R>;

    /// Returns the balanced gadget decomposition of a [`Matrix`] of dimensions
    /// `n × m` as a matrix of dimensions `n × (k × m)`.
    ///
    /// # Arguments
    /// * `mat`: input matrix of dimensions `n × m`
    /// * `b`: basis for the decomposition, must be even
    /// * `padding_size`: indicates whether the decomposition length is the
    ///   specified `k` if `padding_size` is `Some(k)`, or if `k` is the largest
    ///   decomposition length required for `mat` if `padding_size` is `None`
    ///
    /// # Output
    /// Returns `d` the decomposition in basis `b` as a Matrix of dimensions `n
    /// × (k × m)`, i.e.,
    fn gadget_decompose(&self, b: u128, padding_size: usize) -> Matrix<R> {
        cfg_iter!(self.vals)
            .map(|row| row.as_slice().gadget_decompose(b, padding_size))
            .collect::<Vec<_>>()
            .into()
    }
}

impl<R: Ring> GadgetRecompose for Matrix<R> {
    type Output = Matrix<R>;

    fn gadget_recompose(&self, b: u128, padding_size: usize) -> Matrix<R> {
        cfg_iter!(self.vals)
            .map(|row| row.as_slice().gadget_recompose(b, padding_size))
            .collect::<Vec<_>>()
            .into()
    }
}

impl<R: Decompose> GadgetDecompose for SparseMatrix<R> {
    type Output = SparseMatrix<R>;

    /// Returns the balanced gadget decomposition of a [`SparseMatrix`] of
    /// dimensions `n × m` as a matrix of dimensions `n × (k × m)`.
    ///
    /// # Arguments
    /// * `mat`: input matrix of dimensions `n × m`
    /// * `b`: basis for the decomposition, must be even
    /// * `padding_size`: indicates whether the decomposition length is the
    ///   specified `k` if `padding_size` is `Some(k)`, or if `k` is the largest
    ///   decomposition length required for `mat` if `padding_size` is `None`
    ///
    /// # Output
    /// Returns `d` the decomposition in basis `b` as a Matrix of dimensions `n
    /// × (k × m)`, i.e.,
    fn gadget_decompose(&self, b: u128, padding_size: usize) -> SparseMatrix<R> {
        let coeffs = cfg_iter!(self.coeffs)
            .map(|row| row.as_slice().gadget_decompose(b, padding_size))
            .collect::<Vec<_>>();
        SparseMatrix {
            ncols: self.ncols * padding_size,
            nrows: self.nrows,
            coeffs,
        }
    }
}

impl<R: Ring> GadgetRecompose for SparseMatrix<R> {
    type Output = SparseMatrix<R>;

    fn gadget_recompose(&self, b: u128, padding_size: usize) -> SparseMatrix<R> {
        let coeffs = cfg_iter!(self.coeffs)
            .map(|row| row.as_slice().gadget_recompose(b, padding_size))
            .collect::<Vec<_>>();
        SparseMatrix {
            ncols: self.ncols / padding_size,
            nrows: self.nrows,
            coeffs,
        }
    }
}

/// Given a `n*d x n*d` symmetric matrix `mat` and a slice `\[1, b, ...,
/// b^(d-1)\]` `powers_of_basis`, returns the `n x n` symmetric matrix
/// corresponding to $G^T \textt{mat} G$, where $G = I_n \otimes (1, b, ...,
/// b^(\textt{d}-1))$ is the gadget matrix of dimensions `n*d x n`.
pub fn recompose_left_right_symmetric_matrix<F: Clone + Sum + Send + Sync>(
    mat: &SymmetricMatrix<F>,
    powers_of_basis: &[F],
) -> SymmetricMatrix<F>
where
    for<'a> &'a F: Mul<&'a F, Output = F>,
{
    let (and, d) = (mat.size(), powers_of_basis.len());
    assert_eq!(and % d, 0);

    let n = and / d;
    cfg_into_iter!(0..n)
        .map(|i| {
            (0..=i)
                .map(|j| {
                    (0..and)
                        .filter(|k| k / d == i)
                        .flat_map(|k| {
                            (0..and).filter(|l| l / d == j).map(move |l| {
                                &mat[(k, l)] * &(&powers_of_basis[k % d] * &powers_of_basis[l % d])
                            })
                        })
                        .sum()
                })
                .collect()
        })
        .collect::<Vec<_>>()
        .into()
}

#[cfg(test)]
mod tests {
    use crate::{
        cyclotomic_ring::models::goldilocks::{Fq, RqPoly},
        PolyRing, SignedRepresentative,
    };
    use stark_rings_linalg::ops::Transpose;

    use super::*;

    const D: usize = 128;
    const Q: u64 = 65537;
    const BASIS_TEST_RANGE: [u128; 5] = [2, 4, 8, 16, 32];

    type R = Fq;
    type PolyR = RqPoly;

    #[test]
    fn test_decompose_balanced() {
        let vs: Vec<R> = (0..Q).map(R::from).collect();
        for b in BASIS_TEST_RANGE {
            let b_half = R::from(b / 2);
            for v in &vs {
                let decomp = v.decompose(b, 32);

                // Check that all entries are smaller than b/2
                for v_i in &decomp {
                    assert!(*v_i <= b_half || *v_i >= -b_half);
                }

                // Check that the decomposition is correct
                assert_eq!(*v, recompose(&decomp, R::from(b)));
            }
        }
    }

    fn get_test_vec() -> Vec<R> {
        (0..(D as u64)).map(R::from).collect()
    }

    #[test]
    fn test_decompose_balanced_vec() {
        let v = get_test_vec();
        for b in BASIS_TEST_RANGE {
            let b_half = b / 2;
            let decomp = v.decompose_to_vec(b, 16).transpose();

            // Check that all entries are smaller than b/2 in absolute value
            for d_i in &decomp {
                for d_ij in d_i {
                    let s_ij: i128 = SignedRepresentative::from(*d_ij).0;
                    assert!(s_ij.unsigned_abs() <= b_half);
                }
            }

            for i in 0..decomp.len() {
                // Check that the decomposition is correct
                let decomp_i = decomp.iter().map(|d_j| d_j[i]).collect::<Vec<_>>();
                assert_eq!(v[i], recompose(&decomp_i, R::from(b)));
            }
        }
    }

    #[test]
    fn test_decompose_balanced_polyring() {
        let v: PolyR = PolyR::from(get_test_vec());
        for b in BASIS_TEST_RANGE {
            let b_half = b / 2;
            let decomp = v.decompose(b, 16);

            for d_i in &decomp {
                for &d_ij in d_i.coeffs() {
                    let s_ij: i128 = SignedRepresentative::from(d_ij).0;
                    assert!(s_ij.unsigned_abs() <= b_half);
                }
            }

            assert_eq!(v, recompose::<PolyR, u128>(&decomp, b));
        }
    }

    #[test]
    fn test_gadget_decompose() {
        use crate::cyclotomic_ring::models::goldilocks::{Fq, RqPoly};

        let vec: Vec<RqPoly> = vec![
            RqPoly::from((0..24).map(|_| Fq::from(15)).collect::<Vec<Fq>>()),
            RqPoly::from((0..24).map(|_| Fq::from(-15)).collect::<Vec<Fq>>()),
        ];

        let decomposed = vec.gadget_decompose(2, 4);
        let expected = vec![
            RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
            RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
            RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
            RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
            RqPoly::from((0..24).map(|_| Fq::from(-1)).collect::<Vec<Fq>>()),
            RqPoly::from((0..24).map(|_| Fq::from(-1)).collect::<Vec<Fq>>()),
            RqPoly::from((0..24).map(|_| Fq::from(-1)).collect::<Vec<Fq>>()),
            RqPoly::from((0..24).map(|_| Fq::from(-1)).collect::<Vec<Fq>>()),
        ];

        assert_eq!(decomposed, expected);
    }

    #[test]
    fn test_gadget_recompose() {
        use crate::cyclotomic_ring::models::goldilocks::{Fq, RqPoly};

        let expected: Vec<RqPoly> = vec![
            RqPoly::from((0..24).map(|_| Fq::from(15)).collect::<Vec<Fq>>()),
            RqPoly::from((0..24).map(|_| Fq::from(-15)).collect::<Vec<Fq>>()),
        ];

        let decomposed = vec![
            RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
            RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
            RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
            RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
            RqPoly::from((0..24).map(|_| Fq::from(-1)).collect::<Vec<Fq>>()),
            RqPoly::from((0..24).map(|_| Fq::from(-1)).collect::<Vec<Fq>>()),
            RqPoly::from((0..24).map(|_| Fq::from(-1)).collect::<Vec<Fq>>()),
            RqPoly::from((0..24).map(|_| Fq::from(-1)).collect::<Vec<Fq>>()),
        ];

        assert_eq!(decomposed.gadget_recompose(2u128, 4), expected);
    }

    #[test]
    fn test_matrix_gadget_decompose() {
        use crate::cyclotomic_ring::models::goldilocks::{Fq, RqPoly};

        let m: Matrix<RqPoly> = vec![
            vec![
                RqPoly::from((0..24).map(|_| Fq::from(15)).collect::<Vec<Fq>>()),
                RqPoly::from((0..24).map(|_| Fq::from(-15)).collect::<Vec<Fq>>()),
            ],
            vec![
                RqPoly::from((0..24).map(|_| Fq::from(15)).collect::<Vec<Fq>>()),
                RqPoly::from((0..24).map(|_| Fq::from(-15)).collect::<Vec<Fq>>()),
            ],
        ]
        .into();

        let decomposed = m.gadget_decompose(2, 4);
        let expected = vec![
            vec![
                RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
                RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
                RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
                RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
                RqPoly::from((0..24).map(|_| Fq::from(-1)).collect::<Vec<Fq>>()),
                RqPoly::from((0..24).map(|_| Fq::from(-1)).collect::<Vec<Fq>>()),
                RqPoly::from((0..24).map(|_| Fq::from(-1)).collect::<Vec<Fq>>()),
                RqPoly::from((0..24).map(|_| Fq::from(-1)).collect::<Vec<Fq>>()),
            ],
            vec![
                RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
                RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
                RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
                RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
                RqPoly::from((0..24).map(|_| Fq::from(-1)).collect::<Vec<Fq>>()),
                RqPoly::from((0..24).map(|_| Fq::from(-1)).collect::<Vec<Fq>>()),
                RqPoly::from((0..24).map(|_| Fq::from(-1)).collect::<Vec<Fq>>()),
                RqPoly::from((0..24).map(|_| Fq::from(-1)).collect::<Vec<Fq>>()),
            ],
        ];

        assert_eq!(decomposed, expected.into());
    }

    #[test]
    fn test_matrix_gadget_recompose() {
        use crate::cyclotomic_ring::models::goldilocks::{Fq, RqPoly};

        let m: Matrix<RqPoly> = vec![
            vec![
                RqPoly::from((0..24).map(|_| Fq::from(15)).collect::<Vec<Fq>>()),
                RqPoly::from((0..24).map(|_| Fq::from(-15)).collect::<Vec<Fq>>()),
            ],
            vec![
                RqPoly::from((0..24).map(|_| Fq::from(15)).collect::<Vec<Fq>>()),
                RqPoly::from((0..24).map(|_| Fq::from(-15)).collect::<Vec<Fq>>()),
            ],
        ]
        .into();

        let decomposed = m.gadget_decompose(2, 4);
        let recomposed = decomposed.gadget_recompose(2, 4);

        assert_eq!(recomposed, m);
    }

    #[test]
    fn test_sparse_matrix_gadget_decompose() {
        use crate::cyclotomic_ring::models::goldilocks::{Fq, RqPoly};

        let coeffs = vec![
            vec![(
                RqPoly::from((0..24).map(|_| Fq::from(13)).collect::<Vec<Fq>>()),
                1,
            )],
            vec![],
        ];

        let m = SparseMatrix {
            coeffs,
            nrows: 2,
            ncols: 2,
        };

        let decomposed = m.gadget_decompose(2, 4);

        let coeffs = vec![
            vec![
                (
                    RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
                    4,
                ),
                (
                    RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
                    6,
                ),
                (
                    RqPoly::from((0..24).map(|_| Fq::from(1)).collect::<Vec<Fq>>()),
                    7,
                ),
            ],
            vec![],
        ];
        let expected = SparseMatrix {
            coeffs,
            nrows: 2,
            ncols: 2 * 4,
        };

        assert_eq!(decomposed, expected);
    }

    #[test]
    fn test_sparse_matrix_gadget_recompose() {
        use crate::cyclotomic_ring::models::goldilocks::{Fq, RqPoly};

        let coeffs = vec![
            vec![(
                RqPoly::from((0..24).map(|_| Fq::from(13)).collect::<Vec<Fq>>()),
                1,
            )],
            vec![],
        ];

        let m = SparseMatrix {
            coeffs,
            nrows: 2,
            ncols: 2,
        };

        let decomposed = m.gadget_decompose(2, 4);
        let recomposed = decomposed.gadget_recompose(2, 4);

        assert_eq!(recomposed, m);
    }
}
