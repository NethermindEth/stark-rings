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

use crate::{PolyRing, Ring};
use convertible_ring::ConvertibleRing;
use lattirust_linear_algebra::{
    ops::{rounded_div, Transpose},
    Matrix, RowVector, SymmetricMatrix, Vector,
};
pub mod convertible_ring;
mod fq_convertible;
pub(crate) mod representatives;

pub trait Decompose: Sized + Zero + Copy {
    fn decompose_in_place(&self, b: u128, out: &mut [Self]);
    fn decompose(&self, b: u128, padding_size: usize) -> Vec<Self> {
        let mut res = vec![Self::zero(); padding_size];

        self.decompose_in_place(b, &mut res);

        res
    }
}

/// Returns the balanced decomposition of a slice as a Vec of Vecs.
///
/// # Arguments
/// * `v`: input element
/// * `b`: basis for the decomposition, must be even
/// * `padding_size`: indicates whether the output should be padded with zeros to a specified length `k` if `padding_size` is `Some(k)`, or if it should be padded to the largest decomposition length required for `v` if `padding_size` is `None`
///
/// # Output
/// Returns `d`, the decomposition in basis `b` as a Vec of size `decomp_size`, i.e.,
/// $\texttt{v}\[i\] = \sum_{j \in \[k\]} \texttt{b}^j \texttt{d}\[j\]$ and $|\texttt{d}\[j\]| \leq \left\lfloor\frac{\texttt{b}}{2}\right\rfloor$.
pub fn decompose_balanced_in_place<R: ConvertibleRing>(v: &R, b: u128, out: &mut [R]) {
    assert!(
        !b.is_zero() && !b.is_one(),
        "cannot decompose in basis 0 or 1"
    );
    // TODO: not sure if this really necessary, but having b be even allow for more efficient divisions/remainders
    assert_eq!(b % 2, 0, "decomposition basis must be even");

    let mut curr = Into::<R::SignedInt>::into(*v);
    let mut current_i = 0;
    let b = b as i128;
    let b_half_floor = b.div_euclid(2);
    loop {
        let rem = curr.clone() % b; // rem = curr % b is in [-(b-1), (b-1)]

        // Ensure digit is in [-b/2, b/2]
        if rem.abs() <= b_half_floor.into() {
            out[current_i] = Into::<R>::into(rem);
            curr /= b; // Rust integer division rounds towards zero
        } else {
            // The next element in the decomposition is sign(rem) * (|rem| - b)
            if rem < 0.into() {
                out[current_i] = Into::<R>::into(rem.clone() + b);
            } else {
                out[current_i] = Into::<R>::into(rem.clone() - b);
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
        *out_tail_elem = R::zero();
    }
}

pub fn decompose_balanced<R: ConvertibleRing>(v: &R, b: u128, padding_size: usize) -> Vec<R> {
    let mut out = vec![R::zero(); padding_size];

    decompose_balanced_in_place(v, b, &mut out);

    out
}

pub fn gadget_decompose<R: Decompose + Sync + Send>(
    v_s: &[R],
    b: u128,
    // Padding size is the maximum power of the gadget matrix
    padding_size: usize,
) -> Vec<R> {
    let mut out = vec![R::zero(); padding_size * v_s.len()];

    cfg_chunks_mut!(&mut out, padding_size)
        .zip(cfg_iter!(v_s))
        .for_each(|(chunk_mut, v)| v.decompose_in_place(b, chunk_mut));

    out
}

pub fn gadget_recompose<A, B>(
    v_s: &[A],
    b: B,
    // Padding size is the maximum power of the gadget matrix
    padding_size: usize,
) -> Vec<A>
where
    A: Zero + Copy + for<'a> MulAssign<&'a B> + for<'a> AddAssign<&'a A> + Sync + Send,
    B: Copy + Sync + Send,
{
    let mut out = vec![A::zero(); v_s.len() / padding_size];

    cfg_chunks!(v_s, padding_size)
        .zip(cfg_iter_mut!(out))
        .for_each(|(chunk, v)| *v = recompose(chunk, b));

    out
}

/// Returns the balanced decomposition of a slice as a Vec of Vecs.
///
/// # Arguments
/// * `v`: input slice, of length `l`
/// * `b`: basis for the decomposition, must be even
/// * `padding_size`: indicates whether the output should be padded with zeros to a specified length `k` if `padding_size` is `Some(k)`, or if it should be padded to the largest decomposition length required for `v` if `padding_size` is `None`
///
/// # Output
/// Returns `d` the decomposition in basis `b` as a Vec of size `decomp_size`, with each item being a Vec of length `l`, i.e.,
/// for all $i \in \[l\]: \texttt{v}\[i\] = \sum_{j \in \[k\]} \texttt{b}^j \texttt{d}\[i\]\[j\]$ and $|\texttt{d}\[i\]\[j\]| \leq \left\lfloor\frac{\texttt{b}}{2}\right\rfloor$.
pub fn decompose_balanced_vec<D: Ring + Decompose>(
    v: &[D],
    b: u128,
    padding_size: usize,
) -> Vec<Vec<D>> {
    cfg_iter!(v)
        .map(|v_i| v_i.decompose(b, padding_size))
        .collect()
}

/// Returns the balanced decomposition of a [`PolyRing`] element as a Vec of [`PolyRing`] elements.
///
/// # Arguments
/// * `v`: `PolyRing` element to be decomposed
/// * `b`: basis for the decomposition, must be even
/// * `padding_size`: indicates whether the output should be padded with zeros to a specified length `k` if `padding_size` is `Some(k)`, or if it should be padded to the largest decomposition length required for `v` if `padding_size` is `None`
///
/// # Output
/// Returns `d` the decomposition in basis `b` as a Vec of size `decomp_size`, i.e.,
/// for all $\texttt{v} = \sum_{j \in \[k\]} \texttt{b}^j \texttt{d}\[j\]$ and $|\texttt{d}\[j\]| \leq \left\lfloor\frac{\texttt{b}}{2}\right\rfloor$.
pub fn decompose_balanced_polyring<R: PolyRing>(v: &R, b: u128, padding_size: usize) -> Vec<R>
where
    R::BaseRing: Decompose,
{
    decompose_balanced_vec::<R::BaseRing>(v.coeffs(), b, padding_size)
        .transpose()
        .into_iter()
        .map(|v_i| R::from(v_i))
        .collect()
}

pub fn decompose_balanced_slice_polyring<R: PolyRing>(
    v: &[R],
    b: u128,
    padding_size: usize,
) -> Vec<Vec<R>>
where
    R::BaseRing: Decompose,
{
    cfg_iter!(v)
        .map(|ring_elem| decompose_balanced_polyring(ring_elem, b, padding_size))
        .collect()
}

/// Returns the balanced decomposition of a slice of [`PolyRing`] elements as a Vec of [`Vector`] of [`PolyRing`] elements.
///
/// # Arguments
/// * `v`: input slice, of length `l`
/// * `b`: basis for the decomposition, must be even
/// * `padding_size`: indicates whether the output should be padded with zeros to a specified length `k` if `padding_size` is `Some(k)`, or if it should be padded to the largest decomposition length required for `v` if `padding_size` is `None`
///
/// # Output
/// Returns `d` the decomposition in basis `b` as a Vec of size `decomp_size`, with each item being a Vec of length `l`, i.e.,
/// for all $i \in \[l\]: \texttt{v}\[i\] = \sum_{j \in \[k\]} \texttt{b}^j \texttt{d}\[i\]\[j\]$ and $|\texttt{d}\[i\]\[j\]| \leq \left\lfloor\frac{\texttt{b}}{2}\right\rfloor$.
pub fn decompose_balanced_vec_polyring<R: PolyRing>(
    v: &Vector<R>,
    b: u128,
    padding_size: usize,
) -> Vec<Vector<R>>
where
    R::BaseRing: Decompose,
{
    cfg_iter!(v.as_slice())
        .map(|ring_elem| decompose_balanced_polyring(ring_elem, b, padding_size))
        .map(Vector::from)
        .collect()
}

/// Returns the balanced gadget decomposition of a [`Matrix`] of dimensions `m × n` as a matrix of dimensions `m × (k * n)`.
///
/// # Arguments
/// * `mat`: input matrix of dimensions `m × n`
/// * `b`: basis for the decomposition, must be even
/// * `padding_size`: indicates whether the decomposition length is the specified `k` if `padding_size` is `Some(k)`, or if `k` is the largest decomposition length required for `mat` if `padding_size` is `None`
///
/// # Output
/// Returns `d` the decomposition in basis `b` as a Matrix of dimensions `m × (k * n)`, i.e.,
/// $\texttt{mat} = \texttt{d} \times G_\texttt{n}$ where $G_\texttt{n} = I_\texttt{n} \otimes (1, \texttt{b}, \ldots, \texttt{b}^k) \in R^{\texttt{k}\texttt{n} \times \texttt{n}}$ and $|\texttt{d}\[i\]\[j\]| \leq \left\lfloor\frac{\texttt{b}}{2}\right\rfloor$.
pub fn decompose_matrix<F: ConvertibleRing>(
    mat: &Matrix<F>,
    decomposition_basis: u128,
    decomposition_length: usize,
) -> Matrix<F> {
    let row_iter = {
        #[cfg(not(feature = "parallel"))]
        {
            mat.row_iter()
        }
        #[cfg(feature = "parallel")]
        {
            mat.par_row_iter()
        }
    };
    Matrix::<F>::from_rows(
        row_iter
            .map(|s_i| {
                RowVector::<F>::from(
                    s_i.map(|s_ij| {
                        decompose_balanced(&s_ij, decomposition_basis, decomposition_length)
                    })
                    .as_slice()
                    .concat(),
                )
            })
            .collect::<Vec<_>>()
            .as_slice(),
    )
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

/// Given a `n*d x n*d` symmetric matrix `mat` and a slice `\[1, b, ..., b^(d-1)\]` `powers_of_basis`, returns the `n x n` symmetric matrix corresponding to $G^T \textt{mat} G$, where $G = I_n \otimes (1, b, ..., b^(\textt{d}-1))$ is the gadget matrix of dimensions `n*d x n`.
pub fn recompose_left_right_symmetric_matrix<F: Clone + Sum + Send + Sync>(
    mat: &SymmetricMatrix<F>,
    powers_of_basis: &[F],
) -> SymmetricMatrix<F>
where
    for<'a> &'a F: Mul<&'a F, Output = F>,
{
    let (nd, d) = (mat.size(), powers_of_basis.len());
    assert_eq!(nd % d, 0);

    let n = nd / d;
    cfg_into_iter!(0..n)
        .map(|i| {
            (0..=i)
                .map(|j| {
                    (0..nd)
                        .filter(|k| k / d == i)
                        .flat_map(|k| {
                            (0..nd).filter(|l| l / d == j).map(move |l| {
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
    use crate::cyclotomic_ring::models::goldilocks::{Fq, RqPoly};

    use crate::PolyRing;
    use crate::SignedRepresentative;

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
                let decomp = decompose_balanced(v, b, 32);

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
            let decomp = decompose_balanced_vec(&v, b, 16).transpose();

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
            let decomp = decompose_balanced_polyring(&v, b, 16);

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

        let decomposed = gadget_decompose(&vec, 2, 4);
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

        assert_eq!(gadget_recompose(&decomposed, 2u128, 4), expected);
    }

    // TODO: to work we need the unsigned order on the field.
    // #[test]
    // fn test_decompose_balanced_vec_polyring() {
    //     let v = Vector::<PolyR>::from_fn(VEC_LENGTH, |i, _| {
    //         PolyR::from(
    //             (0..(D as u64))
    //                 .step_by((Q / (D as u64)) as usize)
    //                 .map(|x| R::from(x + i as u64))
    //                 .collect::<Vec<_>>(),
    //         )
    //     });
    //     for b in BASIS_TEST_RANGE {
    //         let b_half = R::from(b / 2);
    //         let decomp = decompose_balanced_vec_polyring::<PolyR>(&v, b, None);

    //         println!("{:?}", v);
    //         for v_i in &decomp {
    //             for v_ij in v_i.as_slice() {
    //                 for v_ijk in v_ij.coeffs() {
    //                     println!("{:?} {:?} {:?}", b, b_half, v_ijk);
    //                     assert!(v_ijk <= b_half || v_ijk >= -b_half);
    //                 }
    //             }
    //         }

    //         let mut recomposed = Vector::<PolyR>::zeros(v.len());
    //         for (i, v_i) in decomp.iter().enumerate() {
    //             recomposed += v_i * PolyR::from_scalar(Ring::pow(&R::from(b), [i as u64]));
    //         }
    //         assert_eq!(v, recomposed);
    //     }
    // }

    // #[test]
    // pub fn test_decompose_balanced_matrix() {
    //     const NROWS: usize = 101;
    //     const NCOLS: usize = 42;

    //     let rng = &mut test_rng();
    //     let mat = Matrix::<R>::rand(NROWS, NCOLS, rng);

    //     for b in BASIS_TEST_RANGE {
    //         let b_half = b.div_floor(2);

    //         let decomp_size = balanced_decomposition_max_length(b, (Q - 1) as u128);
    //         let decomp = decompose_matrix(&mat, b, decomp_size);
    //         assert_eq!(decomp.nrows(), mat.nrows());
    //         assert_eq!(decomp.ncols(), mat.ncols() * decomp_size);
    //         for d_ij in decomp.iter() {
    //             assert!(Into::<SignedRepresentative>::into(*d_ij).0.abs() <= b_half as i128);
    //         }

    //         let pows_b = (0..decomp_size)
    //             .map(|i| R::from(b.pow(i as u32)))
    //             .collect::<Vec<_>>();
    //         let recomp = recompose_matrix(&decomp, &pows_b);

    //         assert_eq!(mat, recomp);

    //         // Check that recompose is equivalent to right-multiplication by the gadget matrix
    //         let gadget_vector = Vector::<R>::from_slice(&pows_b);
    //         let gadget_matrix = Matrix::<R>::identity(NCOLS, NCOLS).kronecker(&gadget_vector);
    //         let mat_g = &decomp * &gadget_matrix;
    //         assert_eq!(recomp, mat_g);

    //         // Check that A * M == recompose(A * decompose(M))
    //         let mat_l = Matrix::<R>::rand(64, NROWS, rng);
    //         let lhs = &mat_l * &mat;
    //         let rhs = recompose_matrix(&(&mat_l * &decomp), &pows_b);
    //         assert_eq!(lhs, rhs);
    //     }
    // }

    // #[test]
    // fn test_left_right_decompose_symmetric_matrix() {
    //     const SIZE: usize = 13;
    //     let rng = &mut test_rng();

    //     for b in BASIS_TEST_RANGE {
    //         let decomp_size = balanced_decomposition_max_length(b, (Q - 1) as u128);

    //         let pows_b = (0..decomp_size)
    //             .map(|i| R::from(b.pow(i as u32)))
    //             .collect::<Vec<_>>();
    //         let mat = SymmetricMatrix::<R>::rand(SIZE * decomp_size, rng);

    //         let expected: SymmetricMatrix<R> = recompose_matrix(
    //             &recompose_matrix(&mat.clone().into(), &pows_b.as_slice()).transpose(),
    //             &pows_b.as_slice(),
    //         )
    //         .into();

    //         let actual = recompose_left_right_symmetric_matrix(&mat, &pows_b.as_slice());
    //         assert_eq!(expected, actual);
    //     }
    // }
}
