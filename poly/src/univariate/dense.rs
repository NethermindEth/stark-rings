use ark_ff::Zero;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    fmt,
    ops::{Add, AddAssign, Deref, DerefMut, Mul, Neg, Sub, SubAssign},
    rand::Rng,
    vec::*,
};
use stark_rings::Ring;

use super::{DenseUVPolynomial, Polynomial};

/// Stores a polynomial in coefficient form.
#[derive(Clone, PartialEq, Eq, Hash, Default, CanonicalSerialize, CanonicalDeserialize)]
pub struct DenseUnivariatePolynomial<Rn: Ring> {
    /// The coefficient of `x^i` is stored at location `i` in `self.coeffs`.
    pub coeffs: Vec<Rn>,
}

impl<Rn: Ring> Polynomial<Rn> for DenseUnivariatePolynomial<Rn> {
    type Point = Rn;

    /// Returns the total degree of the polynomial
    fn degree(&self) -> usize {
        if self.is_zero() {
            0
        } else {
            assert!(self.coeffs.last().map_or(false, |coeff| !coeff.is_zero()));
            self.coeffs.len() - 1
        }
    }

    /// Evaluates `self` at the given `point` in `Self::Point`.
    fn evaluate(&self, point: &Rn) -> Rn {
        if self.is_zero() {
            return Rn::zero();
        } else if point.is_zero() {
            return self.coeffs[0];
        }
        self.internal_evaluate(point)
    }
}

#[cfg(feature = "parallel")]
// Set some minimum number of field elements to be worked on per thread
// to avoid per-thread costs dominating parallel execution time.
const MIN_ELEMENTS_PER_THREAD: usize = 16;

impl<Rn: Ring> DenseUnivariatePolynomial<Rn> {
    #[inline]
    // Horner's method for polynomial evaluation
    fn horner_evaluate(poly_coeffs: &[Rn], point: &Rn) -> Rn {
        poly_coeffs
            .iter()
            .rfold(Rn::zero(), move |result, coeff| result * point + coeff)
    }

    #[cfg(not(feature = "parallel"))]
    fn internal_evaluate(&self, point: &Rn) -> Rn {
        Self::horner_evaluate(&self.coeffs, point)
    }

    #[cfg(feature = "parallel")]
    fn internal_evaluate(&self, point: &Rn) -> Rn {
        // Horners method - parallel method
        // compute the number of threads we will be using.
        let num_cpus_available = rayon::current_num_threads();
        let num_coeffs = self.coeffs.len();
        let num_elem_per_thread = max(num_coeffs / num_cpus_available, MIN_ELEMENTS_PER_THREAD);

        // run Horners method on each thread as follows:
        // 1) Split up the coefficients across each thread evenly.
        // 2) Do polynomial evaluation via horner's method for the thread's coefficients
        // 3) Scale the result point^{thread coefficient start index}
        // Then obtain the final polynomial evaluation by summing each threads result.
        self.coeffs
            .par_chunks(num_elem_per_thread)
            .enumerate()
            .map(|(i, chunk)| {
                Self::horner_evaluate(chunk, point) * point.pow([(i * num_elem_per_thread) as u64])
            })
            .sum()
    }
}

impl<Rn: Ring> DenseUVPolynomial<Rn> for DenseUnivariatePolynomial<Rn> {
    /// Constructs a new polynomial from a list of coefficients.
    fn from_coefficients_slice(coeffs: &[Rn]) -> Self {
        Self::from_coefficients_vec(coeffs.to_vec())
    }

    /// Constructs a new polynomial from a list of coefficients.
    fn from_coefficients_vec(coeffs: Vec<Rn>) -> Self {
        let mut result = Self { coeffs };
        // While there are zeros at the end of the coefficient vector, pop them off.
        result.truncate_leading_zeros();
        // Check that either the coefficients vec is empty or that the last coeff is
        // non-zero.
        assert!(result.coeffs.last().map_or(true, |coeff| !coeff.is_zero()));
        result
    }

    /// Returns the coefficients of `self`
    fn coeffs(&self) -> &[Rn] {
        &self.coeffs
    }

    /// Outputs a univariate polynomial of degree `d` where each non-leading
    /// coefficient is sampled uniformly at random from `Rn` and the leading
    /// coefficient is sampled uniformly at random from among the non-zero
    /// elements of `Rn`.
    /// ```
    fn rand<R: Rng>(d: usize, rng: &mut R) -> Self {
        let mut random_coeffs = Vec::new();

        if d > 0 {
            // d - 1 overflows when d = 0
            for _ in 0..=(d - 1) {
                random_coeffs.push(Rn::rand(rng));
            }
        }

        let mut leading_coefficient = Rn::rand(rng);

        while leading_coefficient.is_zero() {
            leading_coefficient = Rn::rand(rng);
        }

        random_coeffs.push(leading_coefficient);

        Self::from_coefficients_vec(random_coeffs)
    }
}

impl<Rn: Ring> DenseUnivariatePolynomial<Rn> {
    fn truncate_leading_zeros(&mut self) {
        while self.coeffs.last().map_or(false, |c| c.is_zero()) {
            self.coeffs.pop();
        }
    }

    /// Perform a naive n^2 multiplication of `self` by `other`.
    pub fn naive_mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            Self::zero()
        } else {
            let mut result = vec![Rn::zero(); self.degree() + other.degree() + 1];
            for (i, self_coeff) in self.coeffs.iter().enumerate() {
                for (j, other_coeff) in other.coeffs.iter().enumerate() {
                    result[i + j] += &(*self_coeff * other_coeff);
                }
            }
            Self::from_coefficients_vec(result)
        }
    }
}

impl<Rn: Ring> fmt::Debug for DenseUnivariatePolynomial<Rn> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        for (i, coeff) in self.coeffs.iter().enumerate().filter(|(_, c)| !c.is_zero()) {
            if i == 0 {
                write!(f, "\n{:?}", coeff)?;
            } else if i == 1 {
                write!(f, " + \n{:?} * x", coeff)?;
            } else {
                write!(f, " + \n{:?} * x^{}", coeff, i)?;
            }
        }
        Ok(())
    }
}

impl<Rn: Ring> Deref for DenseUnivariatePolynomial<Rn> {
    type Target = [Rn];

    fn deref(&self) -> &[Rn] {
        &self.coeffs
    }
}

impl<Rn: Ring> DerefMut for DenseUnivariatePolynomial<Rn> {
    fn deref_mut(&mut self) -> &mut [Rn] {
        &mut self.coeffs
    }
}

impl<'a, Rn: Ring> Add<&'a DenseUnivariatePolynomial<Rn>> for &DenseUnivariatePolynomial<Rn> {
    type Output = DenseUnivariatePolynomial<Rn>;

    fn add(self, other: &'a DenseUnivariatePolynomial<Rn>) -> DenseUnivariatePolynomial<Rn> {
        let mut result = if self.is_zero() {
            other.clone()
        } else if other.is_zero() {
            self.clone()
        } else if self.degree() >= other.degree() {
            let mut result = self.clone();
            result
                .coeffs
                .iter_mut()
                .zip(&other.coeffs)
                .for_each(|(a, b)| {
                    *a += b;
                });
            result
        } else {
            let mut result = other.clone();
            result
                .coeffs
                .iter_mut()
                .zip(&self.coeffs)
                .for_each(|(a, b)| {
                    *a += b;
                });
            result
        };
        result.truncate_leading_zeros();
        result
    }
}

impl<'a, Rn: Ring> AddAssign<&'a Self> for DenseUnivariatePolynomial<Rn> {
    fn add_assign(&mut self, other: &'a Self) {
        if other.is_zero() {
            self.truncate_leading_zeros();
            return;
        }

        if self.is_zero() {
            self.coeffs.clear();
            self.coeffs.extend_from_slice(&other.coeffs);
        } else {
            let other_coeffs_len = other.coeffs.len();
            if other_coeffs_len > self.coeffs.len() {
                // Add the necessary number of zero coefficients.
                self.coeffs.resize(other_coeffs_len, Rn::zero());
            }

            self.coeffs
                .iter_mut()
                .zip(&other.coeffs)
                .for_each(|(a, b)| *a += b);
        }
        self.truncate_leading_zeros();
    }
}

impl<'a, Rn: Ring> AddAssign<(Rn, &'a Self)> for DenseUnivariatePolynomial<Rn> {
    fn add_assign(&mut self, (f, other): (Rn, &'a Self)) {
        // No need to modify self if other is zero
        if other.is_zero() {
            return;
        }

        // If the first polynomial is zero, just copy the second one and scale by f.
        if self.is_zero() {
            self.coeffs.clear();
            self.coeffs.extend_from_slice(&other.coeffs);
            #[allow(clippy::suspicious_op_assign_impl)]
            self.coeffs.iter_mut().for_each(|c| *c *= &f);
            return;
        }

        // If the degree of the first polynomial is smaller, resize it.
        if self.degree() < other.degree() {
            self.coeffs.resize(other.coeffs.len(), Rn::zero());
        }

        // Add corresponding coefficients from the second polynomial, scaled by f.
        #[allow(clippy::suspicious_op_assign_impl)]
        self.coeffs
            .iter_mut()
            .zip(&other.coeffs)
            .for_each(|(a, b)| *a += f * b);

        // If the leading coefficient ends up being zero, pop it off.
        // This can happen:
        // - if they were the same degree,
        // - if a polynomial's coefficients were constructed with leading zeros.
        self.truncate_leading_zeros();
    }
}

impl<Rn: Ring> Neg for DenseUnivariatePolynomial<Rn> {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self {
        self.coeffs.iter_mut().for_each(|coeff| {
            *coeff = -*coeff;
        });
        self
    }
}

impl<'a, Rn: Ring> Sub<&'a DenseUnivariatePolynomial<Rn>> for &DenseUnivariatePolynomial<Rn> {
    type Output = DenseUnivariatePolynomial<Rn>;

    #[inline]
    fn sub(self, other: &'a DenseUnivariatePolynomial<Rn>) -> DenseUnivariatePolynomial<Rn> {
        let mut result = if self.is_zero() {
            let mut result = other.clone();
            result.coeffs.iter_mut().for_each(|c| *c = -(*c));
            result
        } else if other.is_zero() {
            self.clone()
        } else if self.degree() >= other.degree() {
            let mut result = self.clone();
            result
                .coeffs
                .iter_mut()
                .zip(&other.coeffs)
                .for_each(|(a, b)| *a -= b);
            result
        } else {
            let mut result = self.clone();
            result.coeffs.resize(other.coeffs.len(), Rn::zero());
            result
                .coeffs
                .iter_mut()
                .zip(&other.coeffs)
                .for_each(|(a, b)| *a -= b);
            result
        };
        result.truncate_leading_zeros();
        result
    }
}

impl<'a, Rn: Ring> SubAssign<&'a Self> for DenseUnivariatePolynomial<Rn> {
    #[inline]
    fn sub_assign(&mut self, other: &'a Self) {
        if self.is_zero() {
            self.coeffs.resize(other.coeffs.len(), Rn::zero());
        } else if other.is_zero() {
            return;
        } else if self.degree() >= other.degree() {
        } else {
            // Add the necessary number of zero coefficients.
            self.coeffs.resize(other.coeffs.len(), Rn::zero());
        }
        self.coeffs
            .iter_mut()
            .zip(&other.coeffs)
            .for_each(|(a, b)| {
                *a -= b;
            });
        // If the leading coefficient ends up being zero, pop it off.
        // This can happen if they were the same degree, or if other's
        // coefficients were constructed with leading zeros.
        self.truncate_leading_zeros();
    }
}

impl<Rn: Ring> Mul<Rn> for &DenseUnivariatePolynomial<Rn> {
    type Output = DenseUnivariatePolynomial<Rn>;

    #[inline]
    fn mul(self, elem: Rn) -> DenseUnivariatePolynomial<Rn> {
        if self.is_zero() || elem.is_zero() {
            DenseUnivariatePolynomial::zero()
        } else {
            let mut result = self.clone();
            cfg_iter_mut!(result).for_each(|e| {
                *e *= elem;
            });
            result
        }
    }
}

impl<Rn: Ring> Mul<Rn> for DenseUnivariatePolynomial<Rn> {
    type Output = Self;

    #[inline]
    fn mul(self, elem: Rn) -> Self {
        &self * elem
    }
}

/// Performs O(nlogn) multiplication of polynomials if Rn is smooth.
impl<'a, Rn: Ring> Mul<&'a DenseUnivariatePolynomial<Rn>> for &DenseUnivariatePolynomial<Rn> {
    type Output = DenseUnivariatePolynomial<Rn>;

    #[inline]
    fn mul(self, other: &'a DenseUnivariatePolynomial<Rn>) -> DenseUnivariatePolynomial<Rn> {
        if self.is_zero() || other.is_zero() {
            DenseUnivariatePolynomial::zero()
        } else {
            self.naive_mul(other)
        }
    }
}

macro_rules! impl_op {
    ($trait:ident, $method:ident, $ring_bound:ident) => {
        impl<R: $ring_bound> $trait<DenseUnivariatePolynomial<R>> for DenseUnivariatePolynomial<R> {
            type Output = DenseUnivariatePolynomial<R>;

            #[inline]
            fn $method(self, other: DenseUnivariatePolynomial<R>) -> DenseUnivariatePolynomial<R> {
                (&self).$method(&other)
            }
        }

        impl<'a, R: $ring_bound> $trait<&'a DenseUnivariatePolynomial<R>> for DenseUnivariatePolynomial<R> {
            type Output = DenseUnivariatePolynomial<R>;

            #[inline]
            fn $method(self, other: &'a DenseUnivariatePolynomial<R>) -> DenseUnivariatePolynomial<R> {
                (&self).$method(other)
            }
        }

        impl<'a, R: $ring_bound> $trait<DenseUnivariatePolynomial<R>> for &'a DenseUnivariatePolynomial<R> {
            type Output = DenseUnivariatePolynomial<R>;

            #[inline]
            fn $method(self, other: DenseUnivariatePolynomial<R>) -> DenseUnivariatePolynomial<R> {
                self.$method(&other)
            }
        }
    };
}

impl<Rn: Ring> Zero for DenseUnivariatePolynomial<Rn> {
    /// Returns the zero polynomial.
    fn zero() -> Self {
        Self { coeffs: Vec::new() }
    }

    /// Checks if the given polynomial is zero.
    fn is_zero(&self) -> bool {
        self.coeffs.is_empty() || self.coeffs.iter().all(|coeff| coeff.is_zero())
    }
}

impl_op!(Add, add, Ring);
impl_op!(Sub, sub, Ring);
impl_op!(Mul, mul, Ring);

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{One, UniformRand};
    use ark_std::test_rng;
    use stark_rings::cyclotomic_ring::models::goldilocks::RqNTT;

    // fn rand_sparse_poly<R: Rng>(degree: usize, rng: &mut R) -> SparsePolynomial<RqNTT> {
    //     // Initialize coeffs so that its guaranteed to have a x^{degree} term
    //     let mut coeffs = vec![(degree, RqNTT::rand(rng))];
    //     for i in 0..degree {
    //         if !rng.gen_bool(0.8) {
    //             coeffs.push((i, RqNTT::rand(rng)));
    //         }
    //     }
    //     SparsePolynomial::from_coefficients_vec(coeffs)
    // }

    #[test]
    fn rand_dense_poly_degree() {
        let rng = &mut test_rng();

        // if the leading coefficient were uniformly sampled from all of F, this
        // test would fail with high probability ~99.9%
        for i in 1..=30 {
            assert_eq!(DenseUnivariatePolynomial::<RqNTT>::rand(i, rng).degree(), i);
        }
    }

    #[test]
    fn double_polynomials_random() {
        let rng = &mut test_rng();
        for degree in 0..70 {
            let p = DenseUnivariatePolynomial::<RqNTT>::rand(degree, rng);
            let p_double = &p + &p;
            let p_quad = &p_double + &p_double;
            assert_eq!(&(&(&p + &p) + &p) + &p, p_quad);
        }
    }

    #[test]
    fn add_polynomials() {
        let rng = &mut test_rng();
        for a_degree in 0..70 {
            for b_degree in 0..70 {
                let p1 = DenseUnivariatePolynomial::<RqNTT>::rand(a_degree, rng);
                let p2 = DenseUnivariatePolynomial::<RqNTT>::rand(b_degree, rng);
                let res1 = &p1 + &p2;
                let res2 = &p2 + &p1;
                assert_eq!(res1, res2);
            }
        }
    }

    // #[test]
    // fn add_sparse_polynomials() {
    //     let rng = &mut test_rng();
    //     for a_degree in 0..70 {
    //         for b_degree in 0..70 {
    //             let p1 = DenseUnivariatePolynomial::<RqNTT>::rand(a_degree, rng);
    //             let p2 = rand_sparse_poly(b_degree, rng);
    //             let res = &p1 + &p2;
    //             assert_eq!(res, &p1 + &Into::<DenseUnivariatePolynomial<RqNTT>>::into(p2));
    //         }
    //     }
    // }

    // #[test]
    // fn add_assign_sparse_polynomials() {
    //     let rng = &mut test_rng();
    //     for a_degree in 0..70 {
    //         for b_degree in 0..70 {
    //             let p1 = DenseUnivariatePolynomial::<RqNTT>::rand(a_degree, rng);
    //             let p2 = rand_sparse_poly(b_degree, rng);

    //             let mut res = p1.clone();
    //             res += &p2;
    //             assert_eq!(res, &p1 + &Into::<DenseUnivariatePolynomial<RqNTT>>::into(p2));
    //         }
    //     }
    // }

    #[test]
    fn add_polynomials_with_mul() {
        let rng = &mut test_rng();
        for a_degree in 0..70 {
            for b_degree in 0..70 {
                let mut p1 = DenseUnivariatePolynomial::rand(a_degree, rng);
                let p2 = DenseUnivariatePolynomial::rand(b_degree, rng);
                let f = RqNTT::rand(rng);
                let f_p2 = DenseUnivariatePolynomial::from_coefficients_vec(
                    p2.coeffs.iter().map(|c| f * c).collect(),
                );
                let res2 = &f_p2 + &p1;
                p1 += (f, &p2);
                let res1 = p1;
                assert_eq!(res1, res2);
            }
        }
    }

    #[test]
    fn sub_polynomials() {
        let rng = &mut test_rng();
        for a_degree in 0..70 {
            for b_degree in 0..70 {
                let p1 = DenseUnivariatePolynomial::<RqNTT>::rand(a_degree, rng);
                let p2 = DenseUnivariatePolynomial::<RqNTT>::rand(b_degree, rng);
                let res1 = &p1 - &p2;
                let res2 = &p2 - &p1;
                assert_eq!(&res1 + &p2, p1);
                assert_eq!(res1, -res2);
            }
        }
    }

    // #[test]
    // fn sub_sparse_polynomials() {
    //     let rng = &mut test_rng();
    //     for a_degree in 0..70 {
    //         for b_degree in 0..70 {
    //             let p1 = DenseUnivariatePolynomial::<RqNTT>::rand(a_degree, rng);
    //             let p2 = rand_sparse_poly(b_degree, rng);
    //             let res = &p1 - &p2;
    //             assert_eq!(res, &p1 - &Into::<DenseUnivariatePolynomial<RqNTT>>::into(p2));
    //         }
    //     }
    // }

    // #[test]
    // fn sub_assign_sparse_polynomials() {
    //     let rng = &mut test_rng();
    //     for a_degree in 0..70 {
    //         for b_degree in 0..70 {
    //             let p1 = DenseUnivariatePolynomial::<RqNTT>::rand(a_degree, rng);
    //             let p2 = rand_sparse_poly(b_degree, rng);

    //             let mut res = p1.clone();
    //             res -= &p2;
    //             assert_eq!(res, &p1 - &Into::<DenseUnivariatePolynomial<RqNTT>>::into(p2));
    //         }
    //     }
    // }

    #[test]
    fn polynomial_additive_identity() {
        // Test adding polynomials with its negative equals 0
        let mut rng = test_rng();
        for degree in 0..70 {
            let poly = DenseUnivariatePolynomial::<RqNTT>::rand(degree, &mut rng);
            let neg = -poly.clone();
            let result = poly + neg;
            assert!(result.is_zero());
            assert_eq!(result.degree(), 0);

            // Test with SubAssign trait
            let poly = DenseUnivariatePolynomial::<RqNTT>::rand(degree, &mut rng);
            let mut result = poly.clone();
            result -= &poly;
            assert!(result.is_zero());
            assert_eq!(result.degree(), 0);
        }
    }

    #[test]
    fn evaluate_polynomials() {
        let rng = &mut test_rng();
        for a_degree in 0..70 {
            let p = DenseUnivariatePolynomial::rand(a_degree, rng);
            let point: RqNTT = RqNTT::rand(rng);
            let mut total = RqNTT::zero();
            for (i, coeff) in p.coeffs.iter().enumerate() {
                total += &(point.pow([i as u64]) * coeff);
            }
            assert_eq!(p.evaluate(&point), total);
        }
    }

    #[test]
    fn mul_random_element() {
        let rng = &mut test_rng();
        for degree in 0..70 {
            let a = DenseUnivariatePolynomial::<RqNTT>::rand(degree, rng);
            let e = RqNTT::rand(rng);
            assert_eq!(
                &a * e,
                a.naive_mul(&DenseUnivariatePolynomial::from_coefficients_slice(&[e]))
            )
        }
    }

    // #[test]
    // fn mul_polynomials_random() {
    //     let rng = &mut test_rng();
    //     for a_degree in 0..70 {
    //         for b_degree in 0..70 {
    //             let a = DenseUnivariatePolynomial::<RqNTT>::rand(a_degree, rng);
    //             let b = DenseUnivariatePolynomial::<RqNTT>::rand(b_degree, rng);
    //             assert_eq!(&a * &b, a.naive_mul(&b))
    //         }
    //     }
    // }

    // #[test]
    // fn test_leading_zero() {
    //     let n = 10;
    //     let rand_poly = DenseUnivariatePolynomial::rand(n, &mut test_rng());
    //     let coefficients = rand_poly.coeffs.clone();
    //     let leading_coefficient: RqNTT = coefficients[n];

    //     let negative_leading_coefficient = -leading_coefficient;
    //     let inverse_leading_coefficient = leading_coefficient.inverse().unwrap();

    //     let mut inverse_coefficients = coefficients.clone();
    //     inverse_coefficients[n] = inverse_leading_coefficient;

    //     let mut negative_coefficients = coefficients;
    //     negative_coefficients[n] = negative_leading_coefficient;

    //     let negative_poly = DenseUnivariatePolynomial::from_coefficients_vec(negative_coefficients);
    //     let inverse_poly = DenseUnivariatePolynomial::from_coefficients_vec(inverse_coefficients);

    //     let x = &inverse_poly * &rand_poly;
    //     assert_eq!(x.degree(), 2 * n);
    //     assert!(!x.coeffs.last().unwrap().is_zero());

    //     let y = &negative_poly + &rand_poly;
    //     assert_eq!(y.degree(), n - 1);
    //     assert!(!y.coeffs.last().unwrap().is_zero());
    // }

    #[test]
    fn test_add_assign_with_zero_self() {
        // Create a polynomial poly1 which is a zero polynomial
        let mut poly1 = DenseUnivariatePolynomial::<RqNTT> { coeffs: Vec::new() };

        // Create another polynomial poly2, which is: 2 + 3x (coefficients [2, 3])
        let poly2 = DenseUnivariatePolynomial {
            coeffs: vec![RqNTT::from(2 as u32), RqNTT::from(3 as u32)],
        };

        // Add poly2 to the zero polynomial
        // Since poly1 is zero, it should just take the coefficients of poly2.
        poly1 += (RqNTT::one(), &poly2);

        // After addition, poly1 should be equal to poly2
        assert_eq!(
            poly1.coeffs,
            vec![RqNTT::from(2 as u32), RqNTT::from(3 as u32)]
        );
    }

    #[test]
    fn test_add_assign_with_zero_other() {
        // Create a polynomial poly1: 2 + 3x (coefficients [2, 3])
        let mut poly1 = DenseUnivariatePolynomial {
            coeffs: vec![RqNTT::from(2 as u32), RqNTT::from(3 as u32)],
        };

        // Create an empty polynomial poly2 (zero polynomial)
        let poly2 = DenseUnivariatePolynomial::<RqNTT> { coeffs: Vec::new() };

        // Add zero polynomial poly2 to poly1.
        // Since poly2 is zero, poly1 should remain unchanged.
        poly1 += (RqNTT::one(), &poly2);

        // After addition, poly1 should still be [2, 3]
        assert_eq!(
            poly1.coeffs,
            vec![RqNTT::from(2 as u32), RqNTT::from(3 as u32)]
        );
    }

    #[test]
    fn test_add_assign_with_different_degrees() {
        // Create polynomial poly1: 1 + 2x + 3x^2 (coefficients [1, 2, 3])
        let mut poly1 = DenseUnivariatePolynomial {
            coeffs: vec![RqNTT::one(), RqNTT::from(2 as u32), RqNTT::from(3 as u32)],
        };

        // Create another polynomial poly2: 4 + 5x (coefficients [4, 5])
        let poly2 = DenseUnivariatePolynomial {
            coeffs: vec![RqNTT::from(4 as u32), RqNTT::from(5 as u32)],
        };

        // Add poly2 to poly1.
        // poly1 is degree 2, poly2 is degree 1, so poly2 will be padded with a zero
        // to match the degree of poly1.
        poly1 += (RqNTT::one(), &poly2);

        // After addition, the result should be:
        // 5 + 7x + 3x^2 (coefficients [5, 7, 3])
        assert_eq!(
            poly1.coeffs,
            vec![
                RqNTT::from(5 as u32),
                RqNTT::from(7 as u32),
                RqNTT::from(3 as u32)
            ]
        );
    }

    #[test]
    fn test_add_assign_with_equal_degrees() {
        // Create polynomial poly1: 1 + 2x + 3x^2 (coefficients [1, 2, 3])
        let mut poly1 = DenseUnivariatePolynomial {
            coeffs: vec![RqNTT::one(), RqNTT::from(2 as u32), RqNTT::from(3 as u32)],
        };

        // Create polynomial poly2: 4 + 5x + 6x^2 (coefficients [4, 5, 6])
        let poly2 = DenseUnivariatePolynomial {
            coeffs: vec![
                RqNTT::from(4 as u32),
                RqNTT::from(5 as u32),
                RqNTT::from(6 as u32),
            ],
        };

        // Add poly2 to poly1.
        // Since both polynomials have the same degree, we can directly add corresponding terms.
        poly1 += (RqNTT::one(), &poly2);

        // After addition, the result should be:
        // 5 + 7x + 9x^2 (coefficients [5, 7, 9])
        assert_eq!(
            poly1.coeffs,
            vec![
                RqNTT::from(5 as u32),
                RqNTT::from(7 as u32),
                RqNTT::from(9 as u32)
            ]
        );
    }

    #[test]
    fn test_add_assign_with_smaller_degrees() {
        // Create polynomial poly1: 1 + 2x (degree 1)
        let mut poly1 = DenseUnivariatePolynomial {
            coeffs: vec![RqNTT::one(), RqNTT::from(2 as u32)],
        };

        // Create polynomial poly2: 3 + 4x + 5x^2 (degree 2)
        let poly2 = DenseUnivariatePolynomial {
            coeffs: vec![
                RqNTT::from(3 as u32),
                RqNTT::from(4 as u32),
                RqNTT::from(5 as u32),
            ],
        };

        // Add poly2 to poly1.
        // poly1 has degree 1, poly2 has degree 2. So poly1 must be padded with zero coefficients
        // for the higher degree terms to match poly2's degree.
        poly1 += (RqNTT::one(), &poly2);

        // After addition, the result should be:
        // 4 + 6x + 5x^2 (coefficients [4, 6, 5])
        assert_eq!(
            poly1.coeffs,
            vec![
                RqNTT::from(4 as u32),
                RqNTT::from(6 as u32),
                RqNTT::from(5 as u32)
            ]
        );
    }

    // #[test]
    // fn test_add_assign_mixed_with_zero_self() {
    //     // Create a zero DenseUnivariatePolynomial
    //     let mut poly1 = DenseUnivariatePolynomial::<RqNTT> { coeffs: Vec::new() };

    //     // Create a SparsePolynomial: 2 + 3x (coefficients [2, 3])
    //     let poly2 =
    //         SparsePolynomial::from_coefficients_slice(&[(0, RqNTT::from(2)), (1, RqNTT::from(3))]);

    //     // Add poly2 to the zero polynomial
    //     poly1 += &poly2;

    //     // After addition, the result should be 2 + 3x
    //     assert_eq!(poly1.coeffs, vec![RqNTT::from(2), RqNTT::from(3)]);
    // }

    // #[test]
    // fn test_add_assign_mixed_with_zero_other() {
    //     // Create a DenseUnivariatePolynomial: 2 + 3x (coefficients [2, 3])
    //     let mut poly1 = DenseUnivariatePolynomial {
    //         coeffs: vec![RqNTT::from(2), RqNTT::from(3)],
    //     };

    //     // Create a zero SparsePolynomial
    //     let poly2 = SparsePolynomial::from_coefficients_slice(&[]);

    //     // Add poly2 to poly1
    //     poly1 += &poly2;

    //     // After addition, the result should still be 2 + 3x
    //     assert_eq!(poly1.coeffs, vec![RqNTT::from(2), RqNTT::from(3)]);
    // }

    // #[test]
    // fn test_add_assign_mixed_with_different_degrees() {
    //     // Create a DenseUnivariatePolynomial: 1 + 2x + 3x^2 (coefficients [1, 2, 3])
    //     let mut poly1 = DenseUnivariatePolynomial {
    //         coeffs: vec![RqNTT::one(), RqNTT::from(2), RqNTT::from(3)],
    //     };

    //     // Create a SparsePolynomial: 4 + 5x (coefficients [4, 5])
    //     let poly2 =
    //         SparsePolynomial::from_coefficients_slice(&[(0, RqNTT::from(4)), (1, RqNTT::from(5))]);

    //     // Add poly2 to poly1
    //     poly1 += &poly2;

    //     // After addition, the result should be 5 + 7x + 3x^2 (coefficients [5, 7, 3])
    //     assert_eq!(poly1.coeffs, vec![RqNTT::from(5), RqNTT::from(7), RqNTT::from(3)]);
    // }

    // #[test]
    // fn test_add_assign_mixed_with_smaller_degree() {
    //     // Create a DenseUnivariatePolynomial: 1 + 2x (degree 1)
    //     let mut poly1 = DenseUnivariatePolynomial {
    //         coeffs: vec![RqNTT::one(), RqNTT::from(2)],
    //     };

    //     // Create a SparsePolynomial: 3 + 4x + 5x^2 (degree 2)
    //     let poly2 = SparsePolynomial::from_coefficients_slice(&[
    //         (0, RqNTT::from(3)),
    //         (1, RqNTT::from(4)),
    //         (2, RqNTT::from(5)),
    //     ]);

    //     // Add poly2 to poly1
    //     poly1 += &poly2;

    //     // After addition, the result should be: 4 + 6x + 5x^2 (coefficients [4, 6, 5])
    //     assert_eq!(poly1.coeffs, vec![RqNTT::from(4), RqNTT::from(6), RqNTT::from(5)]);
    // }

    // #[test]
    // fn test_add_assign_mixed_with_equal_degrees() {
    //     // Create a DenseUnivariatePolynomial: 1 + 2x + 3x^2 (coefficients [1, 2, 3])
    //     let mut poly1 = DenseUnivariatePolynomial {
    //         coeffs: vec![RqNTT::one(), RqNTT::from(2), RqNTT::from(3)],
    //     };

    //     // Create a SparsePolynomial: 4 + 5x + 6x^2 (coefficients [4, 5, 6])
    //     let poly2 = SparsePolynomial::from_coefficients_slice(&[
    //         (0, RqNTT::from(4)),
    //         (1, RqNTT::from(5)),
    //         (2, RqNTT::from(6)),
    //     ]);

    //     // Add poly2 to poly1
    //     poly1 += &poly2;

    //     // After addition, the result should be 5 + 7x + 9x^2 (coefficients [5, 7, 9])
    //     assert_eq!(poly1.coeffs, vec![RqNTT::from(5), RqNTT::from(7), RqNTT::from(9)]);
    // }

    // #[test]
    // fn test_add_assign_mixed_with_larger_degree() {
    //     // Create a DenseUnivariatePolynomial: 1 + 2x + 3x^2 + 4x^3 (degree 3)
    //     let mut poly1 = DenseUnivariatePolynomial {
    //         coeffs: vec![RqNTT::one(), RqNTT::from(2), RqNTT::from(3), RqNTT::from(4)],
    //     };

    //     // Create a SparsePolynomial: 3 + 4x (degree 1)
    //     let poly2 =
    //         SparsePolynomial::from_coefficients_slice(&[(0, RqNTT::from(3)), (1, RqNTT::from(4))]);

    //     // Add poly2 to poly1
    //     poly1 += &poly2;

    //     // After addition, the result should be: 4 + 6x + 3x^2 + 4x^3 (coefficients [4, 6, 3, 4])
    //     assert_eq!(
    //         poly1.coeffs,
    //         vec![RqNTT::from(4), RqNTT::from(6), RqNTT::from(3), RqNTT::from(4)]
    //     );
    // }

    // #[test]
    // fn test_truncate_leading_zeros_after_addition() {
    //     // Create a DenseUnivariatePolynomial: 1 + 2x + 3x^2 (coefficients [1, 2, 3])
    //     let mut poly1 = DenseUnivariatePolynomial {
    //         coeffs: vec![RqNTT::one(), RqNTT::from(2), RqNTT::from(3)],
    //     };

    //     // Create a SparsePolynomial: -1 - 2x - 3x^2 (coefficients [-1, -2, -3])
    //     let poly2 = SparsePolynomial::from_coefficients_slice(&[
    //         (0, -RqNTT::one()),
    //         (1, -RqNTT::from(2)),
    //         (2, -RqNTT::from(3)),
    //     ]);

    //     // Add poly2 to poly1, which should result in a zero polynomial
    //     poly1 += &poly2;

    //     // The resulting polynomial should be zero, with an empty coefficient vector
    //     assert!(poly1.is_zero());
    //     assert_eq!(poly1.coeffs, vec![]);
    // }

    // #[test]
    // fn test_truncate_leading_zeros_after_sparse_addition() {
    //     // Create a DenseUnivariatePolynomial with leading non-zero coefficients.
    //     let poly1 = DenseUnivariatePolynomial {
    //         coeffs: vec![RqNTT::from(3), RqNTT::from(2), RqNTT::one()],
    //     };

    //     // Create a SparsePolynomial to subtract the coefficients of poly1,
    //     // leaving trailing zeros after addition.
    //     let poly2 = SparsePolynomial::from_coefficients_slice(&[
    //         (0, -RqNTT::from(3)),
    //         (1, -RqNTT::from(2)),
    //         (2, -RqNTT::one()),
    //     ]);

    //     // Perform addition using the Add implementation.
    //     let result = &poly1 + &poly2;

    //     // Assert that the resulting polynomial is zero.
    //     assert!(result.is_zero(), "The resulting polynomial should be zero.");
    //     assert_eq!(result.coeffs, vec![], "Leading zeros were not truncated.");
    // }

    #[test]
    fn test_dense_dense_add_assign_with_zero_self() {
        // Create a zero polynomial
        let mut poly1 = DenseUnivariatePolynomial { coeffs: Vec::new() };

        // Create a non-zero polynomial: 2 + 3x
        let poly2 = DenseUnivariatePolynomial {
            coeffs: vec![RqNTT::from(2 as u32), RqNTT::from(3 as u32)],
        };

        // Add the non-zero polynomial to the zero polynomial
        poly1 += &poly2;

        // Check that poly1 now equals poly2
        assert_eq!(poly1.coeffs, poly2.coeffs);
    }

    #[test]
    fn test_dense_dense_add_assign_with_zero_other() {
        // Create a non-zero polynomial: 2 + 3x
        let mut poly1 = DenseUnivariatePolynomial {
            coeffs: vec![RqNTT::from(2 as u32), RqNTT::from(3 as u32)],
        };

        // Create a zero polynomial
        let poly2 = DenseUnivariatePolynomial { coeffs: Vec::new() };

        // Add the zero polynomial to poly1
        poly1 += &poly2;

        // Check that poly1 remains unchanged
        assert_eq!(
            poly1.coeffs,
            vec![RqNTT::from(2 as u32), RqNTT::from(3 as u32)]
        );
    }

    #[test]
    fn test_dense_dense_add_assign_with_different_degrees() {
        // Create a polynomial: 1 + 2x + 3x^2
        let mut poly1 = DenseUnivariatePolynomial {
            coeffs: vec![RqNTT::one(), RqNTT::from(2 as u32), RqNTT::from(3 as u32)],
        };

        // Create a smaller polynomial: 4 + 5x
        let poly2 = DenseUnivariatePolynomial {
            coeffs: vec![RqNTT::from(4 as u32), RqNTT::from(5 as u32)],
        };

        // Add the smaller polynomial to the larger one
        poly1 += &poly2;

        assert_eq!(
            poly1.coeffs,
            vec![
                RqNTT::from(5 as u32),
                RqNTT::from(7 as u32),
                RqNTT::from(3 as u32)
            ]
        );
    }

    #[test]
    fn test_dense_dense_truncate_leading_zeros_after_addition() {
        // Create a first polynomial
        let mut poly1 = DenseUnivariatePolynomial {
            coeffs: vec![RqNTT::one(), RqNTT::from(2 as u32)],
        };

        // Create another polynomial that will cancel out the first two terms
        let poly2 = DenseUnivariatePolynomial {
            coeffs: vec![-poly1.coeffs[0], -poly1.coeffs[1]],
        };

        // Add the two polynomials
        poly1 += &poly2;

        // Check that the resulting polynomial is zero (empty coefficients)
        assert!(poly1.is_zero());
        assert_eq!(poly1.coeffs, vec![]);
    }

    #[test]
    fn test_dense_dense_add_assign_with_equal_degrees() {
        // Create two polynomials with the same degree
        let mut poly1 = DenseUnivariatePolynomial {
            coeffs: vec![RqNTT::one(), RqNTT::from(2 as u32), RqNTT::from(3 as u32)],
        };
        let poly2 = DenseUnivariatePolynomial {
            coeffs: vec![
                RqNTT::from(4 as u32),
                RqNTT::from(5 as u32),
                RqNTT::from(6 as u32),
            ],
        };

        // Add the polynomials
        poly1 += &poly2;

        // Check the resulting coefficients
        assert_eq!(
            poly1.coeffs,
            vec![
                RqNTT::from(5 as u32),
                RqNTT::from(7 as u32),
                RqNTT::from(9 as u32)
            ]
        );
    }

    #[test]
    fn test_dense_dense_add_assign_with_other_zero_truncates_leading_zeros() {
        // Create a polynomial with leading zeros: 1 + 2x + 0x^2 + 0x^3
        let mut poly1 = DenseUnivariatePolynomial {
            coeffs: vec![
                RqNTT::one(),
                RqNTT::from(2 as u32),
                RqNTT::zero(),
                RqNTT::zero(),
            ],
        };

        // Create a zero polynomial
        let poly2 = DenseUnivariatePolynomial { coeffs: Vec::new() };

        // Add the zero polynomial to poly1
        poly1 += &poly2;

        // Check that the leading zeros are truncated
        assert_eq!(poly1.coeffs, vec![RqNTT::one(), RqNTT::from(2 as u32)]);

        // Ensure the polynomial is not zero (as it has non-zero coefficients)
        assert!(!poly1.is_zero());
    }
}
