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
pub struct DensePolynomial<Rn: Ring> {
    /// The coefficient of `x^i` is stored at location `i` in `self.coeffs`.
    pub coeffs: Vec<Rn>,
}

impl<Rn: Ring> Polynomial<Rn> for DensePolynomial<Rn> {
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

impl<Rn: Ring> DensePolynomial<Rn> {
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

impl<Rn: Ring> DenseUVPolynomial<Rn> for DensePolynomial<Rn> {
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
    /// coefficient is sampled uniformly at random from `F` and the leading
    /// coefficient is sampled uniformly at random from among the non-zero
    /// elements of `F`.
    ///
    /// # Example
    /// ```
    /// use ark_std::test_rng;
    /// use ark_test_curves::bls12_381::Fr;
    /// use ark_poly::{univariate::DensePolynomial, Polynomial, DenseUVPolynomial};
    ///
    /// let rng = &mut test_rng();
    /// let poly = DensePolynomial::<Fr>::rand(8, rng);
    /// assert_eq!(poly.degree(), 8);
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

impl<Rn: Ring> DensePolynomial<Rn> {
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

impl<Rn: Ring> fmt::Debug for DensePolynomial<Rn> {
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

impl<Rn: Ring> Deref for DensePolynomial<Rn> {
    type Target = [Rn];

    fn deref(&self) -> &[Rn] {
        &self.coeffs
    }
}

impl<Rn: Ring> DerefMut for DensePolynomial<Rn> {
    fn deref_mut(&mut self) -> &mut [Rn] {
        &mut self.coeffs
    }
}

impl<'a, Rn: Ring> Add<&'a DensePolynomial<Rn>> for &DensePolynomial<Rn> {
    type Output = DensePolynomial<Rn>;

    fn add(self, other: &'a DensePolynomial<Rn>) -> DensePolynomial<Rn> {
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

impl<'a, Rn: Ring> AddAssign<&'a Self> for DensePolynomial<Rn> {
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

impl<'a, Rn: Ring> AddAssign<(Rn, &'a Self)> for DensePolynomial<Rn> {
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

impl<Rn: Ring> Neg for DensePolynomial<Rn> {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self {
        self.coeffs.iter_mut().for_each(|coeff| {
            *coeff = -*coeff;
        });
        self
    }
}

impl<'a, Rn: Ring> Sub<&'a DensePolynomial<Rn>> for &DensePolynomial<Rn> {
    type Output = DensePolynomial<Rn>;

    #[inline]
    fn sub(self, other: &'a DensePolynomial<Rn>) -> DensePolynomial<Rn> {
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

impl<'a, Rn: Ring> SubAssign<&'a Self> for DensePolynomial<Rn> {
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

impl<Rn: Ring> Mul<Rn> for &DensePolynomial<Rn> {
    type Output = DensePolynomial<Rn>;

    #[inline]
    fn mul(self, elem: Rn) -> DensePolynomial<Rn> {
        if self.is_zero() || elem.is_zero() {
            DensePolynomial::zero()
        } else {
            let mut result = self.clone();
            cfg_iter_mut!(result).for_each(|e| {
                *e *= elem;
            });
            result
        }
    }
}

impl<Rn: Ring> Mul<Rn> for DensePolynomial<Rn> {
    type Output = Self;

    #[inline]
    fn mul(self, elem: Rn) -> Self {
        &self * elem
    }
}

/// Performs O(nlogn) multiplication of polynomials if F is smooth.
impl<'a, Rn: Ring> Mul<&'a DensePolynomial<Rn>> for &DensePolynomial<Rn> {
    type Output = DensePolynomial<Rn>;

    #[inline]
    fn mul(self, other: &'a DensePolynomial<Rn>) -> DensePolynomial<Rn> {
        if self.is_zero() || other.is_zero() {
            DensePolynomial::zero()
        } else {
            self.naive_mul(other)
        }
    }
}

macro_rules! impl_op {
    ($trait:ident, $method:ident, $ring_bound:ident) => {
        impl<F: $ring_bound> $trait<DensePolynomial<F>> for DensePolynomial<F> {
            type Output = DensePolynomial<F>;

            #[inline]
            fn $method(self, other: DensePolynomial<F>) -> DensePolynomial<F> {
                (&self).$method(&other)
            }
        }

        impl<'a, F: $ring_bound> $trait<&'a DensePolynomial<F>> for DensePolynomial<F> {
            type Output = DensePolynomial<F>;

            #[inline]
            fn $method(self, other: &'a DensePolynomial<F>) -> DensePolynomial<F> {
                (&self).$method(other)
            }
        }

        impl<'a, F: $ring_bound> $trait<DensePolynomial<F>> for &'a DensePolynomial<F> {
            type Output = DensePolynomial<F>;

            #[inline]
            fn $method(self, other: DensePolynomial<F>) -> DensePolynomial<F> {
                self.$method(&other)
            }
        }
    };
}

impl<Rn: Ring> Zero for DensePolynomial<Rn> {
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