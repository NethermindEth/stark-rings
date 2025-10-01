use ark_ff::{Field, Fp2, Fp2Config, Fp4, Fp4Config, MontBackend, MontFp};
use ark_std::{mem::swap, ops::Mul, vec::*};

use crate::{
    cyclotomic_ring::{
        CyclotomicConfig, CyclotomicPolyRingGeneral, CyclotomicPolyRingNTTGeneral, CRT, ICRT,
    },
    impl_crt_icrt_for_a_ring,
    poly_ring::PolyRing,
    traits::FromRandomBytes,
    CoeffRing, Cyclotomic, Monomial, OverField,
};

mod ntt;
mod utils;

mod fq_def {
    #![allow(non_local_definitions)]
    use ark_ff::{Fp64, MontBackend};

    #[derive(ark_ff::MontConfig)]
    #[modulus = "15912092521325583641"]
    #[generator = "3"]
    pub struct FqConfig;
    pub type Fq = Fp64<MontBackend<FqConfig, 1>>;
}

pub use fq_def::*;

pub type RqNTT = CyclotomicPolyRingNTTGeneral<FrogRingConfig, 1, { ntt::N }>;
pub type RqPoly = CyclotomicPolyRingGeneral<FrogRingConfig, 1, { ntt::D }>;

pub struct Frog2Config;
pub struct Frog4Config;
pub struct FrogRingConfig;

impl Fp2Config for Frog2Config {
    type Fp = Fq;

    const FROBENIUS_COEFF_FP2_C1: &'static [Self::Fp] =
        &[<Fq as Field>::ONE, MontFp!("15912092521325583640")];
    const NONRESIDUE: Self::Fp = MontFp!("2755067726615789629");
}

pub type Fq2 = Fp2<Frog2Config>;

impl Fp4Config for Frog4Config {
    type Fp2Config = Frog2Config;

    const FROBENIUS_COEFF_FP4_C1: &'static [<Self::Fp2Config as Fp2Config>::Fp] = &[
        <Fq as Field>::ONE,
        MontFp!("2674048055506678227"),
        MontFp!("15912092521325583640"),
        MontFp!("13238044465818905414"),
    ];
    const NONRESIDUE: Fq2 = Fp2::new(<Fq as Field>::ZERO, <Fq as Field>::ONE);
}

pub type Fq4 = Fp4<Frog4Config>;

impl FromRandomBytes<Fq4> for Fq4 {
    #[inline(always)]
    fn byte_size() -> usize {
        4 * 8
    }

    fn try_from_random_bytes(bytes: &[u8]) -> Option<Fq4> {
        Self::from_random_bytes(bytes)
    }
}

impl CyclotomicConfig<1> for FrogRingConfig {
    type BaseCRTField = Fq4;
    type BaseFieldConfig = MontBackend<FqConfig, 1>;

    const CRT_FIELD_EXTENSION_DEGREE: usize = 4;

    fn reduce_in_place(coefficients: &mut Vec<Fq>) {
        let (left, right) = coefficients.split_at_mut(ntt::D);
        for (coeff_left, coeff_right) in left.iter_mut().zip(right) {
            *coeff_left -= coeff_right;
        }

        coefficients.resize(ntt::D, <Fq as Field>::ZERO);
    }

    #[inline(always)]
    fn crt_in_place(coefficients: &mut [Fq]) {
        ntt::frog_crt_in_place(coefficients);
    }

    #[inline(always)]
    fn crt(coefficients: Vec<Fq>) -> Vec<Fq4> {
        ntt::frog_crt(coefficients)
    }

    #[inline(always)]
    fn icrt(evaluations: Vec<Fq4>) -> Vec<Fq> {
        ntt::frog_icrt(evaluations)
    }

    #[inline(always)]
    fn icrt_in_place(evaluations: &mut [Fq]) {
        ntt::frog_icrt_in_place(evaluations);
    }
}

impl OverField for RqNTT {}
impl CoeffRing for RqPoly {}

impl Mul<Fq4> for RqNTT {
    type Output = Self;

    fn mul(self, rhs: Fq4) -> Self {
        Self(self.0.map(|x| x * rhs))
    }
}

impl From<Fq4> for RqNTT {
    fn from(value: Fq4) -> Self {
        Self::from_scalar(value)
    }
}

impl Cyclotomic for RqPoly {
    fn rot(&mut self) {
        let mut buf = -self.0[Self::dimension() - 1];

        for i in 0..Self::dimension() {
            swap(&mut buf, &mut self.0[i]);
        }
    }
}

impl<'a> Mul<&'a Monomial<Self>> 
    for RqPoly {
    type Output = Self;

    fn mul(mut self, m: &'a Monomial<Self>) ->Self {
    use crate::Ring;
    use ark_std::{One, Zero};
        let coeff = m.value();
        let shift = m.index();

        // Monomial coeff is either 0 or 1, so no need to do true multiplication
        if coeff.is_zero() {
            return Self::ZERO;
        }

        if shift == 0 && coeff.is_one() {
            return self;
        }

        if shift > 0 {
            let mut new_coeffs = [<Self as PolyRing>::BaseRing::zero(); ntt::D];
            for (i, c) in self.0.iter().enumerate() {

                let j = i + shift;
                if j < ntt::D {
                    new_coeffs[j] = *c;
                } else {
                    new_coeffs[j - ntt::D] -= c;
                }
            }
            self.0.copy_from_slice(&new_coeffs);
        }

        self
    }
}


impl_crt_icrt_for_a_ring!(RqNTT, RqPoly, FrogRingConfig);

#[cfg(test)]
mod test {
    use ark_poly::{
        univariate::{DenseOrSparsePolynomial, DensePolynomial, SparsePolynomial},
        DenseUVPolynomial,
    };
    use ark_std::UniformRand;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;
    use crate::{
        balanced_decomposition::Decompose,
        cyclotomic_ring::crt::{CRT, ICRT},
        PolyRing, Ring,
    };

    #[test]
    fn test_implements_decompose() {
        fn takes_decompose<T: Decompose>(_x: T) {}

        let x = RqPoly::ONE;

        takes_decompose(x);
    }

    #[test]
    fn test_crt_one() {
        let one = RqPoly::ONE;

        assert_eq!(one.crt(), RqNTT::ONE)
    }

    #[test]
    fn test_icrt_one() {
        let one = RqNTT::ONE;

        assert_eq!(one.icrt(), RqPoly::ONE)
    }

    // Note: if ord X = 8 then X can't be a quadratic residue.
    #[test]
    fn test_nonresidue_is_order_24() {
        let nonresidue: Fq = Frog2Config::NONRESIDUE;

        let mut pow = nonresidue;

        for _i in 0..6 {
            pow *= nonresidue;
            assert_ne!(pow, <Fq as Field>::ONE);
        }

        assert_eq!(pow * nonresidue, <Fq as Field>::ONE);
    }

    #[test]
    fn test_reduce() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut coeffs: Vec<Fq> = (0..(2 * ntt::D)).map(|_| Fq::rand(&mut rng)).collect();

        let poly = DensePolynomial::from_coefficients_slice(&coeffs);

        // X^24 - X^12 + 1
        let the_cyclotomic = SparsePolynomial::from_coefficients_slice(&[
            (16, <Fq as Field>::ONE),
            (0, <Fq as Field>::ONE),
        ]);

        let (_, r) =
            DenseOrSparsePolynomial::divide_with_q_and_r(&poly.into(), &the_cyclotomic.into())
                .unwrap();

        FrogRingConfig::reduce_in_place(&mut coeffs);

        assert_eq!(r.coeffs(), &coeffs);
    }

    #[test]
    fn test_mul_crt() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let coeff_1 = RqPoly::rand(&mut rng);
        let coeff_2 = RqPoly::rand(&mut rng);

        let ntt_form_1: RqNTT = coeff_1.crt();
        let ntt_form_2: RqNTT = coeff_2.crt();

        let ntt_mul = ntt_form_1 * ntt_form_2;
        let coeffs_mul = coeff_1 * coeff_2;

        // ntt_mul.coeffs() performs INTT while coeffs_mul.coeffs() just returns the
        // coefficients
        assert_eq!(ntt_mul.icrt(), coeffs_mul);
    }

    #[test]
    fn test_cyclotomic() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let mut a = RqPoly::rand(&mut rng);
        let initial_a = a;

        let x = RqPoly::x();

        for i in 1..RqPoly::dimension() {
            a.rot();
            assert_eq!(a, initial_a * x.pow([i as u64]));
        }
    }
}
