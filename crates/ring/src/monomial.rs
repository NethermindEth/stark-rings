use super::{CoeffRing, PolyRing, Zq};
use crate::{ConversionError, Ring};
use ark_std::One;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MonomialError<R: PolyRing> {
    #[error("{0}")]
    Conversion(#[from] ConversionError),
    #[error("Range check failed: a {0} != ct {1}")]
    RangeCheck(R::BaseRing, R::BaseRing),
}

/// Single-term polynomial $m = X^i \in R_q$
///
/// We consider only the monomial set $\\{0, 1, X, ..., X^{d-1}\\}$.
pub fn monomial<R: PolyRing>(i: usize, coeff: R::BaseRing) -> R {
    let mut m = R::ZERO;
    m.coeffs_mut()[i] = coeff;
    m
}

/// Zero monomial
pub fn zero_monomial<R: PolyRing>() -> R {
    R::ZERO
}

/// Monomial with coefficient 1
pub fn unit_monomial<R: PolyRing>(i: usize) -> R {
    monomial(i, R::BaseRing::one())
}

/// $\psi$ table function
///
/// Calculates $\psi = \sum_{i \in [1, d')} i (X^{-i} + X^i) \in R_q$.
pub fn psi<R: CoeffRing>() -> R
where
    R::BaseRing: Zq,
{
    let d = R::dimension();
    let d_prime = R::dimension() / 2;
    (1..d_prime)
        .map(|i| {
            (unit_monomial::<R>(i) - unit_monomial::<R>(d - i))
                * <R as PolyRing>::BaseRing::from(i as u128)
        })
        .sum()
}

/// \text{exp} unit function
///
/// Computes a unit monomial using `a` as an exponent, where
/// \text{exp(a)} = \text{sign}(a)X^a.
/// When a < 0, the output will be the unit monomial X^{d-a}.
pub fn exp<R: CoeffRing>(a: R::BaseRing) -> Result<R, MonomialError<R>>
where
    R::BaseRing: Zq,
{
    let centered = a.center().to_usize()?;
    if a.sign() + R::BaseRing::ONE > R::BaseRing::ZERO {
        Ok(unit_monomial(centered))
    } else {
        Ok(unit_monomial(R::dimension() - centered))
    }
}

/// \text{exp} signed function
///
/// Computes a monomial using `a` as an exponent, where
/// \text{exp(a)} = \text{sign}(a)X^a
pub fn exp_signed<R: CoeffRing>(a: R::BaseRing) -> Result<R, MonomialError<R>>
where
    R::BaseRing: Zq,
{
    Ok(monomial::<R>(a.center().to_usize()?, R::BaseRing::one()) * a.sign())
}

/// Monomial range-check
///
/// Calculates $\text{ct}(b \psi) \stackrel{?}{=} a$, where $b = \text{EXP(a)} =
/// \text{sign}(a)X^a$. If the equality holds, then $a \in (-d', d')$.
pub fn psi_range_check<R: CoeffRing>(a: R::BaseRing) -> Result<(), MonomialError<R>>
where
    R::BaseRing: Zq,
{
    let b: R = exp(a)?;

    let ct = (psi::<R>() * b).ct();

    (a == ct)
        .then_some(())
        .ok_or(MonomialError::RangeCheck(a, ct))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cyclotomic_ring::models::frog_ring::RqPoly, PolyRing};
    // RqPoly has degree 16

    #[test]
    fn test_monomial_ops() {
        let zero = zero_monomial::<RqPoly>();
        let one = unit_monomial::<RqPoly>(0);

        let x2 = monomial::<RqPoly>(2, <RqPoly as PolyRing>::BaseRing::one());
        let x15 = monomial::<RqPoly>(15, <RqPoly as PolyRing>::BaseRing::one());

        // 0 + 1 = 1
        assert_eq!(zero + one, one);
        // X^2 + X^2 = 2X^2
        assert_eq!(PolyRing::coeffs(&(x2 + x2))[2], 2.into());
        // X^2 * X^15 = -X
        assert_eq!(
            PolyRing::coeffs(&(x2 * x15))[1],
            -<RqPoly as PolyRing>::BaseRing::ONE
        );
    }

    #[test]
    fn test_monomial_range_check() {
        // Range check will pass for values -7..=7
        let a1 = <RqPoly as PolyRing>::BaseRing::from(1u128);
        let a7 = <RqPoly as PolyRing>::BaseRing::from(7u128);
        let a8 = <RqPoly as PolyRing>::BaseRing::from(8u128);
        let an1 = <RqPoly as PolyRing>::BaseRing::from(0u128) - a1;
        let an8 = <RqPoly as PolyRing>::BaseRing::from(0u128) - a8;

        assert!(psi_range_check::<RqPoly>(a1).is_ok());
        assert!(psi_range_check::<RqPoly>(a7).is_ok());
        assert!(psi_range_check::<RqPoly>(a8).is_err());
        assert!(psi_range_check::<RqPoly>(an1).is_ok());
        assert!(psi_range_check::<RqPoly>(an8).is_err());
    }
}
