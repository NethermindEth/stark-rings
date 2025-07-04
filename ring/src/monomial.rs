use super::{CoeffRing, PolyRing, Zq};
use crate::ConversionError;
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
            (-R::ONE * unit_monomial::<R>(d - i) + unit_monomial::<R>(i))
                * <R as PolyRing>::BaseRing::from(i as u128)
        })
        .sum()
}

/// \text{EXP} function
///
/// Computes a monomial using `a` as an exponent, where
/// \text{EXP(a)} = \text{sign}(a)X^a
pub fn exp<R: CoeffRing>(a: R::BaseRing) -> Result<R, MonomialError<R>>
where
    R::BaseRing: Zq,
{
    Ok(monomial::<R>(a.center().to_usize()?, R::BaseRing::one()) * a.sign())
}

/// Monomial range-check
///
/// Calculates $\text{ct}(b \psi) \stackrel{?}{=} a$, where $b = \text{EXP(a)} = \text{sign}(a)X^a$.
/// If the equality holds, then $a \in (-d', d')$.
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
    use crate::{cyclotomic_ring::models::goldilocks::RqPoly, PolyRing, Ring};
    // RqPoly has degree 24

    #[test]
    fn test_monomial_ops() {
        let zero = zero_monomial::<RqPoly>();
        let one = unit_monomial::<RqPoly>(0);

        let x2 = monomial::<RqPoly>(2, <RqPoly as PolyRing>::BaseRing::one());
        let x23 = monomial::<RqPoly>(23, <RqPoly as PolyRing>::BaseRing::one());

        // 0 + 1 = 1
        assert_eq!(zero + one, one);
        // X^2 + X^2 = 2X^2
        assert_eq!(PolyRing::coeffs(&(x2 + x2))[2], 2.into());
        // X^2 * X^23 = -X
        assert_eq!(
            PolyRing::coeffs(&(x2 * x23))[1],
            -<RqPoly as PolyRing>::BaseRing::ONE
        );
    }

    #[test]
    fn test_monomial_range_check() {
        // Range check will pass for values -11..=11
        let a1 = <RqPoly as PolyRing>::BaseRing::from(1u128);
        let a11 = <RqPoly as PolyRing>::BaseRing::from(11u128);
        let a12 = <RqPoly as PolyRing>::BaseRing::from(12u128);
        let an1 = <RqPoly as PolyRing>::BaseRing::from(0u128) - a1;
        let an12 = <RqPoly as PolyRing>::BaseRing::from(0u128) - a12;

        assert!(psi_range_check::<RqPoly>(a1).is_ok());
        assert!(psi_range_check::<RqPoly>(a11).is_ok());
        assert!(psi_range_check::<RqPoly>(a12).is_err());
        assert!(psi_range_check::<RqPoly>(an1).is_ok());
        assert!(psi_range_check::<RqPoly>(an12).is_err());
    }
}
