use ark_std::{ops::Mul, vec::*};
use crypto_primitives::{ConstRing, Field};

use crate::{cyclotomic_ring::Flatten, traits::FromRandomBytes, Zq};

pub trait PolyRing: ConstRing + From<Vec<Self::BaseRing>> + FromRandomBytes<Self> + From<u128> {
    type BaseRing: ConstRing + From<u128>;

    fn coeffs(&self) -> &[Self::BaseRing];
    fn coeffs_mut(&mut self) -> &mut [Self::BaseRing];
    fn into_coeffs(self) -> Vec<Self::BaseRing>;

    fn dimension() -> usize;

    fn from_scalar(scalar: Self::BaseRing) -> Self;
}

pub trait OverField:
    PolyRing<BaseRing: Field>
    + Mul<Self::BaseRing, Output = Self>
    + From<Self::BaseRing>
    + Flatten
{
}

/// Polynomial ring, coefficients form
pub trait CoeffRing: OverField
where
    Self::BaseRing: Zq,
{
    fn ct(&self) -> Self::BaseRing;
}

impl<O: OverField> CoeffRing for O
where
    O::BaseRing: Zq,
{
    fn ct(&self) -> Self::BaseRing {
        self.coeffs()[0].clone()
    }
}
