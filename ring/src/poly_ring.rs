use ark_crypto_primitives::sponge::Absorb;
use ark_ff::Field;
use ark_std::{ops::Mul, vec::*};

use crate::{cyclotomic_ring::Flatten, traits::FromRandomBytes, Ring, Zq};

pub trait PolyRing: Ring + From<Vec<Self::BaseRing>> + FromRandomBytes<Self> + From<u128> {
    type BaseRing: Ring;

    fn coeffs(&self) -> &[Self::BaseRing];
    fn coeffs_mut(&mut self) -> &mut [Self::BaseRing];
    fn into_coeffs(self) -> Vec<Self::BaseRing>;

    fn dimension() -> usize;

    fn from_scalar(scalar: Self::BaseRing) -> Self;
}

pub trait OverField:
    PolyRing<BaseRing: Field<BasePrimeField: Absorb>>
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
        self.coeffs()[0]
    }
}
