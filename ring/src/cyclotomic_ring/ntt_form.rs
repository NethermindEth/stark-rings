use ark_ff::{Field, Fp};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    fmt::{Debug, Display, Formatter},
    hash::Hash,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    rand::Rng,
    vec::*,
    One, UniformRand, Zero,
};
use derive_more::{From, Into};

use super::ring_config::CyclotomicConfig;
use crate::{
    traits::{FromRandomBytes, MulUnchecked},
    PolyRing, Ring,
};

/// A cyclotomic ring Fp[X]/(Phi_m(X)) in the CRT-form.
/// * `C` is the configuration of the cyclotomic ring.
/// * `N` is the byte size of the underlying prime field.
/// * `D` is the number of factors in the CRT-representation of the ring.
#[derive(From, Into, CanonicalSerialize, CanonicalDeserialize)]
pub struct CyclotomicPolyRingNTTGeneral<C: CyclotomicConfig<N>, const N: usize, const D: usize>(
    pub(crate) [C::BaseCRTField; D],
);

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> CyclotomicPolyRingNTTGeneral<C, N, D> {
    fn from_fn<F>(f: F) -> Self
    where
        F: FnMut(usize) -> C::BaseCRTField,
    {
        Self::from_array(core::array::from_fn::<_, D, _>(f))
    }

    pub(crate) fn from_array(ntt_coeffs: [C::BaseCRTField; D]) -> Self {
        Self(ntt_coeffs)
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> PartialEq
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> Eq
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> Clone
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> Copy
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> Debug
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> ark_std::fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> Display
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> ark_std::fmt::Result {
        write!(f, "CyclotomicPolyRingNTTGeneral(")?;
        let mut iter = self.0.iter();
        if let Some(first) = iter.next() {
            write!(f, "{}", first)?;
            for field_element in iter {
                write!(f, ", {}", field_element)?;
            }
        }
        write!(f, ")")
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> Hash
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn hash<H: ark_std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> Ring
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    const ZERO: Self = Self([<C::BaseCRTField as Field>::ZERO; D]);
    const ONE: Self = Self([<C::BaseCRTField as Field>::ONE; D]);
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> FromRandomBytes<Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn byte_size() -> usize {
        D * C::BaseCRTField::byte_size()
    }

    fn try_from_random_bytes(bytes: &[u8]) -> Option<Self> {
        assert_eq!(bytes.len(), Self::byte_size());

        let coeffs = core::array::from_fn(|i| {
            C::BaseCRTField::try_from_random_bytes(
                &bytes[i * C::BaseCRTField::byte_size()..(i + 1) * C::BaseCRTField::byte_size()],
            )
            .unwrap()
        });
        Some(Self::from_array(coeffs))
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> Default
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    #[inline(always)]
    fn default() -> Self {
        Self::zero()
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> Zero
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.eq(&Self::ZERO)
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> One
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    #[inline(always)]
    fn one() -> Self {
        Self::ONE
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> Mul<Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self.0.iter_mut().zip(rhs.0).for_each(|(lhs, rhs)| {
            if lhs.is_zero() || rhs.is_zero() {
                *lhs = C::BaseCRTField::zero();
            } else {
                *lhs *= rhs;
            }
        });

        self
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> MulUnchecked<Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    type Output = Self;

    fn mul_unchecked(mut self, rhs: Self) -> Self::Output {
        self.0.iter_mut().zip(rhs.0).for_each(|(lhs, rhs)| {
            *lhs *= rhs;
        });

        self
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> Neg
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.0.iter_mut().for_each(|x| {
            *x = x.neg();
        });

        self
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> UniformRand
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self::from_fn(|_| C::BaseCRTField::rand(rng))
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> MulAssign<Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn mul_assign(&mut self, rhs: Self) {
        self.0.iter_mut().zip(rhs.0).for_each(|(lhs, rhs)| {
            if lhs.is_zero() || rhs.is_zero() {
                *lhs = C::BaseCRTField::zero();
            } else {
                *lhs *= rhs;
            }
        });
    }
}

impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> Add<&'a Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    type Output = Self;

    fn add(mut self, rhs: &'a Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0)
            .for_each(|(lhs, rhs)| *lhs += rhs);

        self
    }
}

impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> Sub<&'a Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    type Output = Self;

    fn sub(mut self, rhs: &'a Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0)
            .for_each(|(lhs, rhs)| *lhs -= rhs);

        self
    }
}

impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> Add<&'a mut Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    type Output = Self;

    fn add(mut self, rhs: &'a mut Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0)
            .for_each(|(lhs, rhs)| *lhs += rhs);

        self
    }
}

impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> Sub<&'a mut Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    type Output = Self;

    fn sub(mut self, rhs: &'a mut Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0)
            .for_each(|(lhs, rhs)| *lhs -= rhs);

        self
    }
}

impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> Mul<&'a mut Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    type Output = Self;

    fn mul(mut self, rhs: &'a mut Self) -> Self::Output {
        self.0.iter_mut().zip(rhs.0).for_each(|(lhs, rhs)| {
            if lhs.is_zero() || rhs.is_zero() {
                *lhs = C::BaseCRTField::zero();
            } else {
                *lhs *= rhs;
            }
        });

        self
    }
}

impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> MulUnchecked<&'a mut Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    type Output = Self;

    fn mul_unchecked(mut self, rhs: &'a mut Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0)
            .for_each(|(lhs, rhs)| *lhs *= rhs);

        self
    }
}

impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> AddAssign<&'a mut Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn add_assign(&mut self, rhs: &'a mut Self) {
        self.0
            .iter_mut()
            .zip(rhs.0)
            .for_each(|(lhs, rhs)| *lhs += rhs);
    }
}

impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> SubAssign<&'a mut Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn sub_assign(&mut self, rhs: &'a mut Self) {
        self.0
            .iter_mut()
            .zip(rhs.0)
            .for_each(|(lhs, rhs)| *lhs -= rhs);
    }
}

impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> MulAssign<&'a mut Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn mul_assign(&mut self, rhs: &'a mut Self) {
        self.0.iter_mut().zip(rhs.0).for_each(|(lhs, rhs)| {
            if lhs.is_zero() || rhs.is_zero() {
                *lhs = C::BaseCRTField::zero();
            } else {
                *lhs *= rhs;
            }
        });
    }
}

macro_rules! impl_from_primitive_type {
    ($primitive_type: ty) => {
        impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> From<$primitive_type>
            for CyclotomicPolyRingNTTGeneral<C, N, D>
        {
            fn from(value: $primitive_type) -> Self {
                Self::from_scalar(C::BaseCRTField::from_base_prime_field(Fp::<
                    C::BaseFieldConfig,
                    N,
                >::from(
                    value
                )))
            }
        }
    };
}

macro_rules! impl_add_mul_primitive_type {
    ($primitive_type: ty) => {
        impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> Mul<$primitive_type>
            for CyclotomicPolyRingNTTGeneral<C, N, D>
        {
            type Output = Self;

            fn mul(mut self, rhs: $primitive_type) -> Self::Output {
                let r = C::BaseCRTField::from(rhs);
                self.0.iter_mut().for_each(|lhs| *lhs *= r);

                self
            }
        }

        impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> Mul<$primitive_type>
            for &'a CyclotomicPolyRingNTTGeneral<C, N, D>
        {
            type Output = CyclotomicPolyRingNTTGeneral<C, N, D>;

            fn mul(self, rhs: $primitive_type) -> Self::Output {
                *self * rhs
            }
        }

        impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> MulAssign<$primitive_type>
            for CyclotomicPolyRingNTTGeneral<C, N, D>
        {
            fn mul_assign(&mut self, rhs: $primitive_type) {
                let r = C::BaseCRTField::from(rhs);
                self.0.iter_mut().for_each(|lhs| *lhs *= r);
            }
        }

        impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize>
            MulAssign<&'a $primitive_type> for CyclotomicPolyRingNTTGeneral<C, N, D>
        {
            fn mul_assign(&mut self, rhs: &'a $primitive_type) {
                let r = C::BaseCRTField::from(*rhs);
                self.0.iter_mut().for_each(|lhs| *lhs *= r);
            }
        }
        impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> MulAssign<$primitive_type>
            for &'a mut CyclotomicPolyRingNTTGeneral<C, N, D>
        {
            fn mul_assign(&mut self, rhs: $primitive_type) {
                let r = C::BaseCRTField::from(rhs);
                self.0.iter_mut().for_each(|lhs| *lhs *= r);
            }
        }

        impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> Add<$primitive_type>
            for CyclotomicPolyRingNTTGeneral<C, N, D>
        {
            type Output = Self;

            fn add(mut self, rhs: $primitive_type) -> Self::Output {
                let r = C::BaseCRTField::from(rhs);
                self.0.iter_mut().for_each(|lhs| *lhs += r);

                self
            }
        }

        impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> Add<$primitive_type>
            for &'a CyclotomicPolyRingNTTGeneral<C, N, D>
        {
            type Output = CyclotomicPolyRingNTTGeneral<C, N, D>;

            fn add(self, rhs: $primitive_type) -> Self::Output {
                *self + rhs
            }
        }
        impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> AddAssign<$primitive_type>
            for CyclotomicPolyRingNTTGeneral<C, N, D>
        {
            fn add_assign(&mut self, rhs: $primitive_type) {
                let r = C::BaseCRTField::from(rhs);
                self.0.iter_mut().for_each(|lhs| *lhs += r);
            }
        }

        impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> AddAssign<$primitive_type>
            for &'a mut CyclotomicPolyRingNTTGeneral<C, N, D>
        {
            fn add_assign(&mut self, rhs: $primitive_type) {
                let r = C::BaseCRTField::from(rhs);
                self.0.iter_mut().for_each(|lhs| *lhs += r);
            }
        }

        impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> Sub<$primitive_type>
            for CyclotomicPolyRingNTTGeneral<C, N, D>
        {
            type Output = Self;

            fn sub(mut self, rhs: $primitive_type) -> Self::Output {
                let r = C::BaseCRTField::from(rhs);
                self.0.iter_mut().for_each(|lhs| *lhs -= r);

                self
            }
        }

        impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> Sub<$primitive_type>
            for &'a CyclotomicPolyRingNTTGeneral<C, N, D>
        {
            type Output = CyclotomicPolyRingNTTGeneral<C, N, D>;

            fn sub(self, rhs: $primitive_type) -> Self::Output {
                *self - rhs
            }
        }

        impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> SubAssign<$primitive_type>
            for CyclotomicPolyRingNTTGeneral<C, N, D>
        {
            fn sub_assign(&mut self, rhs: $primitive_type) {
                let r = C::BaseCRTField::from(rhs);
                self.0.iter_mut().for_each(|lhs| *lhs -= r);
            }
        }

        impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> SubAssign<$primitive_type>
            for &'a mut CyclotomicPolyRingNTTGeneral<C, N, D>
        {
            fn sub_assign(&mut self, rhs: $primitive_type) {
                let r = C::BaseCRTField::from(rhs);
                self.0.iter_mut().for_each(|lhs| *lhs -= r);
            }
        }
    };
}
// only works for types that can be cast to Field
impl_add_mul_primitive_type!(u128);
impl_add_mul_primitive_type!(u64);
impl_add_mul_primitive_type!(u32);
impl_add_mul_primitive_type!(u16);
impl_add_mul_primitive_type!(u8);
impl_add_mul_primitive_type!(bool);

impl_from_primitive_type!(u128);
impl_from_primitive_type!(u64);
impl_from_primitive_type!(u32);
impl_from_primitive_type!(u16);
impl_from_primitive_type!(u8);
impl_from_primitive_type!(bool);

impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> Mul<&'a Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    type Output = Self;

    fn mul(mut self, rhs: &'a Self) -> Self::Output {
        self.0.iter_mut().zip(rhs.0).for_each(|(lhs, rhs)| {
            if lhs.is_zero() || rhs.is_zero() {
                *lhs = C::BaseCRTField::zero();
            } else {
                *lhs *= rhs;
            }
        });
        self
    }
}

impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> MulUnchecked<&'a Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    type Output = Self;

    fn mul_unchecked(mut self, rhs: &'a Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0)
            .for_each(|(lhs, rhs)| *lhs *= rhs);
        self
    }
}

impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> AddAssign<&'a Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn add_assign(&mut self, rhs: &'a Self) {
        self.0
            .iter_mut()
            .zip(rhs.0)
            .for_each(|(lhs, rhs)| *lhs += rhs);
    }
}

impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> SubAssign<&'a Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn sub_assign(&mut self, rhs: &'a Self) {
        self.0
            .iter_mut()
            .zip(rhs.0)
            .for_each(|(lhs, rhs)| *lhs -= rhs);
    }
}

impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> MulAssign<&'a Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn mul_assign(&mut self, rhs: &'a Self) {
        self.0.iter_mut().zip(rhs.0).for_each(|(lhs, rhs)| {
            if lhs.is_zero() || rhs.is_zero() {
                *lhs = C::BaseCRTField::zero();
            } else {
                *lhs *= rhs;
            }
        });
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> Add<Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0)
            .for_each(|(lhs, rhs)| *lhs += rhs);

        self
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> Sub<Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self.0
            .iter_mut()
            .zip(rhs.0)
            .for_each(|(lhs, rhs)| *lhs -= rhs);

        self
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> AddAssign<Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn add_assign(&mut self, rhs: Self) {
        self.0
            .iter_mut()
            .zip(rhs.0)
            .for_each(|(lhs, rhs)| *lhs += rhs);
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> SubAssign<Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn sub_assign(&mut self, rhs: Self) {
        self.0
            .iter_mut()
            .zip(rhs.0)
            .for_each(|(lhs, rhs)| *lhs -= rhs);
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> Sum<Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> Sum<&'a Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> Product<Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a, C: CyclotomicConfig<N>, const N: usize, const D: usize> Product<&'a Self>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> PolyRing
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    type BaseRing = C::BaseCRTField;

    fn coeffs(&self) -> &[C::BaseCRTField] {
        &self.0
    }

    fn coeffs_mut(&mut self) -> &mut [Self::BaseRing] {
        &mut self.0
    }

    fn dimension() -> usize {
        D
    }

    fn from_scalar(v: Self::BaseRing) -> Self {
        // NTT([v, 0, ..., 0]) = ([v, ..., v])
        Self::from_array([v; D])
    }

    fn into_coeffs(self) -> Vec<Self::BaseRing> {
        self.0.into()
    }
}

impl<C: CyclotomicConfig<N>, const N: usize, const D: usize> From<Vec<C::BaseCRTField>>
    for CyclotomicPolyRingNTTGeneral<C, N, D>
{
    fn from(value: Vec<C::BaseCRTField>) -> Self {
        Self(value.try_into().expect("Should be of correct length"))
    }
}
