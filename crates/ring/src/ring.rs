use crate::{traits::FromRandomBytes, ConversionError};
use ark_ff::{BigInteger, BitIteratorBE, BitIteratorLE, Field, FpConfig, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::UniformRand;
use crypto_primitives::{ark_ff_fp::Fp, ConstRing, IntRing};
use num_traits::{ConstOne, ConstZero};

pub trait StarkRing: 'static +
    ConstRing +
    Copy +
    From<u64> +
    From<u128> +
    UniformRand +
    // Differs from arkworks
    FromRandomBytes<Self> +
    CanonicalSerialize +
    CanonicalDeserialize
{
    /// Returns `sum([a_i * b_i])`.
    #[inline]
    fn sum_of_products<const T: usize>(a: &[Self; T], b: &[Self; T]) -> Self {
        let mut sum = Self::zero();
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }

    fn square_in_place(&mut self) -> &mut Self {
        *self *= *self;
        self
    }

    /// Returns `self^exp`, where `exp` is an integer represented with `u64` limbs,
    /// least significant limb first.
    #[must_use]
    fn pow_limbs<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        let mut res = Self::one();

        for i in BitIteratorBE::without_leading_zeros(exp) {
            res.square_in_place();

            if i {
                res *= self;
            }
        }
        res
    }

    /// Exponentiates a field element `f` by a number represented with `u64`
    /// limbs, using a precomputed table containing as many powers of 2 of
    /// `f` as the 1 + the floor of log2 of the exponent `exp`, starting
    /// from the 1st power. That is, `powers_of_2` should equal `&[p, p^2,
    /// p^4, ..., p^(2^n)]` when `exp` has at most `n` bits.
    ///
    /// This returns `None` when a power is missing from the table.
    #[inline]
    fn pow_with_table<S: AsRef<[u64]>>(powers_of_2: &[Self], exp: S) -> Option<Self> {
        let mut res = Self::one();
        for (pow, bit) in BitIteratorLE::without_trailing_zeros(exp).enumerate() {
            if bit {
                res *= powers_of_2.get(pow)?;
            }
        }
        Some(res)
    }
}

impl<C, const N: usize> StarkRing for Fp<C, N> where C: FpConfig<N> {}

impl<C: FpConfig<N>, const N: usize> FromRandomBytes<Fp<C, N>> for Fp<C, N> {
    fn byte_size() -> usize {
        Self::ZERO.uncompressed_size() + 9 // TODO: check if this is correct;
                                           // this is inferred from Fp<C,
                                           // N>::from_random_bytes()
    }

    fn try_from_random_bytes(bytes: &[u8]) -> Option<Self> {
        Self::from_random_bytes(bytes)
    }
}

/// Ring of integers
pub trait Zq: IntRing {
    /// Convert self to `u64`
    fn to_u64(&self) -> Result<u64, ConversionError>;
    /// Center self to around `(p-1)/2`
    ///
    /// Smaller values are kept the same, while larger values are shifted to
    /// `q-self`. The absolute value of the signed representation is used as
    /// output.
    fn center(&self) -> Self;
    /// Sign according to the center `(p-1)/2`
    ///
    /// Smaller values are mapped as 1, while larger values are mapped as -1
    /// (`q-1`).
    fn sign(&self) -> Self;

    fn to_usize(&self) -> Result<usize, ConversionError> {
        self.to_u64()?
            .try_into()
            .map_err(|_| ConversionError::Overflow)
    }
}

impl<C: FpConfig<N>, const N: usize> Zq for Fp<C, N> {
    fn to_u64(&self) -> Result<u64, ConversionError> {
        let bi = self.into_bigint();
        if bi > u64::MAX.into() {
            return Err(ConversionError::ToInteger);
        }
        Ok(bi.as_ref()[0])
    }

    fn center(&self) -> Self {
        let bi = self.into_bigint();
        if bi > Self::MODULUS_MINUS_ONE_DIV_TWO {
            let mut q = Self::MODULUS;
            q.sub_with_borrow(&bi);
            Self::from_bigint(q).unwrap()
        } else {
            *self
        }
    }

    fn sign(&self) -> Self {
        let bi = self.into_bigint();
        if bi > Self::MODULUS_MINUS_ONE_DIV_TWO {
            -<Self as ConstOne>::ONE
        } else if bi <= Self::MODULUS_MINUS_ONE_DIV_TWO {
            <Self as ConstOne>::ONE
        } else {
            <Self as ConstZero>::ZERO
        }
    }
}
