use ark_ff::{Field, Fp, FpConfig};
use ark_std::vec::*;

use crate::Ring;

/// The trait for describing cyclotomic ring parameters.
/// It is used to specify:
/// * The field of coefficients of the cyclotomic ring.
/// * The field of CRT-components of the ring.
/// * Implementation of CRT/iCRT and reduction modulo the cyclotomic polynomial.
pub trait CyclotomicConfig<const N: usize>: Send + Sync + 'static + Sized {
    /// The base prime field configuration of the underlying polynomial ring.
    type BaseFieldConfig: FpConfig<N>;
    /// The field of the CRT components.
    type BaseCRTField: Field<BasePrimeField = Fp<Self::BaseFieldConfig, N>> + Ring;

    /// The field of the CRT components is a field extension of the base field.
    /// This constant specifies the degree.
    const CRT_FIELD_EXTENSION_DEGREE: usize;

    /// Given coefficients of a polynomial of degree 2 * phi(D)
    /// reduces it mod Phi_D(X).
    fn reduce_in_place(coefficients: &mut Vec<Fp<Self::BaseFieldConfig, N>>);

    /// Computes the evaluations of the polynomial with
    /// coefficients `coefficients` on the primitive Dth roots of unity.
    fn crt_in_place(coefficients: &mut [Fp<Self::BaseFieldConfig, N>]);

    fn crt(coefficients: Vec<Fp<Self::BaseFieldConfig, N>>) -> Vec<Self::BaseCRTField>;
    fn icrt(evaluations: Vec<Self::BaseCRTField>) -> Vec<Fp<Self::BaseFieldConfig, N>>;

    /// Given primitive Dth root of unity evaluations of a polynomial
    /// computes its coefficient form.
    fn icrt_in_place(evaluations: &mut [Fp<Self::BaseFieldConfig, N>]);
}
