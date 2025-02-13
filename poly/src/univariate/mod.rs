// Copyright 2022 arkworks contributors
// This file is part of the arkworks/algebra library.

// Adapted for rings by Nethermind

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    fmt::Debug,
    hash::Hash,
    ops::{Add, AddAssign, Neg, SubAssign},
    vec::*,
    Zero,
};
use rand::Rng;
use stark_rings::Ring;

mod dense;

/// Describes the common interface for univariate and multivariate polynomials
#[allow(clippy::trait_duplication_in_bounds)]
pub trait Polynomial<Rn: Ring>:
    Sized
    + Clone
    + Debug
    + Hash
    + PartialEq
    + Eq
    + Add
    + Neg
    + Zero
    + CanonicalSerialize
    + CanonicalDeserialize
    + for<'a> AddAssign<&'a Self>
    + for<'a> AddAssign<(Rn, &'a Self)>
    + for<'a> SubAssign<&'a Self>
{
    /// The type of evaluation points for this polynomial.
    type Point: Sized + Clone + Debug + Sync + Hash; // Do we need Ord trait here?

    /// Returns the total degree of the polynomial
    fn degree(&self) -> usize;

    /// Evaluates `self` at the given `point` in `Self::Point`.
    fn evaluate(&self, point: &Self::Point) -> Rn;
}

/// Describes the interface for univariate polynomials
pub trait DenseUVPolynomial<Rn: Ring>: Polynomial<Rn, Point = Rn> {
    /// Constructs a new polynomial from a list of coefficients.
    fn from_coefficients_slice(coeffs: &[Rn]) -> Self;

    /// Constructs a new polynomial from a list of coefficients.
    fn from_coefficients_vec(coeffs: Vec<Rn>) -> Self;

    /// Returns the coefficients of `self`
    fn coeffs(&self) -> &[Rn];

    /// Returns a univariate polynomial of degree `d` where each
    /// coefficient is sampled uniformly at random.
    fn rand<R: Rng>(d: usize, rng: &mut R) -> Self;
}
