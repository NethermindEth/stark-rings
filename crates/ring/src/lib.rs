#![cfg_attr(not(feature = "std"), no_std)]
#![allow(non_snake_case)]
// Exports
pub use balanced_decomposition::{
    representatives::{SignedRepresentative, UnsignedRepresentative},
    Decompose,
};
pub use error::*;
pub use monomial::*;
pub use poly_ring::*;
pub use ring::*;
pub use traits::*;

pub mod balanced_decomposition;
pub mod cyclotomic_ring;
pub mod traits;

mod error;
mod monomial;
mod poly_ring;
mod ring;
pub(crate) mod utils;

#[macro_use]
extern crate ark_std;

extern crate core;
