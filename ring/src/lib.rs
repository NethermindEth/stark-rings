#![cfg_attr(not(feature = "std"), no_std)]
#![allow(non_snake_case)]
#![feature(vec_into_raw_parts)]
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

#[macro_use]
extern crate ark_std;

extern crate core;
