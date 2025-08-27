#![cfg_attr(not(feature = "std"), no_std)]
#![feature(trait_alias)]
#![allow(incomplete_features)]
#![feature(inherent_associated_types)]

#[macro_use]
extern crate ark_std;

mod error;
pub mod matrix;
pub mod ops;
pub mod sparse_matrix;
pub mod symmetric_matrix;

pub use error::AlgebraError;
pub use matrix::Matrix;
pub use sparse_matrix::SparseMatrix;
pub use symmetric_matrix::SymmetricMatrix;
