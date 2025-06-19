#![cfg_attr(not(feature = "std"), no_std)]
#![feature(trait_alias)]
#![allow(incomplete_features)]
#![feature(inherent_associated_types)]

#[macro_use]
extern crate ark_std;

pub mod ops;
pub mod sparse_matrix;
mod symmetric_matrix;

pub type SymmetricMatrix<T> = symmetric_matrix::SymmetricMatrix<T>;
pub type SparseMatrix<T> = sparse_matrix::SparseMatrix<T>;
