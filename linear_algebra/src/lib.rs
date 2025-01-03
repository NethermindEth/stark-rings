#![cfg_attr(not(feature = "std"), no_std)]
#![feature(trait_alias)]
#![allow(incomplete_features)]
#![feature(inherent_associated_types)]

#[macro_use]
extern crate ark_std;

mod generic_matrix;
mod matrix;
pub mod ops;
pub mod serialization;
pub mod sparse_matrix;
mod symmetric_matrix;
mod vector;

pub type Vector<T> = vector::Vector<T>;
pub type SVector<T, const N: usize> = vector::SVector<T, N>;
pub type RowVector<T> = vector::RowVector<T>;
pub type Matrix<T> = matrix::Matrix<T>;
pub type SymmetricMatrix<T> = symmetric_matrix::SymmetricMatrix<T>;
pub type SparseMatrix<T> = sparse_matrix::SparseMatrix<T>;
pub trait Scalar = nalgebra::Scalar;
