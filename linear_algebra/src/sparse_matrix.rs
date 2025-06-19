use crate::Matrix;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::{
    io::{Read, Write},
    rand::Rng,
    vec::*,
    UniformRand, Zero,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseMatrix<R> {
    pub nrows: usize,
    pub ncols: usize,
    pub coeffs: Vec<Vec<(R, usize)>>,
}

impl<R> SparseMatrix<R> {
    pub fn empty() -> Self {
        Self {
            nrows: 0,
            ncols: 0,
            coeffs: vec![],
        }
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn pad_rows(&mut self, new_size: usize) {
        if new_size > self.nrows() {
            self.nrows = new_size;
        }
    }

    pub fn pad_cols(&mut self, new_size: usize) {
        if new_size > self.ncols() {
            self.ncols = new_size;
        }
    }
}

impl<R: Clone + UniformRand + Zero> SparseMatrix<R> {
    /// Create a random sparse matrix with an approximate `sparsity` ratio of zeroes
    pub fn rand<RND: Rng>(rng: &mut RND, nrows: usize, ncols: usize, sparsity: f64) -> Self {
        let mut coeffs = Vec::with_capacity(nrows);

        for _ in 0..nrows {
            let mut row = Vec::new();
            for col in 0..ncols {
                if !rng.gen_bool(sparsity) {
                    row.push((R::rand(rng), col));
                }
            }
            coeffs.push(row);
        }

        Self {
            nrows,
            ncols,
            coeffs,
        }
    }
}

impl<R: Clone + Zero> SparseMatrix<R> {
    pub fn to_dense(&self) -> Matrix<R> {
        let mut s: Vec<Vec<R>> = vec![vec![R::zero(); self.ncols]; self.nrows];
        for (row_i, row) in self.coeffs.iter().enumerate() {
            for (value, col_i) in row.iter() {
                s[row_i][*col_i] = value.clone();
            }
        }
        s.into()
    }

    pub fn from_dense<T: Clone + Into<R> + Zero>(m: &Matrix<T>) -> Self {
        let mut s = SparseMatrix::<R> {
            nrows: m.vals.len(),
            ncols: m.vals.first().unwrap_or(&vec![]).len(),
            coeffs: Vec::new(),
        };
        for m_row in m.vals.iter() {
            let mut row: Vec<(R, usize)> = Vec::new();
            for (col_i, value) in m_row.iter().enumerate() {
                if !value.is_zero() {
                    row.push(((*value).clone().into(), col_i));
                }
            }
            s.coeffs.push(row);
        }
        s
    }
}

impl<R: CanonicalSerialize> CanonicalSerialize for SparseMatrix<R> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        let nrows = self.nrows() as u64;
        let ncols = self.ncols() as u64;
        nrows.serialize_with_mode(&mut writer, compress)?;
        ncols.serialize_with_mode(&mut writer, compress)?;
        self.coeffs.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        8 + 8 + self.coeffs.serialized_size(compress)
    }
}

impl<R: CanonicalDeserialize> Valid for SparseMatrix<R> {
    fn check(&self) -> Result<(), SerializationError> {
        Vec::<Vec<(R, usize)>>::check(&self.coeffs)
    }
}

impl<R: CanonicalDeserialize> CanonicalDeserialize for SparseMatrix<R> {
    fn deserialize_with_mode<Re: Read>(
        mut reader: Re,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let nrows = u64::deserialize_with_mode(&mut reader, compress, validate)? as usize;
        let ncols = u64::deserialize_with_mode(&mut reader, compress, validate)? as usize;
        let coeffs =
            Vec::<Vec<(R, usize)>>::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(Self {
            nrows,
            ncols,
            coeffs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_sparse() -> SparseMatrix<u32> {
        let coeffs = vec![vec![(2, 1usize)], vec![], vec![(1, 0), (4, 1), (3, 2)]];

        SparseMatrix {
            nrows: 3,
            ncols: 3,
            coeffs,
        }
    }

    fn sample_dense() -> Matrix<u32> {
        vec![vec![0, 2, 0], vec![0, 0, 0], vec![1, 4, 3]].into()
    }

    #[test]
    fn test_matrix_dense_to_sparse() {
        assert_eq!(SparseMatrix::from_dense(&sample_dense()), sample_sparse());
    }

    #[test]
    fn test_matrix_sparse_to_dense() {
        assert_eq!(sample_sparse().to_dense(), sample_dense());
    }
}
