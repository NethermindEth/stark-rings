use crate::{AlgebraError, Matrix};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::{
    cmp::Ordering,
    io::{Read, Write},
    iter::Sum,
    ops::{Mul, MulAssign},
    rand::Rng,
    vec::*,
    One, UniformRand, Zero,
};
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};

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
}

impl<R: Clone> SparseMatrix<R> {
    pub fn hconcat(ms: &[SparseMatrix<R>]) -> Option<Self> {
        if ms.is_empty() {
            return Some(Self::empty());
        }

        let nrows = ms[0].nrows;
        if !ms.iter().all(|m| m.nrows == nrows) {
            return None;
        }

        let ncols: usize = ms.iter().map(|m| m.ncols).sum();
        let mut coeffs = vec![vec![]; nrows];

        let mut offset = 0;
        for m in ms {
            for (row, crow) in m.coeffs.iter().zip(coeffs.iter_mut()) {
                let shifted_row: Vec<(R, usize)> = row
                    .iter()
                    .map(|(val, i)| (val.clone(), i + offset))
                    .collect();
                crow.extend(shifted_row);
            }
            offset += m.ncols;
        }

        Some(SparseMatrix {
            nrows,
            ncols,
            coeffs,
        })
    }

    pub fn pad_rows(&mut self, new_size: usize) {
        if new_size > self.nrows() {
            self.nrows = new_size;
            self.coeffs.resize(new_size, vec![]);
        }
    }

    pub fn pad_cols(&mut self, new_size: usize) {
        if new_size > self.ncols() {
            self.ncols = new_size;
        }
    }
}

impl<R: Clone + One> SparseMatrix<R> {
    /// Create a `n * n` identity matrix
    pub fn identity(n: usize) -> Self {
        let mut m = Self {
            nrows: n,
            ncols: n,
            coeffs: vec![vec![]; n],
        };
        for (i, row) in m.coeffs.iter_mut().enumerate() {
            row.push((R::one(), i));
        }
        m
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

impl<R: Clone + for<'a> Mul<&'a R, Output = R> + Sum + Send + Sync + Zero> SparseMatrix<R> {
    pub fn checked_mul_vec(&self, v: &[R]) -> Option<Vec<R>> {
        if self.ncols != v.len() {
            return None;
        }

        Some(
            cfg_iter!(self.coeffs)
                .map(|row| row.iter().map(|(r, i)| r.clone() * &v[*i]).sum())
                .collect(),
        )
    }

    pub fn try_mul_vec(&self, v: &[R]) -> Result<Vec<R>, AlgebraError> {
        self.checked_mul_vec(v)
            .ok_or(AlgebraError::DifferentLengths(self.ncols, v.len()))
    }

    pub fn checked_mul_mat(&self, m: &SparseMatrix<R>) -> Option<SparseMatrix<R>> {
        if self.ncols != m.nrows {
            return None;
        }

        let mut m_cols: Vec<Vec<(R, usize)>> = vec![Vec::new(); m.ncols];
        for (row_idx, row) in m.coeffs.iter().enumerate() {
            for (val, col_idx) in row.iter() {
                m_cols[*col_idx].push((val.clone(), row_idx));
            }
        }

        let coeffs = cfg_iter!(self.coeffs)
            .map(|row| {
                let mut res_row = Vec::new();
                for (j, col) in m_cols.iter().enumerate() {
                    // Compute dot product between this row and column j
                    let mut sum = None;
                    let mut row_iter = row.iter().peekable();
                    let mut col_iter = col.iter().peekable();
                    while let (Some(&(ref r_val, r_idx)), Some(&(ref c_val, c_idx))) =
                        (row_iter.peek(), col_iter.peek())
                    {
                        match r_idx.cmp(c_idx) {
                            Ordering::Less => {
                                row_iter.next();
                            }
                            Ordering::Greater => {
                                col_iter.next();
                            }
                            Ordering::Equal => {
                                let product = r_val.clone() * c_val;
                                if !product.is_zero() {
                                    sum = Some(match sum {
                                        Some(s) => s + product,
                                        None => product,
                                    });
                                }
                                row_iter.next();
                                col_iter.next();
                            }
                        }
                    }
                    if let Some(s) = sum {
                        res_row.push((s, j));
                    }
                }
                res_row
            })
            .collect::<Vec<Vec<(R, usize)>>>();

        Some(SparseMatrix {
            nrows: self.nrows,
            ncols: m.ncols,
            coeffs,
        })
    }

    pub fn try_mul_mat(&self, m: &SparseMatrix<R>) -> Result<SparseMatrix<R>, AlgebraError> {
        self.checked_mul_mat(m)
            .ok_or(AlgebraError::DifferentLengths(self.ncols, m.nrows))
    }
}

impl<R: Clone + for<'a> Mul<&'a R, Output = R> + Send + Sum + Sync + Zero> Mul<&[R]>
    for &SparseMatrix<R>
{
    type Output = Vec<R>;

    fn mul(self, v: &[R]) -> Vec<R> {
        self.try_mul_vec(v).unwrap()
    }
}

impl<R: Clone + for<'a> Mul<&'a R, Output = R> + Send + Sum + Sync + Zero> Mul<&SparseMatrix<R>>
    for &SparseMatrix<R>
{
    type Output = SparseMatrix<R>;

    fn mul(self, m: &SparseMatrix<R>) -> SparseMatrix<R> {
        self.try_mul_mat(m).unwrap()
    }
}

impl<R: for<'a> MulAssign<&'a R> + Send + Sync> MulAssign<&R> for SparseMatrix<R> {
    fn mul_assign(&mut self, r: &R) {
        cfg_iter_mut!(self.coeffs).for_each(|row| row.iter_mut().for_each(|(m_r, _)| *m_r *= r))
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
    fn test_sparse_matrix_identity() {
        let m = SparseMatrix::identity(2);
        let expected = SparseMatrix {
            nrows: 2,
            ncols: 2,
            coeffs: vec![vec![(1, 0)], vec![(1, 1)]],
        };
        assert_eq!(m, expected);
    }

    #[test]
    fn test_matrix_dense_to_sparse() {
        assert_eq!(SparseMatrix::from_dense(&sample_dense()), sample_sparse());
    }

    #[test]
    fn test_matrix_sparse_to_dense() {
        assert_eq!(sample_sparse().to_dense(), sample_dense());
    }

    #[test]
    fn test_sparse_matrix_mul_vec() {
        let m = sample_sparse();
        let v = vec![1, 2, 3];

        let result = m.try_mul_vec(&v).unwrap();
        assert_eq!(result, vec![4, 0, 18]);

        let badv = vec![1, 2];
        let result = m.try_mul_vec(&badv);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_matrix_mul_element() {
        let mut m = sample_sparse();
        let r = 3u32;

        m *= &r;
        assert_eq!(
            m.to_dense(),
            vec![vec![0, 6, 0], vec![0, 0, 0], vec![3, 12, 9]].into()
        )
    }

    #[test]
    fn test_sparse_matrix_mul_mat() {
        let m1 = sample_sparse();
        let m2 =
            SparseMatrix::from_dense(&Matrix::from(vec![vec![1u32, 2], vec![3, 4], vec![5, 6]]));

        let result = m1.try_mul_mat(&m2).unwrap();
        let expected =
            SparseMatrix::from_dense(&Matrix::from(vec![vec![6u32, 8], vec![0, 0], vec![28, 36]]));
        assert_eq!(result, expected);

        let m3 = SparseMatrix::from_dense(&Matrix::from(vec![vec![1u32, 2], vec![3, 4]]));
        let result = m1.try_mul_mat(&m3);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_matrix_hconcat() {
        let m1 = SparseMatrix::<i32>::from_dense(&Matrix::from(vec![vec![1, 2], vec![3, 4]]));
        let m2 = SparseMatrix::from_dense(&Matrix::from(vec![vec![5], vec![6]]));
        let m3 = SparseMatrix::from_dense(&Matrix::from(vec![vec![7, 8, 9], vec![10, 11, 12]]));

        let result = SparseMatrix::hconcat(&[m1.clone(), m2, m3]).unwrap();
        let expected = SparseMatrix::from_dense(&Matrix::from(vec![
            vec![1, 2, 5, 7, 8, 9],
            vec![3, 4, 6, 10, 11, 12],
        ]));
        assert_eq!(result, expected);

        let m4 = SparseMatrix::from_dense(&Matrix::from(vec![vec![1, 2, 3]]));
        let result = SparseMatrix::hconcat(&[m1, m4]);
        assert!(result.is_none());

        let result = SparseMatrix::<u32>::hconcat(&[]);
        assert_eq!(result.unwrap(), SparseMatrix::empty());
    }
}
