use crate::{Matrix, SparseMatrix};
use ark_std::{
    ops::{Add, BitXor, Div, Sub},
    vec::*,
    Zero,
};
use num_integer::Integer;

pub trait Transpose {
    fn transpose(&self) -> Self;
}

impl<R: Zero + Clone> Transpose for Vec<Vec<R>> {
    fn transpose(&self) -> Self {
        let nrows = self.len();
        let ncols = self.iter().map(|d_i| d_i.len()).max().unwrap_or(0);

        let mut res: Vec<Vec<_>> = (0..ncols).map(|_| Vec::with_capacity(nrows)).collect();

        for row in self {
            // Copy existing values from original rows to new cols
            for (c, value) in row.iter().enumerate() {
                res[c].push(value.clone());
            }

            // Pad the shorter rows with zeroes
            for res_row in res.iter_mut().take(ncols).skip(row.len()) {
                res_row.push(R::zero());
            }
        }

        res
    }
}

impl<R: Zero + Clone> Transpose for Matrix<R> {
    fn transpose(&self) -> Self {
        Self {
            vals: self.vals.transpose(),
            nrows: self.nrows,
            ncols: self.ncols,
        }
    }
}

impl<R: Zero + Clone> Transpose for SparseMatrix<R> {
    fn transpose(&self) -> Self {
        let mut res: Vec<Vec<(R, usize)>> = vec![Vec::new(); self.ncols];

        for (row_idx, row) in self.coeffs.iter().enumerate() {
            for (value, col_idx) in row.iter() {
                res[*col_idx].push((value.clone(), row_idx));
            }
        }

        Self {
            coeffs: res,
            nrows: self.ncols,
            ncols: self.nrows,
        }
    }
}

pub fn rounded_div<T, D>(dividend: T, divisor: D) -> T
where
    T: Integer
        + BitXor<Output = T>
        + Add<D, Output = T>
        + Div<D, Output = T>
        + Sub<D, Output = T>
        + From<D>
        + Clone,
    D: Clone + Div<i128, Output = D>,
{
    if dividend.clone() ^ divisor.clone().into() >= T::zero() {
        (dividend + (divisor.clone() / 2)) / divisor
    } else {
        (dividend - (divisor.clone() / 2)) / divisor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SparseMatrix;

    #[test]
    #[rustfmt::skip]
    fn transpose_vec_of_vecs() {
        let v = vec![
            vec![1, 2, 3],
            vec![9],
            vec![7, 7, 7]
        ].transpose();

        #[rustfmt::skip]
        let r = vec![
            vec![1, 9, 7],
            vec![2, 0, 7],
            vec![3, 0, 7]
        ];

        assert_eq!(v, r);
    }

    fn sample_matrix() -> Matrix<u32> {
        vec![vec![0, 2, 0], vec![0, 0, 0], vec![1, 4, 3]].into()
    }

    fn sample_matrix_transposed() -> Matrix<u32> {
        vec![vec![0, 0, 1], vec![2, 0, 4], vec![0, 0, 3]].into()
    }

    #[test]
    fn test_transpose_matrix() {
        let m = sample_matrix();

        let transposed = m.transpose();

        assert_eq!(transposed, sample_matrix_transposed());
    }

    #[test]
    fn test_transpose_sparse_matrix() {
        let m: SparseMatrix<u32> = SparseMatrix::from_dense(&sample_matrix());

        let transposed = m.transpose();

        assert_eq!(
            transposed,
            SparseMatrix::from_dense(&sample_matrix_transposed())
        );
    }
}
