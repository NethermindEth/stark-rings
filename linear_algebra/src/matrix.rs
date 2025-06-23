use crate::AlgebraError;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::{
    io::{Read, Write},
    iter::Sum,
    ops::{Mul, MulAssign},
    rand::Rng,
    vec::*,
    UniformRand, Zero,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Matrix<R> {
    pub nrows: usize,
    pub ncols: usize,
    pub vals: Vec<Vec<R>>,
}

impl<R> Matrix<R> {
    pub fn empty() -> Self {
        Self {
            nrows: 0,
            ncols: 0,
            vals: vec![],
        }
    }
}

impl<R: Clone + Zero> Matrix<R> {
    pub fn zero(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            vals: vec![vec![R::zero(); ncols]; nrows],
        }
    }

    pub fn pad_rows(&mut self, new_size: usize) {
        if new_size > self.nrows {
            self.nrows = new_size;
            self.vals.resize(new_size, vec![R::zero(); self.ncols]);
        }
    }

    pub fn pad_cols(&mut self, new_size: usize) {
        if new_size > self.ncols {
            self.ncols = new_size;
            self.vals
                .iter_mut()
                .for_each(|row| row.resize(new_size, R::zero()));
        }
    }
}

impl<R: Clone + UniformRand + Zero> Matrix<R> {
    pub fn rand<RND: Rng>(rng: &mut RND, nrows: usize, ncols: usize) -> Self {
        let vals = (0..nrows)
            .map(|_| (0..ncols).map(|_| R::rand(rng)).collect::<Vec<R>>())
            .collect::<Vec<Vec<R>>>();
        Self { nrows, ncols, vals }
    }
}

impl<R> From<Vec<Vec<R>>> for Matrix<R> {
    fn from(vecs: Vec<Vec<R>>) -> Matrix<R> {
        Self {
            nrows: vecs.len(),
            ncols: vecs.first().unwrap_or(&vec![]).len(),
            vals: vecs,
        }
    }
}

impl<R: CanonicalSerialize> CanonicalSerialize for Matrix<R> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.vals.serialize_with_mode(&mut writer, compress)?;
        Ok(())
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.vals.serialized_size(compress)
    }
}

impl<R: CanonicalDeserialize> Valid for Matrix<R> {
    fn check(&self) -> Result<(), SerializationError> {
        Vec::<Vec<R>>::check(&self.vals)
    }
}

impl<R: CanonicalDeserialize> CanonicalDeserialize for Matrix<R> {
    fn deserialize_with_mode<Re: Read>(
        mut reader: Re,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let vals = Vec::<Vec<R>>::deserialize_with_mode(&mut reader, compress, validate)?;
        Ok(Self {
            nrows: vals.len(),
            ncols: vals.first().unwrap_or(&vec![]).len(),
            vals,
        })
    }
}

impl<R: Clone + for<'a> Mul<&'a R, Output = R> + Sum> Matrix<R> {
    pub fn checked_mul_vec(&self, v: &[R]) -> Option<Vec<R>> {
        if self.ncols != v.len() {
            return None;
        }

        Some(
            cfg_iter!(self.vals)
                .map(|row| row.iter().zip(v).map(|(r_m, r_v)| r_m.clone() * r_v).sum())
                .collect(),
        )
    }

    pub fn try_mul_vec(&self, v: &[R]) -> Result<Vec<R>, AlgebraError> {
        self.checked_mul_vec(v)
            .ok_or(AlgebraError::DifferentLengths(self.ncols, v.len()))
    }
}

impl<R: Clone + for<'a> Mul<&'a R, Output = R> + Sum> Mul<&[R]> for &Matrix<R> {
    type Output = Vec<R>;

    fn mul(self, v: &[R]) -> Vec<R> {
        self.try_mul_vec(v).unwrap()
    }
}

impl<R: for<'a> MulAssign<&'a R>> MulAssign<&R> for Matrix<R> {
    fn mul_assign(&mut self, r: &R) {
        cfg_iter_mut!(self.vals).for_each(|row| row.iter_mut().for_each(|r_m| *r_m *= r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_matrix() -> Matrix<u32> {
        vec![vec![0, 2, 0], vec![0, 0, 0], vec![1, 4, 3]].into()
    }

    #[test]
    fn test_matrix_mul_vec() {
        let m = sample_matrix();
        let v = vec![1, 2, 3];

        let result = m.try_mul_vec(&v).unwrap();
        assert_eq!(result, vec![4, 0, 18]);

        let badv = vec![1, 2];
        let result = m.try_mul_vec(&badv);
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_mul_element() {
        let mut m = sample_matrix();
        let r = 3u32;

        m *= &r;
        assert_eq!(m, vec![vec![0, 6, 0], vec![0, 0, 0], vec![3, 12, 9]].into())
    }
}
