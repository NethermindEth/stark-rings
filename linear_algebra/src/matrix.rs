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
