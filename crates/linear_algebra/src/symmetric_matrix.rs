use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::{
    io::{Read, Write},
    ops::{Index, IndexMut},
    rand,
    vec::*,
    UniformRand, Zero,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct SymmetricMatrix<F: Clone>(Vec<Vec<F>>);

impl<F: Clone> From<Vec<Vec<F>>> for SymmetricMatrix<F> {
    fn from(value: Vec<Vec<F>>) -> Self {
        assert!(value.iter().enumerate().all(|(i, v_i)| v_i.len() == i + 1), "cannot convert value: Vec<Vec<F>> to SymmetricMatrix<F>, row has wrong number of entries");
        Self(value)
    }
}

impl<F: Zero + Clone> SymmetricMatrix<F> {
    pub fn zero(n: usize) -> SymmetricMatrix<F> {
        SymmetricMatrix::<F>((0..n).map(|i| vec![F::zero(); i + 1]).collect())
    }
}

impl<F: Clone> SymmetricMatrix<F> {
    #[inline]
    pub fn size(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn at(&self, i: usize, j: usize) -> &F {
        debug_assert!(i < self.0.len() && j < self.0.len());
        if j <= i {
            &self.0[i][j]
        } else {
            &self.0[j][i]
        }
    }

    #[inline]
    pub fn at_mut(&mut self, i: usize, j: usize) -> &mut F {
        debug_assert!(i < self.0.len() && j < self.0.len());
        if j <= i {
            &mut self.0[i][j]
        } else {
            &mut self.0[j][i]
        }
    }

    pub fn diag(&self) -> Vec<F> {
        (0..self.size()).map(|i| self.at(i, i).clone()).collect()
    }

    pub fn rows(&self) -> &Vec<Vec<F>> {
        &self.0
    }

    pub fn map<T, M>(&self, func: M) -> SymmetricMatrix<T>
    where
        T: Clone,
        M: Fn(&F) -> T,
    {
        SymmetricMatrix::<T>::from(
            self.rows()
                .iter()
                .map(|row| row.iter().map(&func).collect())
                .collect::<Vec<Vec<T>>>(),
        )
    }

    #[cfg(feature = "parallel")]
    pub fn from_par_fn<Func>(size: usize, func: Func) -> Self
    where
        F: Send + Sync,
        Func: Send + Sync + Fn(usize, usize) -> F,
    {
        Self::from(
            (0..size)
                .into_par_iter()
                .map(|i| (0..i + 1).into_par_iter().map(|j| func(i, j)).collect())
                .collect::<Vec<Vec<F>>>(),
        )
    }
}

impl<F: Clone> Index<(usize, usize)> for SymmetricMatrix<F> {
    type Output = F;

    fn index(&self, index: (usize, usize)) -> &F {
        self.at(index.0, index.1)
    }
}

impl<F: Clone> IndexMut<(usize, usize)> for SymmetricMatrix<F> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut F {
        self.at_mut(index.0, index.1)
    }
}

impl<F: Clone + UniformRand> SymmetricMatrix<F> {
    pub fn rand<Rng: rand::Rng + ?Sized>(n: usize, rng: &mut Rng) -> SymmetricMatrix<F> {
        SymmetricMatrix::<F>(
            (0..n)
                .map(|i| (0..i + 1).map(|_| F::rand(rng)).collect())
                .collect(),
        )
    }
}

impl<F: Clone> CanonicalSerialize for SymmetricMatrix<F>
where
    Vec<Vec<F>>: CanonicalSerialize,
{
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.0.serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.0.serialized_size(compress)
    }
}

impl<F: Clone> Valid for SymmetricMatrix<F>
where
    Vec<Vec<F>>: CanonicalDeserialize,
{
    fn check(&self) -> Result<(), SerializationError> {
        self.0.check()
    }
}

impl<F: Clone> CanonicalDeserialize for SymmetricMatrix<F>
where
    Vec<Vec<F>>: CanonicalDeserialize,
{
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Vec::<Vec<F>>::deserialize_with_mode(reader, compress, validate).map(Self)
    }
}
