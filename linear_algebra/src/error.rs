use thiserror::Error;

#[derive(Debug, Error)]
pub enum AlgebraError {
    /// Fail due to operations on structures of unexpected differing lengths.
    #[error("Unexpected different lengths: {0} and {1}")]
    DifferentLengths(usize, usize),
}
