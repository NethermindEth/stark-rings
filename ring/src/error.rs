use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConversionError {
    #[error("Failed to convert to integer")]
    ToInteger,
    #[error("Integer overflow")]
    Overflow,
}
