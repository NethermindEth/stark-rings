# stark-rings - A Rust implementation of cyclotomic rings over the STARK-friendly rings.

A proof-of-concept implementation of rings of the form `Fp[X]/Phi(X)`, where `Phi(X)` is a cyclotomic polynomial and `p` is a STARK-friendly prime (Goldilocks, BabyBear or the Starknet prime). The choice of parameters for these ring is based on the work 
[Short, Invertible Elements in Partially Splitting Cyclotomic Rings and Applications to Lattice-Based Zero-Knowledge Proofs](https://eprint.iacr.org/2017/523) and targets large enough (~2^128) sets of short invertible elements.

In addition to that, the library contains crates for multilinear extensions over the rings and miscellaneous linear algebra tools.

**DISCLAIMER:** This is a proof-of-concept prototype, and in particular has not received careful code review. This implementation is provided "as is" and NOT ready for production use. Use at your own risk.

## Usage
Import the library:
```toml
[dependencies]
stark-rings = { git = "https://github.com/NethermindEth/stark-rings.git", package = "stark-rings"}
```

## Documentation
See the documentation with:
```bash
RUSTDOCFLAGS="--html-in-header $(pwd)/docs-header.html" cargo doc --open
```

## License
The crates in this repository are licensed under either of the following licenses, at your discretion.

* Apache License Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
* MIT license ([LICENSE-MIT](LICENSE-MIT))


Unless you explicitly state otherwise, any contribution submitted for inclusion in this library by you shall be dual licensed as above (as defined in the Apache v2 License), without any additional terms or conditions.

## Acknowledgments

- This library borrows a lot of ideas and code from [lattirust library](https://github.com/cknabs/lattirust) maintained by [Christian Knabenhans](https://github.com/cknabs).
- The stark-rings-poly crate is mostly word-to-word based on the corresponding crate from [arkworks library](https://github.com/arkworks-rs/algebra).
