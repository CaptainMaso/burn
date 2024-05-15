// #[cfg(feature = "nightly-f16")]
// mod impl_nightly {
//     pub use core::f16;
//     pub use core::f16::consts;

//     pub use half::bf16;
// }

// #[cfg(not(feature = "nightly-f16"))]
// mod impl_half {
//     pub use half::bf16;
//     pub use half::f16;
// }

// #[cfg(feature = "nightly-f16")]
// pub use impl_nightly::*;

// #[cfg(not(feature = "nightly-f16"))]
// pub use impl_half::*;
pub use half::bf16;
pub use half::f16;
