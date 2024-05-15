/// Contains the [`f16`] type
///
/// If nightly is enabled, uses the nightly built-in [`f16`]
///
/// [`f16`]: (https://doc.rust-lang.org/nightly/std/primitive.f16.html)
mod impl_f16;

pub use impl_f16::*;
