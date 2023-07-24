pub fn quantize_to_bitdepth(x: f32, bitdepth: u8) -> f32 {
    quantize_f32(x, 2_u32.pow(bitdepth as u32 - 1) - 1)
}

/// A simple linear quantization of a floating-point number `x` within a range of [-1.0, 1.0] by projecting the number onto a range of integers [-`n_half`, `n_half`]
/// 
/// Note
/// ====
/// For quantizing a 32-bit floating point number to an `n`-bit floating point number, set `n_half` to be 
/// `n_half = 2^(n-1) - 1`
pub fn quantize_f32(x: f32, n_half: u32) -> f32 {
    (x * n_half as f32).round() / n_half as f32
}