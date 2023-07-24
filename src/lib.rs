#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::{ptr::NonNull, ffi::CString};

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Macro to quickly convert any strings and string-likes into an owned CString and then to a C-style string pointer
#[macro_export]
macro_rules! fluid_str {
    ($l:ident) => {
        CString::new($l)?.as_ptr()
    };
    ($l:expr) => {
        CString::new($l)?.as_ptr()
    };
}
/// Convenient alias for `std::os::raw::c_int`
pub type fluid_int = std::os::raw::c_int;
/// NULL
pub const fluid_null: fluid_int = 0;

/// Generic Error to represent a variety of errors emitted by FluidSynth using `FLUID_OK`/`FLUID_FAILED`
#[derive(Debug, Clone)]
pub struct FluidError(String);
impl FluidError {
    pub fn new(message: &str) -> FluidError {
        FluidError(String::from(message))
    }
}
impl std::fmt::Display for FluidError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &self.0)
    }
}
impl std::error::Error for FluidError {  }
/// Generic Error to represent a variety of errors emitted by the mixer
#[derive(Debug, Clone)]
pub struct MixerError(String);
impl MixerError {
    pub fn new(message: &str) -> MixerError {
        MixerError(String::from(message))
    }
}
impl std::fmt::Display for MixerError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &self.0)
    }
}
impl std::error::Error for MixerError {  }
/// Trait defined for `fluid_int` as a quick way to convert `FLUID_OK`/`FLUID_FAILED` into Rust-style results
pub trait FluidStatus {
    fn fluid_result(self, message: &str) -> Result<(), FluidError>;
    fn fluid_expect(self, message: &str);
}
impl FluidStatus for fluid_int {
    /// Convert `FLUID_OK`/`FLUID_FAILED` into a Rust-style `Result`
    fn fluid_result(self, message: &str) -> Result<(), FluidError> {
        if self == FLUID_OK.try_into().expect("`interpret_fluid_status`, unexpected error 1") {
            Ok(())
        } else if self == FLUID_FAILED.try_into().expect("`interpret_fluid_status`, unexpected error 2") {
            Err(FluidError::new(message))
        } else {
            Err(FluidError::new("`interpret_fluid_status`, unexpected error 3"))
        }
    }
    /// Convert `FLUID_OK`/`FLUID_FAILED` into a Rust-style `Result` and then call `unwrap` on it
    fn fluid_expect(self, message: &str) {
        self.fluid_result(message).unwrap()
    }
}

/// An *unsafe* wrapper of `fluid_settings_t` that
///  simply automatically calls `delete_fluid_settings`
///  upon being dropped. Usage of the internal `settings`
///  object thus still requires unsafe code from fluid.
pub struct FluidSettings {
    settings: *mut fluid_settings_t
}
impl FluidSettings {
    pub fn new() -> FluidSettings {
        unsafe {
            FluidSettings { settings: new_fluid_settings() }
        }
    }
    pub fn get(&self) -> *mut fluid_settings_t {
        self.settings
    }
}
impl Drop for FluidSettings {
    fn drop(&mut self) {
        unsafe {
            delete_fluid_settings(self.get());
        }
    }
}

/// An *unsafe* wrapper of `fluid_synth_t` that
///  simply automatically calls `delete_fluid_synth`
///  upon being dropped. Usage of the internal `synth`
///  object thus still requires unsafe code from fluid.
pub struct FluidSynth<'a> {
    _settings: &'a FluidSettings,
    synth: *mut fluid_synth_t
}
// impl<'a> FluidSynth<'a> {
//     pub fn sfload(&self, filename: &str, reset_presets: bool) -> Result<(), Box<dyn std::error::Error>> {
//         let filename = fluid_str!(filename);
//         unsafe {
//             fluid_synth_sfload(self.get(), filename, reset_presets as i32).fluid_result("failed to load soundfont!")?;
//         }
//         Ok(())
//     }
//     pub fn program_reset(&self) -> Result<(), Box<dyn std::error::Error>> {
//         unsafe {
//             fluid_synth_program_reset(self.get()).fluid_result("failed to reset assigned programs!")?;
//         }
//         Ok(())
//     }
//     pub fn get_cc(&self, chan: i32, ctrl: i32, pval: &mut i32) {
//         unsafe {
//             fluid_synth_get_cc(self.get(), chan, ctrl, pval as *mut i32).fluid_expect("failed to get cc");
//         }
//     }
// }
impl<'a> FluidSynth<'a> {
    pub fn new(settings: &'a FluidSettings) -> Option<FluidSynth> {
        unsafe {
            Some(
                FluidSynth { _settings: settings, synth: NonNull::new(new_fluid_synth(settings.get())).map(|nonnull| nonnull.as_ptr())? }
            )
        }
    }
    pub fn get(&self) -> *mut fluid_synth_t {
        self.synth
    }

    pub fn count_effects_channels(&self) -> fluid_int {
        unsafe {
            fluid_synth_count_effects_channels(self.get())
        }
    }
    pub fn count_effects_groups(&self) -> fluid_int {
        unsafe {
            fluid_synth_count_effects_groups(self.get())
        }
    }
    pub fn count_audio_channels(&self) -> fluid_int {
        unsafe {
            fluid_synth_count_audio_channels(self.get())
        }
    }

    pub fn process(&self, out: &mut Mixer, fx: &mut Mixer) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            fluid_synth_process(
                self.get(),
                out.len().ok_or("Mixer has buffer of size 0")? as fluid_int,
                // 4,
                fx.num_channels() as fluid_int * 2,
                fx.fluid_audio_buffers()[..].as_mut_ptr(),
                // 32,
                out.num_channels() as fluid_int * 2,
                out.fluid_audio_buffers()[..].as_mut_ptr()).fluid_result("Failed to render!")?;
        }
        Ok(())
    }
}
impl<'a> Drop for FluidSynth<'a> {
    fn drop(&mut self) {
        unsafe {
            delete_fluid_synth(self.get());
        }
    }
}

pub struct Channel {
    buf: Vec<f32>
}
impl Channel {
    pub fn new() -> Channel {
        Channel { buf: Vec::new() }
    }
    pub fn from_buf(buf: Vec<f32>) -> Channel {
        Channel { buf }
    }
    pub fn len(&self) -> usize {
        self.buf.len()
    }
    pub fn zero(&mut self) {
        for val in &mut self.buf { *val = 0.0; }
    }
    pub fn init(&mut self, len: usize) {
        self.buf.clear();
        self.buf.resize(len, 0.0);
    }
    pub fn audio_buffer(&self) -> &[f32] {
        &self.buf
    }
    pub fn audio_buffer_mut(&mut self) -> &mut [f32] {
        &mut self.buf
    }
    pub fn add(&mut self, other: &Channel) {
        for (i, &val) in other.audio_buffer().iter().enumerate() {
            if let Some(selfval) = self.audio_buffer_mut().get_mut(i) {
                *selfval += val;
            } else {
                break;
            }
        }
    }
    /// As these are raw pointers to the underlying buffers, there are **no Rust safety guarantees**!!!
    /// It is the programmer's responsibility to make sure that the underlying buffers are not dropped
    ///  before and during rendering.
    pub fn raw_audio_buffer_mut(&mut self) -> *mut f32 {
        self.audio_buffer_mut().as_mut_ptr()
    }
}
pub struct StereoChannel {
    l: Channel,
    r: Channel
}
impl StereoChannel {
    pub fn new() -> StereoChannel {
        StereoChannel { l: Channel::new(), r: Channel::new() }
    }
    pub fn from_bufs(bufl: Vec<f32>, bufr: Vec<f32>) -> StereoChannel {
        StereoChannel { l: Channel::from_buf(bufl), r: Channel::from_buf(bufr) }
    }
    pub fn len(&self) -> usize {
        self.l.len()  // self.r should return the same thing
    }
    pub fn zero(&mut self) {
        for val in self.l.audio_buffer_mut() { *val = 0.0; }
        for val in self.r.audio_buffer_mut() { *val = 0.0; }
    }
    pub fn init(&mut self, len: usize) {
        self.l.init(len);
        self.r.init(len);
    }
    pub fn audio_buffers(&self) -> [&[f32]; 2] {
        [self.l.audio_buffer(), self.r.audio_buffer()]
    }
    pub fn audio_buffers_mut(&mut self) -> [&mut [f32]; 2] {
        [self.l.audio_buffer_mut(), self.r.audio_buffer_mut()]
    }
    pub fn add(&mut self, other: &StereoChannel, gain: Option<f32>) {
        let [ol, or] = other.audio_buffers();
        let [sl, sr] = self.audio_buffers_mut();
        for (i, (&lval, &rval)) in ol.iter().zip(or.iter()).enumerate() {
            if let Some(selfval) = sl.get_mut(i) {
                *selfval += lval * gain.unwrap_or(1.0);
                sr[i] += rval * gain.unwrap_or(1.0);
            } else {
                break;
            }
        }
    }
    /// As these are raw pointers to the underlying buffers, there are **no Rust safety guarantees**!!!
    /// It is the programmer's responsibility to make sure that the underlying buffers are not dropped
    ///  before and during rendering.
    pub fn raw_audio_buffers_mut(&mut self) -> [*mut f32; 2] {
        [self.l.raw_audio_buffer_mut(), self.r.raw_audio_buffer_mut()]
    }
}
pub struct Mixer {
    channels: Vec<StereoChannel>
}
impl Mixer {
    pub fn new(num_out_channels: usize) -> Mixer {
        Mixer { channels: (0..num_out_channels).map(|_| StereoChannel::new()).collect() }
    }
    pub fn len(&self) -> Option<usize> {
        Some(self.channels.get(0)?.len())  // All the channels should return the same buffer length
    }
    pub fn zero(&mut self) {
        for channel in &mut self.channels { channel.zero(); }
    }
    pub fn num_channels(&self) -> usize {
        self.channels.len()
    }
    pub fn init_channels(&mut self, num_out_channels: usize) {
        self.channels.clear();
        self.channels.resize_with(num_out_channels, || StereoChannel::new());
    }
    pub fn init_len(&mut self, len: usize) {
        for channel in &mut self.channels {
            channel.init(len);
        }
    }
    pub fn mix(&self) -> StereoChannel {
        let mut master = StereoChannel::new();
        if let Some(len) = self.len() {
            master.init(len);
            for channel in &self.channels {
                master.add(channel, None);
            }
        }
        master
    }
    /// As these are raw pointers to the underlying buffers, there are **no Rust safety guarantees**!!!
    /// It is the programmer's responsibility to make sure that the underlying buffers are not dropped
    ///  before and during rendering.
    pub fn fluid_audio_buffers(&mut self) -> Vec<*mut f32> {
        self.channels.iter_mut().flat_map(|channel| channel.raw_audio_buffers_mut()).collect()
    }
}

/// An *unsafe* wrapper of `fluid_player_t` that
///  simply automatically calls `delete_fluid_player`
///  upon being dropped. Usage of the internal `player`
///  object thus still requires unsafe code from fluid.
pub struct FluidPlayer<'a> {
    _synth: &'a FluidSynth<'a>,
    player: *mut fluid_player_t
}
impl<'a> FluidPlayer<'a> {
    pub fn new(synth: &'a FluidSynth) -> Option<FluidPlayer<'a>> {
        unsafe {
            Some(
                FluidPlayer { _synth: synth, player: NonNull::new(new_fluid_player(synth.get())).map(|nonnull| nonnull.as_ptr())? }
            )
        }
    }
    pub fn get(&self) -> *mut fluid_player_t {
        self.player
    }
}
impl<'a> Drop for FluidPlayer<'a> {
    /// TODO:   WARNING FROM OFFICIAL FluidPlayer DOCS:  Do not call when the synthesizer associated to this player renders audio, i.e. an audio driver is running or any other synthesizer thread concurrently calls fluid_synth_process() or fluid_synth_nwrite_float() or fluid_synth_write_*() ! 
    fn drop(&mut self) {
        unsafe {
            delete_fluid_player(self.get());
        }
    }
}

pub mod math;
// pub mod play;

