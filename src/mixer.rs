use fft_sound_convolution::{StereoFilter, dtype::RingBuffer, Filter, StereoFFTConvolution, TrueStereoFFTConvolution, FFTConvolution};

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

pub struct Channel {
    buf: RingBuffer<f32>
}
impl Channel {
    pub fn new() -> Channel {
        Channel { buf: RingBuffer::new(0) }
    }
    pub fn from_buf(buf: RingBuffer<f32>) -> Channel {
        Channel { buf }
    }
    pub fn len(&self) -> usize {
        self.buf.len()
    }
    pub fn zero(&mut self) {
        for val in self.buf.inner_mut() { *val = 0.0; }
    }
    pub fn expand(&mut self, new_len: usize) -> Result<(), MixerError> {
        if self.len() > new_len {
            Err(MixerError::new("Cannot use `expand` to shrink the size of a channel's buffer!"))
        } else if self.len() == new_len { Ok(()) } else {
            self.buf.to_capacity_back(Some(new_len));
            self.buf.fill_back(0.0);
            Ok(())
        }
    }
    pub fn drain(&mut self, len: usize) -> Vec<f32> {
        let drained = self.buf.drain(..len);
        self.buf.fill_back(0.0);
        drained
    }
    pub fn drain_to_channel(&mut self, len: usize) -> Channel {
        Self::from_buf(RingBuffer::from(self.drain(len)))
    }
    pub fn init(&mut self, len: usize) {
        self.buf.to_capacity_back(Some(len));
        self.buf.initialize_again(0.0);
        self.buf.fill_back(0.0);
    }
    pub fn audio_buffer(&self) -> &[f32] {
        self.buf.inner()
    }
    pub fn audio_buffer_mut(&mut self) -> &mut [f32] {
        self.buf.inner_mut()
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
    pub fn from_channels(l: Channel, r: Channel) -> StereoChannel {
        StereoChannel { l, r }
    }
    pub fn from_bufs(bufl: RingBuffer<f32>, bufr: RingBuffer<f32>) -> StereoChannel {
        StereoChannel { l: Channel::from_buf(bufl), r: Channel::from_buf(bufr) }
    }
    pub fn len(&self) -> usize {
        self.l.len()  // self.r should return the same thing
    }
    pub fn zero(&mut self) {
        for val in self.l.audio_buffer_mut() { *val = 0.0; }
        for val in self.r.audio_buffer_mut() { *val = 0.0; }
    }
    pub fn expand(&mut self, new_len: usize) -> Result<(), MixerError> {
        self.l.expand(new_len)?;
        self.r.expand(new_len)?;
        Ok(())
    }
    pub fn drain(&mut self, len: usize) -> StereoChannel {
        StereoChannel::from_channels(self.l.drain_to_channel(len), self.r.drain_to_channel(len))
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

pub struct RawMixerData {
    channels: Vec<StereoChannel>
}
impl RawMixerData {
    pub fn new(num_out_channels: usize) -> RawMixerData {
        RawMixerData { channels: (0..num_out_channels).map(|_| StereoChannel::new()).collect() }
    }
    pub fn from_channels(channels: Vec<StereoChannel>) -> Result<RawMixerData, MixerError> {
        if !channels.is_empty() && !channels.iter().all(|x| x.len() == channels[0].len()) {
            Err(MixerError::new("Failed to create `RawMixerData` from the provided channels! Not all channels have the same internal buffer length!"))
        } else {
            Ok(RawMixerData { channels })
        }
    }
    pub fn len(&self) -> Option<usize> {
        Some(self.channels.get(0)?.len())  // All the channels should return the same buffer length
    }
    pub fn zero(&mut self) {
        for channel in &mut self.channels { channel.zero(); }
    }
    pub fn expand(&mut self, new_len: usize) -> Result<(), MixerError> {
        for channel in &mut self.channels { channel.expand(new_len)?; }
        Ok(())
    }
    pub fn drain(&mut self, len: usize) -> Result<RawMixerData, MixerError> {
        Self::from_channels(self.channels.iter_mut().map(|x| x.drain(len)).collect())
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
    pub fn add(&mut self, other: &RawMixerData, gain: Option<f32>) {
        for (target, source) in self.channels.iter_mut().zip(other.channels.iter()) {
            target.add(source, gain);
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

pub struct Mixer {
    internal_buffer: RawMixerData,
    latency: usize,
    latency_compensation_delays: Vec<StereoDelay>,
    drain_len: usize,
    fx: Vec<Vec<Box<dyn StereoFX>>>,
    buffer_size: usize,
    num_out_channels: usize
}
impl Mixer {
    pub fn new(buffer_size: usize, num_out_channels: usize, fx: Vec<Vec<Box<dyn StereoFX>>>) -> Mixer {
        let mut internal_buffer = RawMixerData::new(num_out_channels);
        let drain_len = Self::calculate_max_drain_len(&fx).unwrap_or(0);
        internal_buffer.init_len(buffer_size + drain_len);
        let latency = Self::calculate_max_latency(&fx).unwrap_or(0);
        let latency_compensation_delays = fx.iter().map(|x| StereoDelay::from_delay_samples(latency - Self::calculate_fx_chain_latency(x.iter()))).collect();
        Mixer { internal_buffer, latency, latency_compensation_delays, drain_len, fx, buffer_size, num_out_channels }
    }
    /// Must be called whenever the effects are modified
    pub fn update_fx(&mut self) {
        self.drain_len = Self::calculate_max_drain_len(&self.fx).unwrap_or(0);
        let _: Result<(), _> = self.internal_buffer.expand(self.buffer_size + self.drain_len);
        self.latency = Self::calculate_max_latency(&self.fx).unwrap_or(0);
        for (channel_i, fx_chain) in self.fx.iter().enumerate() {
            self.latency_compensation_delays[channel_i].resize(self.latency - Self::calculate_fx_chain_latency(fx_chain.iter()));
        }
    }
    pub fn add(&mut self, chunk: &RawMixerData) {
        for (i, (internal, source)) in self.internal_buffer.channels.iter_mut().zip(chunk.channels.iter()).enumerate() {
            for ((&l, &r), (internal_l, internal_r)) in source.l.audio_buffer().iter().zip(source.r.audio_buffer())
            .zip(internal.l.audio_buffer_mut().iter_mut().zip(internal.r.audio_buffer_mut().iter_mut())) {
                let mut pair = self.latency_compensation_delays[i].compute((l as f64, r as f64));
                for effect in self.fx[i].iter_mut() {
                    pair = effect.compute(pair);
                }
                *internal_l += pair.0 as f32;
                *internal_r += pair.1 as f32;
            }
        }
    }
    pub fn add_and_drain(&mut self, chunk: &RawMixerData) -> Result<RawMixerData, MixerError> {
        self.add(chunk);
        let num_samples = chunk.len().ok_or(MixerError::new("Could not obtain length of chunk! Chunk does not contain any channels."))?;
        self.internal_buffer.drain(num_samples)
    }
    pub fn render_late_fx(&mut self) {
        let mut empty_buffer = RawMixerData::new(self.num_out_channels);
        empty_buffer.init_len(self.drain_len);
        self.add(&empty_buffer);
    }
    pub fn drain_fx(&mut self) -> Result<RawMixerData, MixerError> {
        let mut empty_buffer = RawMixerData::new(self.num_out_channels);
        empty_buffer.init_len(self.drain_len);
        self.add_and_drain(&empty_buffer)
    }
    fn calculate_max_latency(fx: &Vec<Vec<Box<dyn StereoFX>>>) -> Option<usize> {
        fx.iter().map(|x| Self::calculate_fx_chain_latency(x.iter())).max()
    }
    fn calculate_max_drain_len(fx: &Vec<Vec<Box<dyn StereoFX>>>) -> Option<usize> {
        fx.iter().map(|x| Self::calculate_fx_chain_drain_len(x.iter())).max()
    }
    fn calculate_fx_chain_latency<'a, I>(fx_chain: I) -> usize
    where
        I: Iterator<Item = &'a Box<dyn StereoFX>> {
        fx_chain.fold(0, |acc, x| acc + x.get_latency_samples())
    }
    fn calculate_fx_chain_drain_len<'a, I>(fx_chain: I) -> usize
    where
        I: Iterator<Item = &'a Box<dyn StereoFX>> {
        fx_chain.fold(0, |acc, x| acc + x.get_latency_samples() + x.get_tail_samples())
    }
}

pub trait FX {
    fn get_latency_samples(&self) -> usize;
    fn get_tail_samples(&self) -> usize;
}
pub trait MonoFX: Filter + FX {  }
impl<T> MonoFX for T where T: Filter + FX {  }
pub trait StereoFX: StereoFilter + FX {  }
impl<T> StereoFX for T where T: StereoFilter + FX {  }

pub struct Delay {
    out: RingBuffer<f64>,
    delay: usize
}
impl Delay {
    pub fn new(delay: usize) -> Delay {
        Delay {
            out: RingBuffer::new(delay).initialize(0.0),
            delay
        }
    }
    pub fn resize(&mut self, new_delay: usize) {
        self.out.to_capacity_front(Some(new_delay));
        self.out.fill_front(0.0);
        self.delay = new_delay;
    }
}
impl Filter for Delay {
    fn clear(&mut self) {
        self.out.initialize_again(0.0);
    }
    fn compute(&mut self, signal: f64) -> f64 {
        let buffered_signal = self.out.pop_front().unwrap();
        self.out.push_back(signal);
        buffered_signal
    }
}
impl FX for Delay {
    fn get_latency_samples(&self) -> usize {
        0
    }
    fn get_tail_samples(&self) -> usize {
        self.delay
    }
}

pub struct StereoDelay {
    pub l: Delay,
    pub r: Delay
}
impl StereoDelay {
    pub fn new(l: Delay, r: Delay) -> StereoDelay {
        StereoDelay {
            l, r
        }
    }
    pub fn from_delay_samples(delay: usize) -> StereoDelay {
        StereoDelay {
            l: Delay::new(delay),
            r: Delay::new(delay)
        }
    }
    pub fn resize(&mut self, new_delay: usize) {
        self.l.resize(new_delay);
        self.r.resize(new_delay);
    }
}
impl StereoFilter for StereoDelay {
    fn clear(&mut self) {
        self.l.clear();
        self.r.clear();
    }
    fn compute(&mut self, signal: (f64, f64)) -> (f64, f64) {
        (self.l.compute(signal.0), self.r.compute(signal.1))
    }
}
impl FX for StereoDelay {
    fn get_latency_samples(&self) -> usize {
        0
    }
    fn get_tail_samples(&self) -> usize {
        self.l.get_tail_samples().max(self.r.get_tail_samples())
    }
}

impl FX for FFTConvolution {
    fn get_latency_samples(&self) -> usize {
        self.window_size()
    }
    fn get_tail_samples(&self) -> usize {
        self.internal_buffer_size() - self.window_size()
    }
}
impl FX for StereoFFTConvolution {
    fn get_latency_samples(&self) -> usize {
        self.window_size()
    }
    fn get_tail_samples(&self) -> usize {
        self.internal_buffer_size() - self.window_size()
    }
}
impl FX for TrueStereoFFTConvolution {
    fn get_latency_samples(&self) -> usize {
        self.window_size()
    }
    fn get_tail_samples(&self) -> usize {
        self.internal_buffer_size() - self.window_size()
    }
}

