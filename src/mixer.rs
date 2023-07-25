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
pub struct RawMixerData {
    channels: Vec<StereoChannel>
}
impl RawMixerData {
    pub fn new(num_out_channels: usize) -> RawMixerData {
        RawMixerData { channels: (0..num_out_channels).map(|_| StereoChannel::new()).collect() }
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

pub struct Mixer {
    
}