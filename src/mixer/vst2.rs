use core::panic;
use std::{path::Path, sync::{Arc, Mutex}};

use fft_sound_convolution::dtype::RingBuffer;
use lazy_static::lazy_static;
use vst::{self, host::{Host, PluginLoader, PluginInstance, HostBuffer}, prelude::{Plugin, PluginParameters}};

use super::{FX, IsModulatable, SoundProcessor, StereoChannel};

pub struct SimpleHost;
impl Host for SimpleHost {  }

lazy_static! {
    static ref HOST: Arc<Mutex<SimpleHost>> = Arc::new(Mutex::new(SimpleHost));
}

pub struct SimpleVst2 {
    loader: PluginLoader<SimpleHost>,
    instance: PluginInstance,
    sample_rate: f32,
    block_size: i64,
    host_buffer: HostBuffer<f32>
}
unsafe impl Send for SimpleVst2 {  }
impl SimpleVst2 {
    pub fn new<P: AsRef<Path>>(path: P, sample_rate: f32, block_size: i64) -> Result<SimpleVst2, Box<dyn std::error::Error>> {
        let mut loader = PluginLoader::load(path.as_ref(), HOST.clone())?;
        let mut instance = loader.instance()?;

        let info = instance.get_info();

        println!(
            "===== VST2 Loaded =====\n\t\
            Loaded '{}':\n\t\
            Vendor: {}\n\t\
            Presets: {}\n\t\
            Preset Chunks: {}\n\t\
            Parameters: {}\n\t\
            VST ID: {}\n\t\
            Version: {}\n\t\
            Initial Delay: {} samples",
            info.name, info.vendor, info.presets, info.preset_chunks, info.parameters, info.unique_id, info.version, info.initial_delay
        );
        
        instance.init();
        instance.set_sample_rate(sample_rate);
        instance.set_block_size(block_size);
        instance.resume();

        Ok(SimpleVst2 { loader, instance, sample_rate, block_size, host_buffer: HostBuffer::new(2, 2) })
    }
    pub fn get_parameter_object(&mut self) -> Arc<dyn PluginParameters> {
        self.instance.get_parameter_object()
    }
}
impl SoundProcessor for SimpleVst2 {
    fn clear(&mut self) {
        panic!("`clear` has not been implemented for `SimpleVst2`! Recreate the instance instead if this is necessary!");
    }
    // fn compute(&mut self, signal: (f64, f64)) -> (f64, f64) {
    //     let mut outputs = vec![vec![0.0]; 2];
    //     let mut audio_buffer = self.host_buffer.bind(&[[signal.0 as f32], [signal.1 as f32]], &mut outputs);
    //     self.instance.process(&mut audio_buffer);
    //     (outputs[0][0] as f64, outputs[1][0] as f64)
    // }
    fn compute(&mut self, signal: &super::StereoChannel) -> super::StereoChannel {
        let mut out = StereoChannel::from_bufs(RingBuffer::new(signal.len()).initialize(0.0), RingBuffer::new(signal.len()).initialize(0.0));
        for ((in_l, in_r), (out_l, out_r)) in signal.l.audio_buffer().chunks(self.block_size as usize).zip(
            signal.r.audio_buffer().chunks(self.block_size as usize)).zip(
                out.l.audio_buffer_mut().chunks_mut(self.block_size as usize).zip(
                    out.r.audio_buffer_mut().chunks_mut(self.block_size as usize))) {
                        let mut audio_buffer = self.host_buffer.bind(&[in_l, in_r], &mut [out_l, out_r]);
                        self.instance.process(&mut audio_buffer);
        }
        out
    }
}
impl FX for SimpleVst2 {
    fn get_latency_samples(&self) -> usize {
        self.instance.get_info().initial_delay as usize
    }
    fn get_tail_samples(&self) -> usize {
        ((self.sample_rate * self.instance.get_tail_size() as f32).ceil() as usize).max(self.get_latency_samples())
    }
}
impl IsModulatable for SimpleVst2 {  }

