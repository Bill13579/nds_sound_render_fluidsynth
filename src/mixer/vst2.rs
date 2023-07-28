use core::panic;
use std::{path::Path, sync::{Arc, Mutex}};

use fft_sound_convolution::StereoFilter;
use vst::{self, host::{Host, PluginLoader, PluginInstance, HostBuffer}, prelude::{Plugin, PluginParameters}};

use super::{FX, IsModulatable};

pub struct SimpleHost;
impl Host for SimpleHost {  }

pub struct SimpleVst2 {
    host: Arc<Mutex<SimpleHost>>,
    loader: PluginLoader<SimpleHost>,
    instance: PluginInstance,
    sample_rate: f32,
    host_buffer: HostBuffer<f32>
}
unsafe impl Send for SimpleVst2 {  }
impl SimpleVst2 {
    pub fn new<P: AsRef<Path>>(path: P, sample_rate: f32) -> Result<SimpleVst2, Box<dyn std::error::Error>> {
        let host = Arc::new(Mutex::new(SimpleHost));
        let mut loader = PluginLoader::load(path.as_ref(), host.clone())?;
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
        instance.set_block_size(1);
        instance.resume();

        Ok(SimpleVst2 { host, loader, instance, sample_rate, host_buffer: HostBuffer::new(2, 2) })
    }
    pub fn get_parameter_object(&mut self) -> Arc<dyn PluginParameters> {
        self.instance.get_parameter_object()
    }
}
impl StereoFilter for SimpleVst2 {
    fn clear(&mut self) {
        panic!("`clear` has not been implemented for `SimpleVst2`! Recreate the instance instead if this is necessary!");
    }
    fn compute(&mut self, signal: (f64, f64)) -> (f64, f64) {
        let mut outputs = vec![vec![0.0]; 2];
        let mut audio_buffer = self.host_buffer.bind(&[[signal.0 as f32], [signal.1 as f32]], &mut outputs);
        self.instance.process(&mut audio_buffer);
        (outputs[0][0] as f64, outputs[1][0] as f64)
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

