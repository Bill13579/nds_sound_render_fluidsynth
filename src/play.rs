/// A wrapper around the library of a simple stateful MIDI-file player

use fft_sound_convolution::{StereoFFTConvolution, StereoFilter};
use tinyaudio::{OutputDeviceParameters, run_output_device};

use crate::{*, math::quantize_to_bitdepth};
use std::{path::{Path, PathBuf}, sync::mpsc};

pub struct Player {

}

/// TODO: Removed creation of FluidPlayer from here
/// TODO: Removed audio context creation
/// TODO: Removed IR
/// TODO: Removed check for player status
/// TODO: Need to implement everything commented out down below
pub struct SynthCore {
    settings: Arc<FluidSettings>,
    synth: Arc<FluidSynth>,
    bitdepth: u8
}
impl SynthCore {
    pub fn new(channels: fluid_int, bitdepth: u8, sample_rate: u32, interp: fluid_int) -> Result<SynthCore, Box<dyn std::error::Error>> {
        let settings = Arc::new(FluidSettings::new());

        unsafe {
            fluid_settings_setstr(settings.get(), fluid_str!("audio.sample-format"), fluid_str!("16bits")).fluid_result("Failed to modify settings!")?;
            fluid_settings_setint(settings.get(), fluid_str!("player.reset-synth"), false as fluid_int).fluid_result("Failed to modify settings!")?;
            fluid_settings_setstr(settings.get(), fluid_str!("player.timing-source"), fluid_str!("sample")).fluid_result("Failed to modify settings!")?;
            fluid_settings_setint(settings.get(), fluid_str!("synth.audio-channels"), channels).fluid_result("Failed to modify settings!")?;
            fluid_settings_setint(settings.get(), fluid_str!("synth.audio-groups"), channels).fluid_result("Failed to modify settings!")?;
            fluid_settings_setnum(settings.get(), fluid_str!("synth.sample-rate"), sample_rate as f64).fluid_result("Failed to modify settings!")?;
        }
            //     // Settings of note:
        //     //  audio.periods - Determines latency
        //     //  audio.period-size - Determines latency
        //     //  audio.sample-format - Determines format of samples, float or 16-bit ints
        //     //  audio.file.* - Rendering to files
        //     //  player.reset-synth - Affects looping
        //     //  player.timing-source - Related to timing
        //     //  synth.audio-channels - Number of stereo channels outputted (multi-channel audio instead of usual stereo)
        //     //  synth.audio-groups - Number of output channels on the sound card
        //     //  synth.chorus.active, synth.chorus.* - On or off on chorus, chorus settings
        //     //  synth.cpu-cores - Multi-threading
        //     //  synth.midi-channels - Synth number of midi channels
        //     //  synth.midi-bank-select - How the synth reacts to Bank Select messages
        //     //  synth.min-note-length - Set a minimum note duration so that shorter notes have a better chance of sounding right
        //     //  synth.overflow.* - ???
        //     //  synth.polyphony - Polyphony
        //     //  synth.reverb.active, synth.reverb.* - On or off on reverb, reverb settings
        //     //  synth.sample-rate - Synth sample rate, should match audio device
        //     //  synth.verbose - Verbose

        let synth = Arc::new(FluidSynth::new(settings.clone()).ok_or("Failed to create synth!")?);

        unsafe {
            fluid_synth_set_interp_method(synth.get(), -1, interp as fluid_int).fluid_result("Failed to set sample interpolation method!")?;
        }

        Ok(SynthCore {
            settings,
            synth,
            bitdepth
        })
    }
    pub fn sfload(&self, sound_font: &PathBuf) -> Result<fluid_int, Box<dyn std::error::Error>> {
        Ok(unsafe {
            fluid_synth_sfload(self.synth.get(), fluid_str!(sound_font.display().to_string()), true as fluid_int)
        }.fluid_result_ret("Failed to load SoundFont!")?)
    }
    pub fn render(&self, buffer_size: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut out = Mixer::new(self.synth.count_audio_channels().try_into()?); out.init_len(buffer_size);
        let mut fx = Mixer::new((self.synth.count_effects_channels() * self.synth.count_effects_groups()).try_into()?); fx.init_len(buffer_size);

        self.synth.process(&mut out, &mut fx)?;
        let master = out.mix();
        let [left, right] = master.audio_buffers();

        let master_samples: Vec<f32> = left.iter().zip(right.iter()).map(|(&l, &r)| [l, r])
            .flatten()
            .map(|x| quantize_to_bitdepth(x, self.bitdepth)) // Quantization
            .collect();

        Ok(master_samples)
    }
}

// pub fn render<P: AsRef<Path>>(sound_font: &PathBuf, input_file_path: P, output_file_path: P, buffer_size: usize, channels: fluid_int, bitdepth: u8, sample_rate: u32, interp: fluid_int, repeat: fluid_int, play: bool, irpath: Option<PathBuf>, outgain: Option<f32>, irgain: Option<f32>) -> Result<(), Box<dyn std::error::Error>> {
//     unsafe {
//         fluid_player_add(player.get(), fluid_str!(input_file_path.as_ref().display().to_string())).fluid_result(&format!("Failed to load midi file \"{}\"!", input_file_path.as_ref().display()))?;
//         fluid_player_set_loop(player.get(), repeat).fluid_result("Failed to set loop!")?;
//         fluid_player_play(player.get()).fluid_result("Failed to start the MIDI player!")?;
//     }

//     // Setup the audio output.
//     let params = OutputDeviceParameters {
//         channels_count: 2,
//         sample_rate: sample_rate as usize,
//         channel_sample_count: buffer_size,
//     };
//     let (tx, rx) = mpsc::sync_channel::<OutputAudio>(0);
//     // Continue to setup the audio output.
//     let _device = run_output_device(
//         params,
//         move |data| {
//             if play {
//                 match rx.recv() {
//                     Ok(OutputAudio::CHUNK(chunk)) => {
//                         for (i, val) in chunk.into_iter().enumerate() {
//                             if let Some(out) = data.get_mut(i) {
//                                 *out = val;
//                             } else {
//                                 break;
//                             }
//                         }
//                     },
//                     Ok(OutputAudio::DONE) | Err(_) => {},
//                 }
//             }
//         },
//     )
//     .unwrap();


//     Ok(())
// }