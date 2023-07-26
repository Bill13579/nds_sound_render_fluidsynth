/// A wrapper around the library of a simple stateful MIDI-file player

use fft_sound_convolution::{StereoFFTConvolution, StereoFilter};
use tinyaudio::{OutputDeviceParameters, run_output_device};

use crate::{*, math::quantize_to_bitdepth};
use std::{path::{Path, PathBuf}, sync::mpsc};

// TODO
pub struct AudioSystem {

}

pub struct SequencerSubsystem<'a> {
    synthcore: Arc<SynthCore>,
    sequencer: FluidSequencer<'a>
}
impl<'a> SequencerSubsystem<'a> {
    pub fn new(synthcore: Arc<SynthCore>) -> Option<SequencerSubsystem<'a>> {
        let mut sequencer = FluidSequencer::new()?;
        sequencer.register_fluidsynth(synthcore.synth.clone()); //NOTE: Ignoring return value, which is the client ID, since it's stored inside `FluidSequencer` anyways and automatically freed
        Some(SequencerSubsystem { synthcore, sequencer })
    }
    pub fn send(&self, midi_event: FluidMIDIEvent) -> Result<(), FluidError> {
        self.sequencer.add_midi_event_to_buffer(&midi_event)
    }
    pub fn send_percussive_note(&self, chan: fluid_int, key: fluid_int, velocity: fluid_int) -> Result<(), FluidError> {
        self.send_note_on(chan, key, velocity)?;
        self.send_note_off(chan, key, velocity)?;
        Ok(())
    }
    pub fn send_note_on(&self, chan: fluid_int, key: fluid_int, velocity: fluid_int) -> Result<(), FluidError> {
        let on = FluidMIDIEvent::create_note_on_event(chan, key, velocity).ok_or(FluidError::new("Failed to create note on event!"))?;
        self.send(on)?;
        Ok(())
    }
    pub fn send_note_off(&self, chan: fluid_int, key: fluid_int, velocity: fluid_int) -> Result<(), FluidError> {
        let off = FluidMIDIEvent::create_note_off_event(chan, key, velocity).ok_or(FluidError::new("Failed to create note off event!"))?;
        self.send(off)?;
        Ok(())
    }
    pub fn send_control_change(&self, chan: fluid_int, cc: fluid_int, value: fluid_int) -> Result<(), FluidError> {
        let cc = FluidMIDIEvent::create_control_change_event(chan, cc, value).ok_or(FluidError::new("Failed to create control change event!"))?;
        self.send(cc)?;
        Ok(())
    }
    /// Sends a bank select control message on CC0
    pub fn send_bank_select(&self, chan: fluid_int, bank: fluid_int) -> Result<(), FluidError> {
        self.send_control_change(chan, 0, bank)?;
        Ok(())
    }
    pub fn send_program_change(&self, chan: fluid_int, program: fluid_int) -> Result<(), FluidError> {
        let pc = FluidMIDIEvent::create_program_change_event(chan, program).ok_or(FluidError::new("Failed to create program change event!"))?;
        self.send(pc)?;
        Ok(())
    }
    /// Sends a bank select message and a program change message at the same time to switch patches
    pub fn send_patch_select(&self, chan: fluid_int, bank: fluid_int, program: fluid_int) -> Result<(), FluidError> {
        self.send_bank_select(chan, bank)?;
        self.send_program_change(chan, program)?;
        Ok(())
    }
}

pub struct FilePlayerSubsystem {
    synthcore: Arc<SynthCore>,
    player: FluidPlayer
}
#[repr(i32)]
pub enum PlayerStatus {
    Ready = fluid_player_status_FLUID_PLAYER_READY,
    Playing = fluid_player_status_FLUID_PLAYER_PLAYING,
    Stopping = fluid_player_status_FLUID_PLAYER_STOPPING,
    Done = fluid_player_status_FLUID_PLAYER_DONE
}
impl FilePlayerSubsystem {
    pub fn new(synthcore: Arc<SynthCore>) -> Option<FilePlayerSubsystem> {
        let player = FluidPlayer::new(synthcore.synth.clone())?;
        Some(FilePlayerSubsystem { synthcore, player })
    }
    pub fn set_loop(&self, play_loop: bool) -> Result<(), FluidError> {
        unsafe {
            fluid_player_set_loop(self.player.get(), play_loop as fluid_int).fluid_result("Failed to set loop!")
        }
    }
    pub fn add<P: AsRef<Path>>(&self, midi_file_path: P) -> Result<(), Box<dyn std::error::Error>> {
        unsafe {
            fluid_player_add(self.player.get(), fluid_str!(midi_file_path.as_ref().display().to_string())).fluid_result(&format!("Failed to load midi file \"{}\"!", midi_file_path.as_ref().display()))?;
        }
        Ok(())
    }
    pub fn play(&self) -> Result<(), FluidError> {
        unsafe {
            fluid_player_play(self.player.get()).fluid_result("Failed to start the MIDI player!")
        }
    }
    pub fn get_status(&self) -> PlayerStatus {
        let fluid_status;
        unsafe {
            fluid_status = fluid_player_get_status(self.player.get());
        }
        match fluid_status {
            fluid_player_status_FLUID_PLAYER_READY => PlayerStatus::Ready,
            fluid_player_status_FLUID_PLAYER_PLAYING => PlayerStatus::Playing,
            fluid_player_status_FLUID_PLAYER_STOPPING => PlayerStatus::Stopping,
            fluid_player_status_FLUID_PLAYER_DONE => PlayerStatus::Done,
            _ => panic!("Unknown `fluid_player_status` value! Has the API changed?")
        }
    }
}

/// TODO: Removed creation of FluidPlayer from here
/// TODO: Removed audio context creation
/// TODO: Removed IR -DONE
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
    pub fn render(&self, buffer_size: usize) -> Result<mixer::RawMixerData, Box<dyn std::error::Error>> {
        let mut out = mixer::RawMixerData::new(self.synth.count_audio_channels().try_into()?); out.init_len(buffer_size);
        let mut fx = mixer::RawMixerData::new((self.synth.count_effects_channels() * self.synth.count_effects_groups()).try_into()?); fx.init_len(buffer_size);

        self.synth.process(&mut out, &mut fx)?;

        Ok(out)
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



// let master = out.mix();
// let [left, right] = master.audio_buffers();

// let master_samples: Vec<f32> = left.iter().zip(right.iter()).map(|(&l, &r)| [l, r])
//     .flatten()
//     .map(|x| quantize_to_bitdepth(x, self.bitdepth)) // Quantization
//     .collect();

//     Ok(())
// }