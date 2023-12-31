/// A wrapper around the library of a simple stateful MIDI-file player

use tinyaudio::{OutputDeviceParameters, run_output_device, BaseAudioOutputDevice};

use crate::{*, math::quantize_to_bitdepth, mixer::{Mixer, DeadRawMixerData, RawMixerData}};
use std::{path::{Path, PathBuf}, collections::HashMap, sync::Mutex, error::Error, os::raw::c_void};

/// Generic Error to represent a variety of errors emitted by the audio system
#[derive(Debug, Clone)]
pub struct AudioSystemError(String);
impl AudioSystemError {
    pub fn new(message: &str) -> AudioSystemError {
        AudioSystemError(String::from(message))
    }
}
impl std::fmt::Display for AudioSystemError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &self.0)
    }
}
impl std::error::Error for AudioSystemError {  }

pub enum AudioSource {
    Sequencer(SequencerSubsystem),
    File(FilePlayerSubsystem)
}
impl AudioSource {
    /// Replenish output buffers with new samples. Returns a boolean representing if playback has finished.
    /// 
    /// Note
    /// ====
    /// As a MIDI sequencer never actually finishes playing, if the audio source is a sequencer, the returned boolean will always be `false`.
    pub fn replenish(&mut self, dest: &mut DeadRawMixerData) -> Result<bool, Box<dyn std::error::Error>> {
        match self {
            AudioSource::Sequencer(seq) => seq.replenish(dest).map(|_| false),
            AudioSource::File(file) => file.replenish(dest)
        }
    }
}
pub struct AudioSystem {
    sound_sources: HashMap<String, (DeadRawMixerData, Option<AudioSource>)>,
    sample_rate: usize,
    bitdepth: u8,
    buffer_size: usize,
}
impl AudioSystem {
    pub fn new(sample_rate: usize, bitdepth: u8, buffer_size: usize) -> Result<(Arc<Mutex<AudioSystem>>, Box<dyn BaseAudioOutputDevice>), Box<dyn std::error::Error>> {
        let audio_system = Arc::new(Mutex::new(AudioSystem {
            sound_sources: HashMap::new(),
            sample_rate, bitdepth, buffer_size
        }));
        let params = OutputDeviceParameters {
            channels_count: 2,
            sample_rate,
            channel_sample_count: buffer_size
        };
        let audio_system_copy = audio_system.clone();
        let device = run_output_device(params, move |data| {
            audio_system_copy.lock().unwrap().run(data).expect("Failed to render some samples!!");
        });
        Ok((audio_system, device?))
    }
    pub fn add(&mut self, name: String, audio_source: AudioSource) -> Option<(DeadRawMixerData, Option<AudioSource>)> {
        self.sound_sources.insert(name, (DeadRawMixerData::empty(), Some(audio_source)))
    }
    pub fn get(&self, name: &str) -> Option<&AudioSource> {
        self.sound_sources.get(name)?.1.as_ref()
    }
    pub fn get_mut(&mut self, name: &str) -> Option<&mut AudioSource> {
        self.sound_sources.get_mut(name)?.1.as_mut()
    }
    pub fn sound_sources(&self) -> &HashMap<String, (DeadRawMixerData, Option<AudioSource>)> {
        &self.sound_sources
    }
    pub fn sound_sources_mut(&mut self) -> &mut HashMap<String, (DeadRawMixerData, Option<AudioSource>)> {
        &mut self.sound_sources
    }
    /// Refill buffers
    fn refill_buffers(&mut self) -> Result<usize, Box<dyn std::error::Error>> {
        let mut results = Vec::with_capacity(self.sound_sources.len());
        let mut min_buffer_size = self.sound_sources.values().next().map(|(x, _)| x.len()).unwrap_or(0);
        self.sound_sources.retain(|_, (buf, sound_source)| {
            if buf.is_empty() {
                if let Some(sound_source_unwrap) = sound_source {
                    let is_playback_finished_result = sound_source_unwrap.replenish(buf);
                    let is_playback_finished = *is_playback_finished_result.as_ref().unwrap_or(&true);
                    results.push(is_playback_finished_result.map(|_| ()));
                    if is_playback_finished {
                        *sound_source = None;
                    }
                    if buf.len() < min_buffer_size {
                        min_buffer_size = buf.len();
                    }
                    return true;
                } else {
                    return false;
                }
            } else {
                if buf.len() < min_buffer_size {
                    min_buffer_size = buf.len();
                }
                return true;
            }
        });
        results.into_iter().try_for_each(|x| x)?;
        Ok(min_buffer_size)
    }
    /// Gather samples from all the buffers
    fn gather_and_interweave(&mut self, n: usize) -> Vec<f32> {
        let mut out: Vec<f32> = vec![0.0; n * 2];
        for (part, _) in self.sound_sources.values_mut() {
            let (part_l, part_r) = part.drain_n(n);
            for (stereo_pair, (pl, pr)) in out.chunks_mut(2).zip(part_l.zip(part_r)) {
                stereo_pair[0] += pl;
                stereo_pair[1] += pr;
            }
        }
        out
    }
    pub fn run(&mut self, data: &mut [f32]) -> Result<(), Box<dyn std::error::Error>> {
        let mut data_len = data.len() / 2;
        let mut min_buffer_size = self.refill_buffers()?;
        if min_buffer_size == 0 { return Ok(()); } // All sound sources have been exhausted, the remaining space in the data buffer can be left empty
        while data_len != 0 {
            let chunk_len = data_len.min(min_buffer_size);
            let chunk = self.gather_and_interweave(chunk_len);

            let left = data.len() - data_len * 2;
            (&mut data[left..(left + chunk_len * 2)]).copy_from_slice(&chunk);

            data_len -= chunk_len;
            min_buffer_size = self.refill_buffers()?;
            if min_buffer_size == 0 { return Ok(()); } // All sound sources have been exhausted, the remaining space in the data buffer can be left empty
        }
        Ok(())
    }
}

pub struct SequencerSubsystem {
    synthcore: Arc<SynthCore>,
    sequencer: FluidSequencer,
    channel_mixer: Mixer,
    master_mixer: Mixer
}
impl SequencerSubsystem {
    pub fn new(synthcore: Arc<SynthCore>, channel_mixer: Mixer, master_mixer: Mixer) -> Option<SequencerSubsystem> {
        let mut sequencer = FluidSequencer::new()?;
        sequencer.register_fluidsynth(synthcore.synth.clone()); //NOTE: Ignoring return value, which is the client ID, since it's stored inside `FluidSequencer` anyways and automatically freed
        Some(SequencerSubsystem { synthcore, sequencer, channel_mixer, master_mixer })
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
    /// Replenish output buffers with new samples.
    pub fn replenish(&mut self, dest: &mut DeadRawMixerData) -> Result<(), Box<dyn std::error::Error>> {
        let raw = self.synthcore.render(self.channel_mixer.buffer_size())?;
        let processed = RawMixerData::from_channels(vec![self.channel_mixer.add_and_drain(&raw)?.mix()])?;
        let processed = self.master_mixer.add_and_drain(&processed)?;
        *dest = processed.into();
        Ok(())
    }
}

pub struct FilePlayerSubsystem {
    synthcore: Arc<SynthCore>,
    player: FluidPlayer,
    address: Box<_PlayerAndSynthAddress>,
    channel_mixer: Mixer,
    master_mixer: Mixer
}
#[repr(i32)]
#[derive(PartialEq)]
pub enum PlayerStatus {
    Ready = fluid_player_status_FLUID_PLAYER_READY,
    Playing = fluid_player_status_FLUID_PLAYER_PLAYING,
    Stopping = fluid_player_status_FLUID_PLAYER_STOPPING,
    Done = fluid_player_status_FLUID_PLAYER_DONE
}
#[derive(Debug)]
struct _PlayerAndSynthAddress {
    player: *mut fluid_player_t,
    synth: *mut fluid_synth_t
}
unsafe impl Send for _PlayerAndSynthAddress {  }
impl FilePlayerSubsystem {
    #[no_mangle]
    extern "C" fn handle_midi_event(data: *mut c_void, event: *mut fluid_midi_event_t) -> fluid_int {
        let address;
        unsafe {
            address = &*(data as *const _PlayerAndSynthAddress);
        }
        let midi_event = FluidMIDIEvent::from_raw(event);
        if midi_event.get_type() == MIDIMetaEventType::MetaText as fluid_int {
            if let Ok(st) = midi_event.get_text_nul() {
                if st.starts_with("Jump ") {
                    if let Ok(jump) = st[5..].parse::<fluid_int>() {
                        println!("{}", jump);
                        unsafe {
                            fluid_player_seek(address.player, jump);
                        }
                    }
                }
            }
        }
        midi_event.into_raw(); // Release the midi event pointer. Otherwise the event will be dropped!
        unsafe {
            fluid_synth_handle_midi_event(address.synth as *mut c_void, event)
        }
    }
    pub fn new(synthcore: Arc<SynthCore>, channel_mixer: Mixer, master_mixer: Mixer) -> Option<FilePlayerSubsystem> {
        let player = FluidPlayer::new(synthcore.synth.clone())?;
        let address = Box::new(_PlayerAndSynthAddress { player: player.get(), synth: synthcore.synth.get() });
        unsafe {
            fluid_player_set_playback_callback(player.get(), Some(FilePlayerSubsystem::handle_midi_event), address.as_ref() as *const _PlayerAndSynthAddress as *mut c_void).fluid_result("").ok()?;
        }
        Some(FilePlayerSubsystem { address, synthcore, player, channel_mixer, master_mixer })
    }
    pub fn set_loop(&self, n: fluid_int) -> Result<(), FluidError> {
        unsafe {
            fluid_player_set_loop(self.player.get(), n).fluid_result("Failed to set loop!")
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
    /// Replenish output buffers with new samples. Returns a boolean representing
    ///  if playback has finished.
    pub fn replenish(&mut self, dest: &mut DeadRawMixerData) -> Result<bool, Box<dyn std::error::Error>> {
        if self.get_status() == PlayerStatus::Done {
            let processed = RawMixerData::from_channels(vec![self.channel_mixer.drain_fx()?.mix()])?;
            let processed = self.master_mixer.add_and_drain(&processed)?;
            *dest = processed.into();
            let processed = self.master_mixer.drain_fx()?.into();
            dest.extend(processed);
            Ok(true)
        } else {
            let raw = self.synthcore.render(self.channel_mixer.buffer_size())?;
            let processed = RawMixerData::from_channels(vec![self.channel_mixer.add_and_drain(&raw)?.mix()])?;
            let processed = self.master_mixer.add_and_drain(&processed)?;
            *dest = processed.into();
            Ok(false)
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
    synth: Arc<FluidSynth>
}
impl SynthCore {
    pub fn new(channels: fluid_int, sample_rate: u32, interp: fluid_int) -> Result<SynthCore, Box<dyn std::error::Error>> {
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
            synth
        })
    }
    pub fn settings(&self) -> &Arc<FluidSettings> {
        &self.settings
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