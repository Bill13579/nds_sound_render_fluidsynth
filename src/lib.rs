#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::{ptr::NonNull, ffi::{CString, c_void}, sync::Arc};

use indexmap::IndexMap;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

mod mixer;

/// Macro to quickly convert any strings and string-likes into an owned CString and then to a C-style string pointer
///
/// Note
/// ====
/// THE STRING IS DROPPED WHEN THE CURRENT SCOPE ENDS.
/// Memory management of a string created in this way is done by Rust. Only use if you are sure that the pointer will not
///  be used again beyond the current scope.
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
/// Trait defined for `fluid_int` as a quick way to convert `FLUID_OK`/`FLUID_FAILED` into Rust-style results
pub trait FluidStatus {
    fn fluid_result_ret(self, message: &str) -> Result<fluid_int, FluidError>;
    fn fluid_result(self, message: &str) -> Result<(), FluidError>;
    fn fluid_expect(self, message: &str);
}
impl FluidStatus for fluid_int {
    /// Convert return value/`FLUID_FAILED` into a Rust-style `Result`
    fn fluid_result_ret(self, message: &str) -> Result<fluid_int, FluidError> {
        if self == FLUID_FAILED.try_into().expect("`interpret_fluid_status`, unexpected error") {
            Err(FluidError::new(message))
        } else {
            Ok(self)
        }
    }
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
/// 
/// Note
/// ====
/// `Send` and `Sync` are implemented on `FluidSettings`, and so the `synth.threadsafe-api`
///  setting **must** be set to `1 (TRUE)` in order to avoid data races.
/// While the official documentation only states that the synth API's are thread-safe by design,
///  due to the inclusion of thread-safe settings modification API's like `fluid_settings_copystr`,
///  it should be ok to mark this as thread-safe.
/// This and `FluidSynth` are the only two structs explicitly marked as thread-safe.
pub struct FluidSettings {
    settings: *mut fluid_settings_t
}
unsafe impl Send for FluidSettings {  }
unsafe impl Sync for FluidSettings {  }
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
/// 
/// Note
/// ====
/// `Send` and `Sync` are implemented on `FluidSynth`, and so the `synth.threadsafe-api`
///  setting **must** be set to `1 (TRUE)` in order to avoid data races.
pub struct FluidSynth {
    _settings: Arc<FluidSettings>,
    synth: *mut fluid_synth_t
}
unsafe impl Send for FluidSynth {  }
unsafe impl Sync for FluidSynth {  }
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
impl FluidSynth {
    pub fn new(settings: Arc<FluidSettings>) -> Option<FluidSynth> {
        unsafe {
            let synth = NonNull::new(new_fluid_synth(settings.get())).map(|nonnull| nonnull.as_ptr())?;
            Some(
                FluidSynth { _settings: settings, synth }
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

    pub fn process(&self, out: &mut mixer::RawMixerData, fx: &mut mixer::RawMixerData) -> Result<(), Box<dyn std::error::Error>> {
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
impl Drop for FluidSynth {
    fn drop(&mut self) {
        unsafe {
            delete_fluid_synth(self.get());
        }
    }
}

/// Playback on synth can be stopped temporarily
pub trait Stoppable {
    fn stop(&mut self) -> Result<(), FluidError>;
    fn play(&mut self) -> Result<(), FluidError>;
}

/// An *unsafe* wrapper of `fluid_player_t` that
///  simply automatically calls `delete_fluid_player`
///  upon being dropped. Usage of the internal `player`
///  object thus still requires unsafe code from fluid.
pub struct FluidPlayer {
    _synth: Arc<FluidSynth>,
    player: *mut fluid_player_t
}
unsafe impl Send for FluidPlayer {  }
impl FluidPlayer {
    pub fn new(synth: Arc<FluidSynth>) -> Option<FluidPlayer> {
        unsafe {
            let player = NonNull::new(new_fluid_player(synth.get())).map(|nonnull| nonnull.as_ptr())?;
            Some(
                FluidPlayer { _synth: synth, player }
            )
        }
    }
    pub fn get(&self) -> *mut fluid_player_t {
        self.player
    }
}
impl Stoppable for FluidPlayer {
    fn stop(&mut self) -> Result<(), FluidError> {
        unsafe {
            fluid_player_stop(self.get())
        }.fluid_result("Failed to stop the MIDI file player! (shouldn't happen https://www.fluidsynth.org/api/group__midi__player.html#gae630aa680bb891be30bffa3d6d5e1b21)")
    }
    fn play(&mut self) -> Result<(), FluidError> {
        unsafe {
            fluid_player_play(self.get())
        }.fluid_result("Failed to start the MIDI file player!")
    }
}
impl Drop for FluidPlayer {
    /// TODO:   WARNING FROM OFFICIAL FluidPlayer DOCS:  Do not call when the synthesizer associated to this player renders audio, i.e. an audio driver is running or any other synthesizer thread concurrently calls fluid_synth_process() or fluid_synth_nwrite_float() or fluid_synth_write_*() ! 
    fn drop(&mut self) {
        unsafe {
            delete_fluid_player(self.get());
        }
    }
}

/// An *unsafe* wrapper of `fluid_midi_event_t` that
///  simply automatically calls `delete_fluid_midi_event`
///  upon being dropped. Usage of the internal `midi_event`
///  object thus still requires unsafe code from fluid.
/// 
/// Note
/// ====
/// |Event Type     |Contents                       |
/// |:--------------|:------------------------------|
/// |NOTEON		    |type, channel, key, velocity   |
/// |NOTEOFF		|type, channel, key, velocity   |
/// |CC CHANGE	    |type, channel, cc #, value     |
/// |PRGM CHANGE	|type, channel, program         |
pub struct FluidMIDIEvent {
    midi_event: *mut fluid_midi_event_t
}
#[repr(i32)]
enum MIDIEventType {
    VoiceNoteOff = 0x80,
    VoiceNoteOn = 0x90,
    VoiceAftertouch = 0xA0,
    VoiceControlChange = 0xB0,
    VoiceProgramChange = 0xC0,
    VoiceChannelPressure = 0xD0,
    VoicePitchBend = 0xE0,
    SystemExclusive = 0xF0
}
impl FluidMIDIEvent {
    pub fn new() -> Option<FluidMIDIEvent> {
        Some(FluidMIDIEvent {
            midi_event: unsafe {
                NonNull::new(new_fluid_midi_event()).map(|nonnull| nonnull.as_ptr())?
            }
        })
    }
    pub fn create_note_on_event(chan: fluid_int, key: fluid_int, velocity: fluid_int) -> Option<FluidMIDIEvent> {
        unsafe {
            let evt = FluidMIDIEvent::new()?;
            fluid_midi_event_set_type(evt.get(), MIDIEventType::VoiceNoteOn as fluid_int);
            fluid_midi_event_set_channel(evt.get(), chan);
            fluid_midi_event_set_key(evt.get(), key);
            fluid_midi_event_set_velocity(evt.get(), velocity);
            Some(evt)
        }
    }
    pub fn create_note_off_event(chan: fluid_int, key: fluid_int, velocity: fluid_int) -> Option<FluidMIDIEvent> {
        unsafe {
            let evt = FluidMIDIEvent::new()?;
            fluid_midi_event_set_type(evt.get(), MIDIEventType::VoiceNoteOff as fluid_int);
            fluid_midi_event_set_channel(evt.get(), chan);
            fluid_midi_event_set_key(evt.get(), key);
            fluid_midi_event_set_velocity(evt.get(), velocity);
            Some(evt)
        }
    }
    pub fn create_control_change_event(chan: fluid_int, cc: fluid_int, value: fluid_int) -> Option<FluidMIDIEvent> {
        unsafe {
            let evt = FluidMIDIEvent::new()?;
            fluid_midi_event_set_type(evt.get(), MIDIEventType::VoiceControlChange as fluid_int);
            fluid_midi_event_set_channel(evt.get(), chan);
            fluid_midi_event_set_control(evt.get(), cc);
            fluid_midi_event_set_value(evt.get(), value);
            Some(evt)
        }
    }
    pub fn create_program_change_event(chan: fluid_int, program: fluid_int) -> Option<FluidMIDIEvent> {
        unsafe {
            let evt = FluidMIDIEvent::new()?;
            fluid_midi_event_set_type(evt.get(), MIDIEventType::VoiceProgramChange as fluid_int);
            fluid_midi_event_set_channel(evt.get(), chan);
            fluid_midi_event_set_program(evt.get(), program);
            Some(evt)
        }
    }
    pub fn get(&self) -> *mut fluid_midi_event_t {
        self.midi_event
    }
}
impl Drop for FluidMIDIEvent {
    fn drop(&mut self) {
        unsafe {
            delete_fluid_midi_event(self.get());
        }
    }
}

/// An *unsafe* wrapper of `fluid_sequencer_t` that
///  simply automatically calls `delete_fluid_sequencer`
///  upon being dropped. Usage of the internal `sequencer`
///  object thus still requires unsafe code from fluid.
pub struct FluidSequencer {
    clients: IndexMap<fluid_seq_id_t, Arc<dyn Send + Sync>>,
    sequencer: *mut fluid_sequencer_t
}
unsafe impl Send for FluidSequencer {  }
impl FluidSequencer {
    pub fn new() -> Option<FluidSequencer> {
        Some(FluidSequencer {
            clients: IndexMap::new(),
            sequencer: unsafe {
                NonNull::new(new_fluid_sequencer2(false as fluid_int)).map(|nonnull| nonnull.as_ptr())?
            }
        })
    }
    pub fn get(&self) -> *mut fluid_sequencer_t {
        self.sequencer
    }

    pub fn add_midi_event_to_buffer(&self, event: &FluidMIDIEvent) -> Result<(), FluidError> {
        unsafe {
            fluid_sequencer_add_midi_event_to_buffer(self.get() as *mut c_void, event.get())
        }.fluid_result("Failed to add MIDI event to buffer")
    }
    pub fn register_fluidsynth(&mut self, synth: Arc<FluidSynth>) -> fluid_seq_id_t {
        let seq_id;
        unsafe {
            seq_id = fluid_sequencer_register_fluidsynth(self.get(), synth.get());
        }
        self.clients.insert(seq_id, synth);
        seq_id
    }
    fn _unregister_client(&mut self, seq_id: fluid_seq_id_t) {
        unsafe {
            fluid_sequencer_unregister_client(self.get(), seq_id);
        }
    }
    pub fn unregister_client(&mut self, seq_id: fluid_seq_id_t) -> Option<Arc<dyn Send + Sync>> {
        self._unregister_client(seq_id);
        self.clients.shift_remove(&seq_id)
    }
}
impl Drop for FluidSequencer {
    fn drop(&mut self) {
        for &seq_id in self.clients.keys() {
            unsafe {
                fluid_sequencer_unregister_client(self.get(), seq_id);
            }
        }
        unsafe {
            delete_fluid_sequencer(self.get());
        }
    }
}

pub mod math;
pub mod play;

