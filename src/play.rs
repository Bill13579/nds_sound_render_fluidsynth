// /// A wrapper around the library of a simple stateful MIDI-file player

// use fft_sound_convolution::{StereoFFTConvolution, StereoFilter};
// use tinyaudio::{OutputDeviceParameters, run_output_device};

// use crate::{*, math::quantize_to_bitdepth};
// use std::{path::{Path, PathBuf}, sync::mpsc};

// pub struct Player<'a> {
//     settings: FluidSettings,
//     synth: FluidSynth<'a>,
//     player: FluidPlayer<'a>
// }
// impl<'a> Player<'a> {
//     pub fn new(channels: fluid_int, bitdepth: u8, sample_rate: u32) -> Result<Player<'a>, Box<dyn std::error::Error>> {
//         let settings = FluidSettings::new();

//         unsafe {
//             fluid_settings_setstr(settings.get(), fluid_str!("audio.sample-format"), fluid_str!("16bits")).fluid_result("Failed to modify settings!")?;
//             fluid_settings_setint(settings.get(), fluid_str!("player.reset-synth"), false as fluid_int).fluid_result("Failed to modify settings!")?;
//             fluid_settings_setstr(settings.get(), fluid_str!("player.timing-source"), fluid_str!("sample")).fluid_result("Failed to modify settings!")?;
//             fluid_settings_setint(settings.get(), fluid_str!("synth.audio-channels"), channels).fluid_result("Failed to modify settings!")?;
//             fluid_settings_setint(settings.get(), fluid_str!("synth.audio-groups"), channels).fluid_result("Failed to modify settings!")?;
//             fluid_settings_setnum(settings.get(), fluid_str!("synth.sample-rate"), sample_rate as f64).fluid_result("Failed to modify settings!")?;
//         }
//             //     // Settings of note:
//         //     //  audio.periods - Determines latency
//         //     //  audio.period-size - Determines latency
//         //     //  audio.sample-format - Determines format of samples, float or 16-bit ints
//         //     //  audio.file.* - Rendering to files
//         //     //  player.reset-synth - Affects looping
//         //     //  player.timing-source - Related to timing
//         //     //  synth.audio-channels - Number of stereo channels outputted (multi-channel audio instead of usual stereo)
//         //     //  synth.audio-groups - Number of output channels on the sound card
//         //     //  synth.chorus.active, synth.chorus.* - On or off on chorus, chorus settings
//         //     //  synth.cpu-cores - Multi-threading
//         //     //  synth.midi-channels - Synth number of midi channels
//         //     //  synth.midi-bank-select - How the synth reacts to Bank Select messages
//         //     //  synth.min-note-length - Set a minimum note duration so that shorter notes have a better chance of sounding right
//         //     //  synth.overflow.* - ???
//         //     //  synth.polyphony - Polyphony
//         //     //  synth.reverb.active, synth.reverb.* - On or off on reverb, reverb settings
//         //     //  synth.sample-rate - Synth sample rate, should match audio device
//         //     //  synth.verbose - Verbose

//         let synth = FluidSynth::new(&settings).ok_or("Failed to create synth!")?;
//         let player = FluidPlayer::new(&synth).ok_or("Failed to create player!")?;

//         Ok(Player {
//             settings,
//             synth,
//             player
//         })
//     }
// }

// pub fn render<P: AsRef<Path>>(sound_font: &PathBuf, input_file_path: P, output_file_path: P, buffer_size: usize, channels: fluid_int, bitdepth: u8, sample_rate: u32, interp: fluid_int, repeat: fluid_int, play: bool, irpath: Option<PathBuf>, outgain: Option<f32>, irgain: Option<f32>) -> Result<(), Box<dyn std::error::Error>> {
    


    
//     unsafe {
//         fluid_synth_sfload(synth.get(), fluid_str!(sound_font.display().to_string()), true as fluid_int);
//         fluid_synth_set_interp_method(synth.get(), -1, interp as fluid_int).fluid_result("Failed to set sample interpolation method!")?;
//         fluid_player_add(player.get(), fluid_str!(input_file_path.as_ref().display().to_string())).fluid_result(&format!("Failed to load midi file \"{}\"!", input_file_path.as_ref().display()))?;
//         fluid_player_set_loop(player.get(), repeat).fluid_result("Failed to set loop!")?;
//         fluid_player_play(player.get()).fluid_result("Failed to start the MIDI player!")?;
//     }
    

//     // Setup file writer
//     let spec = hound::WavSpec {
//         channels: 2,
//         sample_rate: sample_rate,
//         bits_per_sample: 32,
//         sample_format: hound::SampleFormat::Float,
//     };
//     let mut writer = hound::WavWriter::create(output_file_path, spec)?;

//     // Setup the audio output.
//     let params = OutputDeviceParameters {
//         channels_count: 2,
//         sample_rate: sample_rate as usize,
//         channel_sample_count: buffer_size,
//     };
//     let (tx, rx) = mpsc::sync_channel::<OutputAudio>(0);
//     enum OutputAudio {
//         CHUNK(Vec<f32>),
//         DONE
//     }
//     // - Initialize IR
//     let mut ir_left = vec![1.0_f64];
//     let mut ir_right = vec![1.0_f64];
//     if let Some(irpath) = &irpath {
//         let mut reader = hound::WavReader::open(irpath)?;
//         ir_left = vec![1.0_f64; reader.len() as usize / 2];
//         ir_right = vec![1.0_f64; reader.len() as usize / 2];
//         for (i, sample) in reader.samples::<f32>().into_iter().enumerate() {
//             if i % 2 == 0 {
//                 ir_left[i / 2] = sample? as f64;
//             } else {
//                 ir_right[i / 2] = sample? as f64;
//             }
//         }
//     }
//     let mut stereo_fft_convolution = StereoFFTConvolution::new(ir_left, ir_right, 512);
//     let mut stereo_fft_convolution_residual_samples = (stereo_fft_convolution.internal_buffer_size() - 512) as isize;
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
    
//     let mut out = Mixer::new(synth.count_audio_channels().try_into()?); out.init_len(buffer_size);
//     let mut fx = Mixer::new((synth.count_effects_channels() * synth.count_effects_groups()).try_into()?); fx.init_len(buffer_size);

//     loop {
//         out.zero();
//         fx.zero();
//         synth.process(&mut out, &mut fx)?;

//         let master = out.mix();
//         let [left, right] = master.audio_buffers();

//         let master_samples: Vec<f32> = left.iter().zip(right.iter()).map(|(&l, &r)| [l, r])
//             .map(|[l, r]| {
//                 if irpath.is_some() {
//                     // Do convolution
//                     let (convl, convr) = stereo_fft_convolution.compute((l as f64, r as f64));
//                     let (convl, convr) = (convl as f32, convr as f32);
//                     [l * outgain.unwrap_or(1.0) + convl * irgain.unwrap_or(1.0), r * outgain.unwrap_or(1.0) + convr * irgain.unwrap_or(1.0)]
//                 } else {
//                     [l, r]
//                 }
//             })
//             .flatten()
//             .map(|x| quantize_to_bitdepth(x, bitdepth)) // Quantization
//             .collect();
//         for sample in &master_samples {
//             writer.write_sample(*sample)?;
//         }
//         if play {
//             tx.send(OutputAudio::CHUNK(master_samples))?;
//         }

//         unsafe {
//             if fluid_player_get_status(player.get()) == fluid_player_status_FLUID_PLAYER_DONE {
//                 if stereo_fft_convolution_residual_samples <= 0 {
//                     if play {
//                         tx.send(OutputAudio::DONE)?;
//                     }
//                     break;
//                 }
//                 stereo_fft_convolution_residual_samples -= buffer_size as isize;
//             }
//         }
//     }

//     Ok(())
// }