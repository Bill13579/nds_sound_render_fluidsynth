use fft_sound_convolution::{StereoFFTConvolution, StereoFilter};
use nds_sound_render::*;

use std::{path::{Path}, sync::mpsc};
use hound;
use std::path::PathBuf;
use clap::{Parser, ValueEnum};
use glob::glob;
use tinyaudio::prelude::*;
use std::ffi::CString;

/// Synth interpolation method
#[repr(i32)]
#[derive(ValueEnum, Clone)]
pub enum Interp {
    _NONE = fluid_interp_FLUID_INTERP_NONE,
    _LINEAR = fluid_interp_FLUID_INTERP_LINEAR,
    _4THORDER = fluid_interp_FLUID_INTERP_4THORDER,
    _7THORDER = fluid_interp_FLUID_INTERP_7THORDER
}
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Sets the path to the `.sf2` Soundfont file
    #[arg(value_name = "SF2")]
    sf2: PathBuf,

    /// Sets the path of the MIDI-file to be rendered
    #[arg(value_name = "INPUT")]
    input_glob: String,

    /// Sets the folder to output rendered wave-files in
    #[arg(short = 'o', long, value_name = "OUTPUT")]
    output_folder: Option<PathBuf>,

    /// Internal audio buffer size
    /// 
    /// Default: 128
    #[arg(short = 'B', long, default_value_t = 128)]
    buffer_size: usize,

    /// Number of output channels
    /// 
    /// The default is `16` as the NDS's audio system had 16 PCM channels
    #[arg(short = 'c', long, default_value_t = 16)]
    channels: u16,

    /// Target bit-depth for bit reduction (set to 0 to disable)
    /// 
    /// NDS supports 16-bit audio, but in reality it seems that the internal processing could end up reducing the output bit-depth to 10-bits.
    /// Source: https://www.reddit.com/r/emulation/comments/ru5nld/i_really_love_the_sound_of_the_nintendo_ds/
    #[arg(short = 'b', long, default_value_t = 10)]
    bitdepth: u8,

    /// Target sample rate for resampling
    /// 
    /// The Nintendo DS's audio systems do not do any interpolation on resampling of audio samples, which means sound coming out of the NDS tend to contain a lot more high-frequency content, a sort of a ringing effect that is awesome, and so to recreate it the audio can be resampled the same way here inside the patched `rustysynth` SF2 player.
    /// Sources indicate different sample rates, but here the one suggested by Wenting Zhang, 32728.5 Hz, is used. https://www.zephray.me/post/nds_3ds_sound_quality/
    /// There is also 32768 Hz, suggested by Justme from https://retrocomputing.stackexchange.com/questions/24952/is-sound-generation-on-the-nintendo-ds-always-clipped-to-10-bits
    #[arg(short = 's', long, default_value_t = 32729)]
    sample_rate: u32,

    /// Interpolation method for resampling
    /// 
    /// Default is none to match NDS hardware.
    #[arg(value_enum, short = 'i', long, default_value_t = Interp::_NONE)]
    interp: Interp,

    /// How many times to loop the midi files (Use -1 to repeat forever)
    #[arg(short = 'r', long, default_value_t = 1)]
    #[structopt(allow_hyphen_values = true)]
    repeat: i32,

    /// Play each MIDI file in addition to creating the wave files
    #[arg(short, long, action)]
    play: bool,

    /// FX IR Convolution Reverb
    /// 
    /// If a path to an impulse response is given, IR convolution is performed and a separate reverb channel is created and mixed.
    #[arg(long)]
    ir: Option<PathBuf>,
    /// Out channel gain
    #[arg(long)]
    outgain: Option<f32>,
    /// IR channel gain
    #[arg(long)]
    irgain: Option<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let output_folder;
    if let Some(custom_output_folder) = cli.output_folder {
        if std::fs::metadata(&custom_output_folder)?.is_dir() {
            output_folder = custom_output_folder;
        } else {
            return Err("Output path must be a folder!".into());
        }
    } else {
        output_folder = std::env::current_dir()?;
    }

    fn valid_midi_file<P: AsRef<Path>>(path: P) -> bool {
            if let Ok(file_metadata) = std::fs::metadata(&path) {
                let is_file = file_metadata.is_file();
                let extension = path.as_ref().extension();
                if let Some(extension) = extension {
                    if let Some(extension) = extension.to_str() {
                        is_file && extension == "mid"
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            }
    }
    let input_file_paths: Vec<(PathBuf, PathBuf)> = glob(&cli.input_glob).expect("Failed to read glob pattern").into_iter().filter_map(|entry| {
        match entry {
            Ok(path) => {
                if !valid_midi_file(&path) {
                    println!("Skipping {}!", path.display());
                    None
                } else {
                    if let Some(input_file_name) = path.file_name() {
                        let mut output_path = output_folder.clone();
                        PathBuf::push(&mut output_path, input_file_name);
                        output_path.set_extension("wav");
                        Some((path, output_path))
                    } else {
                        None
                    }
                }
            },
            Err(e) => {
                println!("{:?}", e);
                None
            }
        }
    }).collect();

    // sound_font - Loaded Soundfont
    // input_file_paths - MIDI files to render and where to render them to
    // output_folder - Output path
    // buffer_size
    // channels
    // bitdepth - Target bit-depth for bit reduction
    // sample_rate - Target sample rate for zero-interpolation resampling
    // interp
    // repeat
    // play
    // ir
    // outgain
    // irgain

    for (input_file_path, output_file_path) in input_file_paths {
        print!("Rendering {}... ", input_file_path.display());
        render(&cli.sf2, input_file_path, output_file_path, cli.buffer_size, cli.channels as fluid_int, cli.bitdepth, cli.sample_rate, cli.interp.clone(), cli.repeat, cli.play, cli.ir.clone(), cli.outgain, cli.irgain)?;
        println!("done!");
    }

    println!("\nFriendly Friends!~ Keep up your training!\n\n");

    Ok(())
}

pub fn render<P: AsRef<Path>>(sound_font: &PathBuf, input_file_path: P, output_file_path: P, buffer_size: usize, channels: fluid_int, bitdepth: u8, sample_rate: u32, interp: Interp, repeat: fluid_int, play: bool, irpath: Option<PathBuf>, outgain: Option<f32>, irgain: Option<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let settings = FluidSettings::new();

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

    let synth = FluidSynth::new(&settings).ok_or("Failed to create synth!")?;
    let player = FluidPlayer::new(&synth).ok_or("Failed to create player!")?;

    unsafe {
        fluid_synth_sfload(synth.get(), fluid_str!(sound_font.display().to_string()), true as fluid_int);
        fluid_synth_set_interp_method(synth.get(), -1, interp as fluid_int).fluid_result("Failed to set sample interpolation method!")?;
        fluid_player_add(player.get(), fluid_str!(input_file_path.as_ref().display().to_string())).fluid_result(&format!("Failed to load midi file \"{}\"!", input_file_path.as_ref().display()))?;
        fluid_player_set_loop(player.get(), repeat).fluid_result("Failed to set loop!")?;
        fluid_player_play(player.get()).fluid_result("Failed to start the MIDI player!")?;
    }

    // Setup file writer
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(output_file_path, spec)?;

    // Setup the audio output.
    let params = OutputDeviceParameters {
        channels_count: 2,
        sample_rate: sample_rate as usize,
        channel_sample_count: buffer_size,
    };
    let (tx, rx) = mpsc::sync_channel::<OutputAudio>(0);
    enum OutputAudio {
        CHUNK(Vec<f32>),
        DONE
    }
    // - Initialize IR
    let mut ir_left = vec![1.0_f64];
    let mut ir_right = vec![1.0_f64];
    if let Some(irpath) = &irpath {
        let mut reader = hound::WavReader::open(irpath)?;
        ir_left = vec![1.0_f64; reader.len() as usize / 2];
        ir_right = vec![1.0_f64; reader.len() as usize / 2];
        for (i, sample) in reader.samples::<f32>().into_iter().enumerate() {
            if i % 2 == 0 {
                ir_left[i / 2] = sample? as f64;
            } else {
                ir_right[i / 2] = sample? as f64;
            }
        }
    }
    let mut stereo_fft_convolution = StereoFFTConvolution::new(ir_left, ir_right, 512);
    let mut stereo_fft_convolution_residual_samples = (stereo_fft_convolution.internal_buffer_size() - 512) as isize;
    // Continue to setup the audio output.
    let _device = run_output_device(
        params,
        move |data| {
            if play {
                match rx.recv() {
                    Ok(OutputAudio::CHUNK(chunk)) => {
                        for (i, val) in chunk.into_iter().enumerate() {
                            if let Some(out) = data.get_mut(i) {
                                *out = val;
                            } else {
                                break;
                            }
                        }
                    },
                    Ok(OutputAudio::DONE) | Err(_) => {},
                }
            }
        },
    )
    .unwrap();
    
    let mut out = Mixer::new(synth.count_audio_channels().try_into()?); out.init_len(buffer_size);
    let mut fx = Mixer::new((synth.count_effects_channels() * synth.count_effects_groups()).try_into()?); fx.init_len(buffer_size);

    loop {
        out.zero();
        fx.zero();
        synth.process(&mut out, &mut fx)?;

        let master = out.mix();
        let [left, right] = master.audio_buffers();

        let master_samples: Vec<f32> = left.iter().zip(right.iter()).map(|(&l, &r)| [l, r])
            .map(|[l, r]| {
                if irpath.is_some() {
                    // Do convolution
                    let (convl, convr) = stereo_fft_convolution.compute((l as f64, r as f64));
                    let (convl, convr) = (convl as f32, convr as f32);
                    [l * outgain.unwrap_or(1.0) + convl * irgain.unwrap_or(1.0), r * outgain.unwrap_or(1.0) + convr * irgain.unwrap_or(1.0)]
                } else {
                    [l, r]
                }
            })
            .flatten()
            .map(|x| math::quantize_to_bitdepth(x, bitdepth)) // Quantization
            .collect();
        for sample in &master_samples {
            writer.write_sample(*sample)?;
        }
        if play {
            tx.send(OutputAudio::CHUNK(master_samples))?;
        }

        unsafe {
            if fluid_player_get_status(player.get()) == fluid_player_status_FLUID_PLAYER_DONE {
                if stereo_fft_convolution_residual_samples <= 0 {
                    if play {
                        tx.send(OutputAudio::DONE)?;
                    }
                    break;
                }
                stereo_fft_convolution_residual_samples -= buffer_size as isize;
            }
        }
    }

    Ok(())
}

