use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
};

use rayon::prelude::*;

use std::time::{Instant};

use rayon::iter::IntoParallelRefMutIterator;
use serde_yaml::{from_reader, to_writer};

#[cfg(target_arch="x86_64")]
use rustfft::FftPlannerSse as FftPlanner;

#[cfg(target_arch="aarch64")]
use rustfft::FftPlanner as FftPlanner;

use rand::{
    Rng,
    SeedableRng, //, thread_rng
};

use rand_chacha::ChaCha8Rng;

use rand_distr::StandardNormal;

use ndarray::{
    s
    , Array2,
    Axis,
};

//use ndarray_npy::write_npy;

use num::complex::Complex;

use clap::{App, Arg};

use pinknoise::VmPinkRng;

fn main() {
    let matches = App::new("observe 1/f noise")
        .arg(
            Arg::new("nch")
                .short('c')
                .long("nch")
                .takes_value(true)
                .value_name("number of channels")
                .required(true)
                .about("number of channels of fft"),
        )
        .arg(
            Arg::new("ncum")
                .short('m')
                .long("ncum")
                .takes_value(true)
                .value_name("number of cummulation")
                .required(true)
                .about("number of fft per point"),
        )
        .arg(
            Arg::new("npt")
                .short('n')
                .long("npt")
                .takes_value(true)
                .value_name("number of points")
                .required(true)
                .about("number of points"),
        )
        .arg(
            Arg::new("state_file_prefix")
                .short('s')
                .long("state")
                .takes_value(true)
                .value_name("state file prefix")
                .required(false)
                .default_value("state")
                .about("pink noise generator state"),
        )
        .arg(
            Arg::new("pink_rng_order")
                .short('p')
                .long("po")
                .takes_value(true)
                .value_name("order")
                .required(false)
                .default_value("48")
                .about("pink noise generator order"),
        )
        .arg(
            Arg::new("gain_std")
                .short('g')
                .long("gs")
                .takes_value(true)
                .value_name("gain std")
                .required(false)
                .default_value("0.01")
                .about("gain std in dB"),
        )
        .arg(
            Arg::new("gain_std2")
                .short('G')
                .long("gs2")
                .takes_value(true)
                .value_name("gain std2")
                .required(false)
                .default_value("0")
                .about("gain std in fraction"),
        )
        .arg(
            Arg::new("out")
                .short('o')
                .long("out")
                .takes_value(true)
                .value_name("out name")
                .required(true)
                .about("output file"),
        )
        .get_matches();

    let nch = matches.value_of("nch").unwrap().parse::<usize>().unwrap();
    let ncum = matches.value_of("ncum").unwrap().parse::<usize>().unwrap();
    let outname = matches.value_of("out").unwrap();
    let npt = matches.value_of("npt").unwrap().parse::<usize>().unwrap();
    let pno = matches
        .value_of("pink_rng_order")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let gs = matches
        .value_of("gain_std")
        .unwrap()
        .parse::<f64>()
        .unwrap();

    let gs2=matches.value_of("gain_std2").unwrap().parse::<f64>().unwrap();
    let state_fname = matches.value_of("state_file_prefix").unwrap();
    let state1_name=state_fname.to_string()+".1.yaml";
    let state2_name=state_fname.to_string()+".2.yaml";

    
    let fft = {
        #[cfg(target_arch="x86_64")]
        let mut planner = FftPlanner::<f64>::new().unwrap();

        #[cfg(target_arch="aarch64")]
        let mut planner = FftPlanner::<f64>::new();

        planner.plan_fft_forward(nch * 2)};

    let mut scratch=vec![Complex::<f64>::default();fft.get_inplace_scratch_len()];
    let mut buffer = Array2::<Complex<f64>>::zeros((ncum, nch * 2));
    

    let mut rng = ChaCha8Rng::from_entropy();
    let mut rng2 = ChaCha8Rng::from_entropy();

    let vmpn1 = if Path::new(state1_name.as_str()).exists() {
        println!("read pink noise state from {}", state1_name.as_str());
        let mut state_file = File::open(state1_name.as_str()).unwrap();
        from_reader(&mut state_file).unwrap()
    } else {
        VmPinkRng::<f64>::new(pno, &mut rng)
    };

    let vmpn2 = if Path::new(state2_name.as_str()).exists() {
        println!("read pink noise state from {}", state2_name.as_str());
        let mut state_file = File::open(state2_name.as_str()).unwrap();
        from_reader(&mut state_file).unwrap()
    } else {
        VmPinkRng::<f64>::new(pno, &mut rng)
    };

    let mut vmpns=[vmpn1, vmpn2];

    let now = Instant::now();
    for i in 0..npt {
        {
            println!("{} {} {:?}",i,  i as f64 / npt as f64, now.elapsed());
        }

        let pink_noise_numbers:Vec<_>=vmpns.par_iter_mut().zip([&mut rng, &mut rng2].par_iter_mut()).map(|(g,r)|{
            (0..ncum*2*nch).map(|_|{g.get(*r)}).collect::<Vec<_>>()
        }).collect();

        buffer.iter_mut().zip(pink_noise_numbers[0].iter().zip(pink_noise_numbers[1].iter())).for_each(|(x, (&r1,&r2))| {
            
            let gain = 10_f64.powf(r1 * gs / 10.0);
            let g2=r2*gs2;
            let signal: f64 = rng.sample(StandardNormal);
            *x = (signal * gain+signal.powi(2)*g2).into();
        });

        fft.process_with_scratch(buffer.as_slice_mut().unwrap(), &mut scratch);
        //fft.process_outofplace_with_scratch(buffer.as_slice_mut().unwrap(), buffer_fft_output.as_slice_mut().unwrap(), &mut scratch);

        /*
        buffer.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            //fft.process(row.as_slice_mut().unwrap());
            fft.process_with
        });*/
        let outbuf = buffer
            .slice(s![.., ..nch])
            .map(|x| x.norm_sqr())
            .mean_axis(Axis(0))
            .unwrap();

        let mut state_file = File::create(state1_name.as_str()).unwrap();
        to_writer(&mut state_file, &vmpns[0]).unwrap();
        let mut state_file = File::create(state2_name.as_str()).unwrap();
        to_writer(&mut state_file, &vmpns[1]).unwrap();

        let mut outfile = OpenOptions::new()
            .create(true)
            .append(true)
            .write(true)
            .open(outname)
            .unwrap();
        outfile
            .write(unsafe {
                std::slice::from_raw_parts(
                    outbuf.as_slice().unwrap().as_ptr() as *const u8,
                    outbuf.len() * std::mem::size_of::<f64>(),
                )
            })
            .unwrap();
    }
}
