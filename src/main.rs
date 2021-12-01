use rustfft::{
    FftPlanner
};

use rand::{
    thread_rng, SeedableRng
};

use rand_chacha::{
    ChaCha8Rng
};

use ndarray::{
    Array2
    , Axis
    , s
    , parallel::prelude::*
};

use ndarray_npy::{
    write_npy
};

use num::{
    complex::{
        Complex
    }
};

use clap::{
    App
    , Arg
};

use pinknoise::{
    VmPinkRGN
};


fn main() {
    let matches=App::new("observe 1/f noise")
        .arg(
            Arg::new("nch")
            .short('c')
            .long("nch")
            .takes_value(true)
            .value_name("number of channels")
            .required(true)
            .about("number of channels of fft")
        )
        .arg(
            Arg::new("ncum")
            .short('m')
            .long("ncum")
            .takes_value(true)
            .value_name("number of cummulation")
            .required(true)
            .about("number of fft per point")
        )
        .arg(
            Arg::new("npt")
            .short('n')
            .long("npt")
            .takes_value(true)
            .value_name("number of points")
            .required(true)
            .about("number of points")
        )
        .arg(
            Arg::new("out")
            .short('o')
            .long("out")
            .takes_value(true)
            .value_name("out name")
            .required(true)
            .about("output file")
        )
        .get_matches();

    let nch=matches.value_of("nch").unwrap().parse::<usize>().unwrap();
    let ncum=matches.value_of("ncum").unwrap().parse::<usize>().unwrap();
    let outname=matches.value_of("out").unwrap();
    let npt=matches.value_of("npt").unwrap().parse::<usize>().unwrap();

    let mut planner=FftPlanner::<f64>::new();
    let fft=planner.plan_fft_forward(nch*2);
    let mut buffer=Array2::<Complex<f64>>::zeros((ncum, nch*2));

    let mut rng=ChaCha8Rng::from_entropy();

    let mut vmpn=VmPinkRGN::<i32, 48>::new(16, &mut rng);

    let mut output=Array2::zeros((npt, nch));
    
    for (i, mut row) in output.axis_iter_mut(Axis(0)).enumerate(){
        {
            println!("{}", i as f64/ npt as f64);
        }
        buffer.iter_mut().for_each(|x|{
            *x=(vmpn.get(&mut rng) as f64).into();
        });
        buffer.axis_iter_mut(Axis(0)).for_each(|mut row|{
            fft.process(row.as_slice_mut().unwrap());
        });
        let outbuf=buffer.slice(s![.., ..nch]).map(|x| x.norm_sqr()).mean_axis(Axis(0)).unwrap();
        row.assign(&outbuf);
    }
    write_npy(outname, &output).unwrap();
}
