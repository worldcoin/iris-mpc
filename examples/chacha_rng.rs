// use cudarc::driver::CudaDevice;
// use gpu_iris_mpc::{
//     rng::chacha_field::ChaChaCudaFeRng,
//     setup::{id::PartyID, shamir::Shamir},
// };
// use rand::{thread_rng, Rng};

// const P: u16 = 65519;
// const P32: u32 = P as u32;

// fn main() {
//     // we need three RNG seeds
//     // party i has seeds i, i-1
//     // we let party i generate seed i and send it to party i+1 in the beginning
//     let seed0 = thread_rng().gen::<[u32; 8]>();
//     let seed1 = thread_rng().gen::<[u32; 8]>();
//     let seed2 = thread_rng().gen::<[u32; 8]>();

//     // each party has 2 RNGs, initialized with the seeds
//     // we also have a internal buffer size of valid field elements to be produced, needs to be a multiple of 1000
//     const RNG_BUF_SIZE: usize = 1000 * 1000 * 31; // lets do 31 million at a time here

//     // party 0
//     // we construct the RNGs, using the first device, and the seeds
//     let mut chacha1_p0_0 = ChaChaCudaFeRng::init(RNG_BUF_SIZE, CudaDevice::new(0).unwrap(), seed0);
//     let mut chacha2_p0_0 = ChaChaCudaFeRng::init(RNG_BUF_SIZE, CudaDevice::new(0).unwrap(), seed2);

//     // // we can also use the same seeds to construct new RNGs, but on a different device, however, we need to set a new nonce to ensure the RNG streams are different
//     // let mut chacha1_p0_1 = ChaChaCudaFeRng::init(RNG_BUF_SIZE, CudaDevice::new(1).unwrap(), seed1);
//     // // default nonce is 0
//     // chacha1_p0_1.get_mut_chacha().set_nonce(1);
//     // let mut chacha2_p0_1 = ChaChaCudaFeRng::init(RNG_BUF_SIZE, CudaDevice::new(1).unwrap(), seed2);
//     // chacha2_p0_1.get_mut_chacha().set_nonce(1);

//     // party 1 (this would be on another machine, but we simulate it here)
//     // we construct the RNGs, using the first device, and the seeds
//     let mut chacha1_p1_0 = ChaChaCudaFeRng::init(RNG_BUF_SIZE, CudaDevice::new(0).unwrap(), seed1);
//     let mut chacha2_p1_0 = ChaChaCudaFeRng::init(RNG_BUF_SIZE, CudaDevice::new(0).unwrap(), seed0);

//     // party 2
//     // we construct the RNGs, using the first device, and the seeds
//     let mut chacha1_p2_0 = ChaChaCudaFeRng::init(RNG_BUF_SIZE, CudaDevice::new(0).unwrap(), seed2);
//     let mut chacha2_p2_0 = ChaChaCudaFeRng::init(RNG_BUF_SIZE, CudaDevice::new(0).unwrap(), seed1);

//     // lets fill the buffers for all parties
//     chacha1_p0_0.fill_rng();
//     chacha2_p0_0.fill_rng();

//     chacha1_p1_0.fill_rng();
//     chacha2_p1_0.fill_rng();

//     chacha1_p2_0.fill_rng();
//     chacha2_p2_0.fill_rng();

//     // parties now have filled buffers with valid FEs
//     assert!(chacha1_p0_0.data().iter().all(|&x| x < P));
//     assert!(chacha2_p0_0.data().iter().all(|&x| x < P));
//     assert!(chacha1_p1_0.data().iter().all(|&x| x < P));
//     assert!(chacha2_p1_0.data().iter().all(|&x| x < P));
//     assert!(chacha1_p2_0.data().iter().all(|&x| x < P));
//     assert!(chacha2_p2_0.data().iter().all(|&x| x < P));

//     // parties have the same values due to the seeds
//     assert!(chacha1_p0_0.data() == chacha2_p1_0.data());
//     assert!(chacha1_p1_0.data() == chacha2_p2_0.data());
//     assert!(chacha1_p2_0.data() == chacha2_p0_0.data());

//     // we can use the above values to mask other shares before sending them
//     // a random sharing of 17 and 29:
//     let [p0_share1, p1_share1, p2_share1] = Shamir::share_d1(17, &mut thread_rng());
//     let [p0_share2, p1_share2, p2_share2] = Shamir::share_d1(29, &mut thread_rng());

//     // multiply them
//     let p0_share = ((p0_share1 as u32 * p0_share2 as u32) % P32) as u16;
//     let p1_share = ((p1_share1 as u32 * p1_share2 as u32) % P32) as u16;
//     let p2_share = ((p2_share1 as u32 * p2_share2 as u32) % P32) as u16;

//     //  get lagrange coeffs to convert to additive SS
//     let lagrange_coeff0 = Shamir::my_lagrange_coeff_d2(PartyID::ID0);
//     let lagrange_coeff1 = Shamir::my_lagrange_coeff_d2(PartyID::ID1);
//     let lagrange_coeff2 = Shamir::my_lagrange_coeff_d2(PartyID::ID2);

//     let send0 = ((p0_share as u32 * lagrange_coeff0 as u32) % P32) as u16;
//     let send1 = ((p1_share as u32 * lagrange_coeff1 as u32) % P32) as u16;
//     let send2 = ((p2_share as u32 * lagrange_coeff2 as u32) % P32) as u16;

//     // unmasked shares reconstruct to the correct result
//     assert!((send0 as u32 + send1 as u32 + send2 as u32) % P32 == 17 * 29);

//     // mask shares before sending
//     let send0 =
//         (P32 + send0 as u32 + chacha1_p0_0.data()[0] as u32 - chacha2_p0_0.data()[0] as u32) % P32;
//     let send1 =
//         (P32 + send1 as u32 + chacha1_p1_0.data()[0] as u32 - chacha2_p1_0.data()[0] as u32) % P32;
//     let send2 =
//         (P32 + send2 as u32 + chacha1_p2_0.data()[0] as u32 - chacha2_p2_0.data()[0] as u32) % P32;

//     // masked shares reconstruct to the correct result
//     assert!((send0 as u32 + send1 as u32 + send2 as u32) % P32 == 17 * 29);

//     // repeated calls to fill_rng will produce new values
//     chacha1_p0_0.fill_rng();
//     chacha2_p0_0.fill_rng();

//     // if you only need the values on the GPU, call this instead (much faster due to no host copy)
//     chacha1_p0_0.fill_rng_no_host_copy();
//     chacha2_p0_0.fill_rng_no_host_copy();
//     // and get the reference to the CudaSlice to pass to other stuff
//     let _slice1 = chacha1_p0_0.cuda_slice();
//     let _slice2 = chacha2_p0_0.cuda_slice();
// }

fn main(){}