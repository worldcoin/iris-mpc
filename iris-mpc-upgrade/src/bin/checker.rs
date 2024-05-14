use iris_mpc_upgrade::{
    shamir::{Shamir, P32},
    PartyID,
};
use itertools::izip;

// quick checking script that recombines the shamir shares for a local server setup and prints the iris code share

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let dir0 = &args[1];
    let dir1 = &args[2];
    let dir2 = &args[3];
    let num = args[4].parse::<usize>().unwrap();

    let share0 = std::fs::read_to_string(format!("{}/code_share_{}", dir0, num))
        .unwrap()
        .trim()
        .lines()
        .map(|x| x.parse::<u16>().unwrap())
        .collect::<Vec<_>>();
    let share1 = std::fs::read_to_string(format!("{}/code_share_{}", dir1, num))
        .unwrap()
        .trim()
        .lines()
        .map(|x| x.parse::<u16>().unwrap())
        .collect::<Vec<_>>();
    let share2 = std::fs::read_to_string(format!("{}/code_share_{}", dir2, num))
        .unwrap()
        .trim()
        .lines()
        .map(|x| x.parse::<u16>().unwrap())
        .collect::<Vec<_>>();

    for (j, (a, b, c)) in izip!(share0, share1, share2).enumerate() {
        let shares = [a, b, c];
        let lagrange_d1 = [
            Shamir::my_lagrange_coeff_d1(PartyID::ID0, PartyID::ID1) as u32,
            Shamir::my_lagrange_coeff_d1(PartyID::ID1, PartyID::ID0) as u32,
        ];
        let s1 = (((shares[0] as u32 * lagrange_d1[0]) % P32
            + (shares[1] as u32 * lagrange_d1[1]) % P32)
            % P32) as u16;

        let lagrange_d1 = [
            Shamir::my_lagrange_coeff_d1(PartyID::ID0, PartyID::ID2) as u32,
            Shamir::my_lagrange_coeff_d1(PartyID::ID2, PartyID::ID0) as u32,
        ];
        let s2 = (((shares[0] as u32 * lagrange_d1[0]) % P32
            + (shares[2] as u32 * lagrange_d1[1]) % P32)
            % P32) as u16;
        assert_eq!(s1, s2);

        let lagrange_d1 = [
            Shamir::my_lagrange_coeff_d1(PartyID::ID1, PartyID::ID2) as u32,
            Shamir::my_lagrange_coeff_d1(PartyID::ID2, PartyID::ID1) as u32,
        ];
        let s3 = (((shares[1] as u32 * lagrange_d1[0]) % P32
            + (shares[2] as u32 * lagrange_d1[1]) % P32)
            % P32) as u16;
        assert_eq!(s1, s3);

        println!("{num},{j}: {s1}");
    }
}
