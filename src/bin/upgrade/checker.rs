use gpu_iris_mpc::setup::galois::degree4::basis::Monomial;
use gpu_iris_mpc::setup::galois::degree4::{GaloisRingElement, ShamirGaloisRingShare};
use gpu_iris_mpc::setup::id::PartyID;
use itertools::{izip, Itertools};

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

    for (j, ((a0, a1, a2, a3), (b0, b1, b2, b3), (c0, c1, c2, c3))) in izip!(
        share0.into_iter().tuples(),
        share1.into_iter().tuples(),
        share2.into_iter().tuples()
    )
    .enumerate()
    {
        let shares = [
            GaloisRingElement::<Monomial>::from_coefs([a0, a1, a2, a3]),
            GaloisRingElement::<Monomial>::from_coefs([b0, b1, b2, b3]),
            GaloisRingElement::<Monomial>::from_coefs([c0, c1, c2, c3]),
        ];
        let lagrange_d1 = [
            ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(PartyID::ID0, PartyID::ID1),
            ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(PartyID::ID1, PartyID::ID0),
        ];
        let s1 = shares[0] * lagrange_d1[0] + shares[1] * lagrange_d1[1];

        let lagrange_d1 = [
            ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(PartyID::ID0, PartyID::ID2),
            ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(PartyID::ID2, PartyID::ID0),
        ];
        let s2 = shares[0] * lagrange_d1[0] + shares[2] * lagrange_d1[1];

        assert_eq!(s1, s2);

        let lagrange_d1 = [
            ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(PartyID::ID1, PartyID::ID2),
            ShamirGaloisRingShare::deg_1_lagrange_polys_at_zero(PartyID::ID2, PartyID::ID1),
        ];
        let s3 = shares[1] * lagrange_d1[0] + shares[2] * lagrange_d1[1];

        assert_eq!(s1, s3);

        let res = s1.to_basis_A();

        println!("{num},{}: {}", j * 4, res.coefs[0]);
        println!("{num},{}: {}", j * 4 + 1, res.coefs[1]);
        println!("{num},{}: {}", j * 4 + 2, res.coefs[2]);
        println!("{num},{}: {}", j * 4 + 3, res.coefs[3]);
    }
}
