//! A simple implementation of the tripartite Diffie-Hellman key exchange using BLS12-381.
//! This is used to derive a shared secret between three parties in one communication round.
//!
//! The implementation is based on the paper [A One Round Protocol for Tripartite Diffie–Hellman](https://cgi.di.uoa.gr/~aggelos/crypto/page4/assets/joux-tripartite.pdf) by Antoine Joux.
//!
//! # Overview
//! Each party generates a private key $k_i$ and computes the corresponding public keys $pk_i = g^{k_i}$ in both G1 and G2.
//! The parties then exchange their public keys (This is the only communication round).
//!
//! Each party can then compute the shared secret as follows (abusing notation to automatically select the G1 or G2 element from the public key):
//! - Party A computes $H(e(pk_B, pk_C)^{k_A})$
//! - Party B computes $H(e(pk_C, pk_A)^{k_B})$
//! - Party C computes $H(e(pk_A, pk_B)^{k_C})$
//!
//! We use SHA-256 as the hash function H to derive a 32 byte shared secret.
//! Due to the bi-linear properties of the pairing, all parties arrive at the same shared secret.

use ark_ec::{pairing::Pairing, CurveGroup, PrimeGroup};
use ark_ff::UniformRand;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use sha2::Digest;

/// A private key for the tripartite Diffie-Hellman key exchange.
/// Internally an element of the scalar field of BLS12-381.
pub struct PrivateKey {
    k: ark_bls12_381::Fr,
}

impl PrivateKey {
    /// Serialize the private key to a byte vector.
    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes = vec![];
        self.k
            .serialize_compressed(&mut bytes)
            .expect("we can serialize field elements");
        bytes
    }

    /// Tries to deserialize the private key from a byte slice.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, ark_serialize::SerializationError> {
        let k = ark_bls12_381::Fr::deserialize_compressed(bytes)?;
        Ok(Self { k })
    }

    /// Generate a new random private key.
    pub fn random<R: rand::RngCore + rand::CryptoRng>(rng: &mut R) -> Self {
        Self {
            k: ark_bls12_381::Fr::rand(rng),
        }
    }

    /// Compute the corresponding public key.
    pub fn public_key(&self) -> PublicKeys {
        let g1 = ark_bls12_381::G1Projective::generator();
        let g2 = ark_bls12_381::G2Projective::generator();
        PublicKeys {
            pk_g1: (g1 * self.k).into_affine(),
            pk_g2: (g2 * self.k).into_affine(),
        }
    }

    /// Derive the shared secret given the public keys of the other two parties.
    ///
    /// The order of the two provided public keys does not matter, as long as each party inputs the public keys of the other two parties.
    /// We suggest the following ordering:
    /// - Party A: (pk_B, pk_C)
    /// - Party B: (pk_C, pk_A)
    /// - Party C: (pk_A, pk_B)
    pub fn derive_shared_secret(&self, pk_1: &PublicKeys, pk_2: &PublicKeys) -> [u8; 32] {
        let g_t = ark_bls12_381::Bls12_381::pairing(pk_1.pk_g1, pk_2.pk_g2);
        let shared_field = g_t * self.k;
        let mut writer = vec![];
        shared_field
            .0
            .serialize_compressed(&mut writer)
            .expect("we are able to serialize the target group element");
        sha2::Sha256::digest(&writer).into()
    }
}

/// A public key for the tripartite Diffie-Hellman key exchange.
///
/// Contains a G1 and a G2 element of BLS12-381.
pub struct PublicKeys {
    pk_g1: ark_bls12_381::G1Affine,
    pk_g2: ark_bls12_381::G2Affine,
}

impl PublicKeys {
    /// Serialize the public key to a byte vector.
    pub fn serialize(&self) -> Vec<u8> {
        let mut bytes = vec![];
        (self.pk_g1, self.pk_g2)
            .serialize_compressed(&mut bytes)
            .expect("we can serialize group elements");
        bytes
    }

    /// Tries to deserialize the public key from a byte slice.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, ark_serialize::SerializationError> {
        let (pk_g1, pk_g2): (ark_bls12_381::G1Affine, ark_bls12_381::G2Affine) =
            CanonicalDeserialize::deserialize_compressed(bytes)?;
        Ok(Self { pk_g1, pk_g2 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tripartite_dh() {
        let mut rng = rand::thread_rng();
        let sk_a = PrivateKey::random(&mut rng);
        let sk_b = PrivateKey::random(&mut rng);
        let sk_c = PrivateKey::random(&mut rng);

        let pk_a = sk_a.public_key();
        let pk_b = sk_b.public_key();
        let pk_c = sk_c.public_key();

        let ss_a = sk_a.derive_shared_secret(&pk_b, &pk_c);
        let ss_b = sk_b.derive_shared_secret(&pk_c, &pk_a);
        let ss_c = sk_c.derive_shared_secret(&pk_a, &pk_b);

        assert_eq!(ss_a, ss_b);
        assert_eq!(ss_b, ss_c);

        let mut rng = rand::thread_rng();
        let sk_a = PrivateKey::random(&mut rng);
        let sk_b = PrivateKey::random(&mut rng);
        let sk_c = PrivateKey::random(&mut rng);

        let pk_a = sk_a.public_key();
        let pk_b = sk_b.public_key();
        let pk_c = sk_c.public_key();

        let ss2_a = sk_a.derive_shared_secret(&pk_b, &pk_c);
        let ss2_b = sk_b.derive_shared_secret(&pk_c, &pk_a);
        let ss2_c = sk_c.derive_shared_secret(&pk_a, &pk_b);

        assert_eq!(ss2_a, ss2_b);
        assert_eq!(ss2_b, ss2_c);

        // pairing is non-degenerate, so new keys should produce a different shared secret
        assert_ne!(ss_a, ss2_a);
    }

    #[test]
    fn test_serde() {
        let mut rng = rand::thread_rng();
        let sk = PrivateKey::random(&mut rng);
        let pk = sk.public_key();

        let sk_bytes = sk.serialize();
        let pk_bytes = pk.serialize();

        let sk2 = PrivateKey::deserialize(&sk_bytes).expect("we can deserialize the private key");
        let pk2 = PublicKeys::deserialize(&pk_bytes).expect("we can deserialize the public key");

        assert_eq!(sk.k, sk2.k);
        assert_eq!(pk.pk_g1, pk2.pk_g1);
        assert_eq!(pk.pk_g2, pk2.pk_g2);

        let ss1 = sk.derive_shared_secret(&pk, &pk);
        let ss2 = sk2.derive_shared_secret(&pk2, &pk2);
        assert_eq!(ss1, ss2);
    }
}
