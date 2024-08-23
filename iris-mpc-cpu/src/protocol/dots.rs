use super::iris::WorkerThread;
use crate::{
    error::Error,
    networks::network_trait::NetworkTrait,
    shares::{
        ring_impl::RingElement,
        vecshare::{SliceShare, VecShare},
    },
    utils::Utils,
};
use iris_mpc_common::{
    galois_engine::degree4::GaloisRingIrisCodeShare, iris_db::iris::IrisCodeArray,
};

impl<N: NetworkTrait> WorkerThread<N> {
    pub(crate) fn rep3_dot_many(
        &mut self,
        a: SliceShare<'_, u16>,
        b: &[VecShare<u16>],
    ) -> Result<VecShare<u16>, Error> {
        let len = b.len();
        if a.len() != IrisCodeArray::IRIS_CODE_SIZE {
            return Err(Error::InvalidSize);
        }

        let mut shares_a = Vec::with_capacity(len);

        for b_ in b.iter() {
            let mut rand = self.prf.gen_zero_share();
            if a.len() != b_.len() {
                return Err(Error::InvalidSize);
            }
            for (a__, b__) in a.iter().zip(b_.iter()) {
                rand += a__ * b__;
            }
            shares_a.push(rand);
        }

        // Network: reshare
        let bytes = Utils::blocking_send_slice_and_receive(&mut self.network, &shares_a)?;
        let shares_b = Utils::ring_iter_from_bytes(bytes, len)?;
        let res = VecShare::from_avec_biter(shares_a, shares_b);

        Ok(res)
    }

    pub(crate) fn shamir_dot_many(
        &mut self,
        a: &GaloisRingIrisCodeShare,
        b: &[GaloisRingIrisCodeShare],
    ) -> Result<Vec<RingElement<u16>>, Error> {
        let len = b.len();
        let mut shares = Vec::with_capacity(len);

        for b_ in b.iter() {
            let mut rand = self.prf.gen_zero_share();
            let res = RingElement(a.trick_dot(b_));
            rand += res;
            shares.push(rand);
        }

        Ok(shares)
    }
}
