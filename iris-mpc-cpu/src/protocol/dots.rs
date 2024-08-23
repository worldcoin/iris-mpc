use super::iris_worker::IrisWorker;
use crate::{
    error::Error,
    networks::network_trait::NetworkTrait,
    shares::{
        ring_impl::RingElement,
        share::Share,
        vecshare::{SliceShare, VecShare},
    },
    utils::Utils,
};
use iris_mpc_common::{
    galois_engine::degree4::GaloisRingIrisCodeShare, iris_db::iris::IrisCodeArray,
};

impl<N: NetworkTrait> IrisWorker<N> {
    pub async fn async_rep3_dot(
        &mut self,
        a: SliceShare<'_, u16>,
        b: SliceShare<'_, u16>,
    ) -> Result<Share<u16>, Error> {
        if a.len() != IrisCodeArray::IRIS_CODE_SIZE || b.len() != IrisCodeArray::IRIS_CODE_SIZE {
            return Err(Error::InvalidSize);
        }

        let mut rand = self.prf.gen_zero_share();
        for (a_, b_) in a.iter().zip(b.iter()) {
            rand += a_ * b_;
        }

        // Network: reshare
        let res_a = rand;
        let bytes_to_send = Utils::ring_to_bytes(&res_a);
        self.network.send_next_id(bytes_to_send).await?;
        let response = self.network.receive_prev_id().await?;
        let res_b = Utils::ring_from_bytes(response)?;

        let res = Share::new(res_a, res_b);
        Ok(res)
    }

    pub fn rep3_dot(
        &mut self,
        a: SliceShare<'_, u16>,
        b: SliceShare<'_, u16>,
    ) -> Result<Share<u16>, Error> {
        if a.len() != IrisCodeArray::IRIS_CODE_SIZE || b.len() != IrisCodeArray::IRIS_CODE_SIZE {
            return Err(Error::InvalidSize);
        }

        let mut rand = self.prf.gen_zero_share();
        for (a_, b_) in a.iter().zip(b.iter()) {
            rand += a_ * b_;
        }

        // Network: reshare
        let res_a = rand;
        let bytes_to_send = Utils::ring_to_bytes(&res_a);
        let response = Utils::blocking_send_and_receive(&mut self.network, bytes_to_send)?;
        let res_b = Utils::ring_from_bytes(response)?;

        let res = Share::new(res_a, res_b);
        Ok(res)
    }

    pub fn rep3_dot_many(
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

    pub fn shamir_dot(
        &mut self,
        a: &GaloisRingIrisCodeShare,
        b: &GaloisRingIrisCodeShare,
    ) -> Result<RingElement<u16>, Error> {
        let mut rand = self.prf.gen_zero_share();
        let res = RingElement(a.trick_dot(b));
        rand += res;
        Ok(rand)
    }

    pub fn shamir_dot_many(
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
