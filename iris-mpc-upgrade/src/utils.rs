use crate::{
    db::V1Db,
    packets::{MaskShareMessage, TwoToThreeIrisCodeMessage},
    OldIrisShareSource,
};
use axum::{routing::get, Router};
use eyre::Context;
use futures::{Stream, StreamExt};
use iris_mpc_common::galois_engine::degree4::{
    GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare,
};
use mpc_uniqueness_check::{bits::Bits, distance::EncodedBits};
use rand_chacha::ChaCha20Rng;
use std::{
    array,
    convert::TryFrom,
    io::{Error as IoError, ErrorKind},
};

pub fn install_tracing() {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let fmt_layer = fmt::layer().with_target(true).with_line_number(true);
    let filter_layer = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap();

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .init();
}

pub struct V1Database {
    pub db: V1Db,
}

pub fn get_shares_from_shares(
    party_id: u8,
    serial_id: u64,
    share: &EncodedBits,
    rng: &mut ChaCha20Rng,
) -> [TwoToThreeIrisCodeMessage; 3] {
    let [a, b, c] = {
        let galois_shared_iris_code =
            GaloisRingIrisCodeShare::reencode_extended_iris_code(&share.0, rng);
        [
            galois_shared_iris_code[0].coefs,
            galois_shared_iris_code[1].coefs,
            galois_shared_iris_code[2].coefs,
        ]
    };

    let message_a = TwoToThreeIrisCodeMessage {
        id:       serial_id,
        party_id: 0,
        from:     party_id,
        data:     a,
    };
    let message_b = TwoToThreeIrisCodeMessage {
        id:       serial_id,
        party_id: 1,
        from:     party_id,
        data:     b,
    };
    let message_c = TwoToThreeIrisCodeMessage {
        id:       serial_id,
        party_id: 2,
        from:     party_id,
        data:     c,
    };
    [message_a, message_b, message_c]
}

pub fn get_shares_from_masks(
    party_id: u8,
    serial_id: u64,
    mask: &Bits,
    rng: &mut ChaCha20Rng,
) -> [MaskShareMessage; 3] {
    let extended_masks = array::from_fn(|i| mask[i] as u16);
    let [ma, mb, mc] = {
        let [a, b, c] = GaloisRingIrisCodeShare::reencode_extended_iris_code(&extended_masks, rng);
        [
            GaloisRingTrimmedMaskCodeShare::from(a).coefs,
            GaloisRingTrimmedMaskCodeShare::from(b).coefs,
            GaloisRingTrimmedMaskCodeShare::from(c).coefs,
        ]
    };

    let masks_a = MaskShareMessage {
        id:       serial_id,
        party_id: 0,
        from:     party_id,
        data:     ma,
    };
    let masks_b = MaskShareMessage {
        id:       serial_id,
        party_id: 1,
        from:     party_id,
        data:     mb,
    };
    let masks_c = MaskShareMessage {
        id:       serial_id,
        party_id: 2,
        from:     party_id,
        data:     mc,
    };
    [masks_a, masks_b, masks_c]
}

impl OldIrisShareSource for V1Database {
    async fn load_code_share(&self, share_id: u64) -> eyre::Result<EncodedBits> {
        self.db.fetch_share(share_id).await.map(|(_, x)| x)
    }

    async fn load_mask(&self, share_id: u64) -> eyre::Result<Bits> {
        self.db.fetch_mask(share_id).await.map(|(_, x)| x)
    }

    fn stream_shares(
        &self,
        share_id_range: std::ops::Range<u64>,
    ) -> eyre::Result<impl Stream<Item = eyre::Result<(u64, EncodedBits)>>> {
        Ok(self.db.stream_shares(share_id_range).map(|x| match x {
            Ok((idx, share)) => Ok((u64::try_from(idx).expect("share_id fits into u64"), share)),
            Err(e) => Err(e.into()),
        }))
    }

    fn stream_masks(
        &self,
        share_id_range: std::ops::Range<u64>,
    ) -> eyre::Result<impl Stream<Item = eyre::Result<(u64, Bits)>>> {
        Ok(self.db.stream_masks(share_id_range).map(|x| match x {
            Ok((idx, share)) => Ok((u64::try_from(idx).expect("share_id fits into u64"), share)),
            Err(e) => Err(e.into()),
        }))
    }
}

pub async fn spawn_healthcheck_server(healthcheck_port: usize) -> eyre::Result<()> {
    let app = Router::new().route("/health", get(|| async {})); // Implicit 200 response
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", healthcheck_port))
        .await
        .wrap_err("Healthcheck listener bind error")?;
    axum::serve(listener, app)
        .await
        .wrap_err("healthcheck listener server launch error")?;
    Ok(())
}

pub fn extract_domain(address: &str, remove_protocol: bool) -> Result<String, IoError> {
    // Try to split the address into domain and port parts.
    let mut address = address.trim().to_string();
    if remove_protocol {
        address = address
            .strip_prefix("http://")
            .or_else(|| address.strip_prefix("https://"))
            .unwrap_or(&address)
            .to_string();
    }

    if let Some((domain, _port)) = address.rsplit_once(':') {
        Ok(domain.to_string())
    } else {
        Err(IoError::new(
            ErrorKind::InvalidInput,
            "Invalid address format",
        ))
    }
}
