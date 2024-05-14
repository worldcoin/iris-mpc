use criterion::{criterion_group, criterion_main, Criterion};
use iris_mpc_upgrade::{
    packets::{ShamirSharesMessage, TwoToThreeIrisCodeMessage},
    share::RepShare,
    upgrade::IrisCodeUpgrader,
    NewIrisShareSink, PartyID, Seed,
};

struct DummySink;

impl NewIrisShareSink for DummySink {
    fn store_code_share(&self, _share_id: u64, _share: Vec<u16>) -> std::io::Result<()> {
        Ok(())
    }

    fn store_mask_share(&self, _share_id: u64, _sharee: Vec<u16>) -> std::io::Result<()> {
        Ok(())
    }
}

fn upgrade(
    msg1: TwoToThreeIrisCodeMessage,
    msg2: TwoToThreeIrisCodeMessage,
    masks: ShamirSharesMessage,
) {
    let seed1 = Seed::from([0u8; 16]);
    let seed2 = Seed::from([1u8; 16]);
    let upgrader = IrisCodeUpgrader::new(seed1, seed2, PartyID::ID0);

    let (local, remote) = upgrader.stage1(msg1, msg2).unwrap();
    let (local, remote) = upgrader.stage2(local, remote).unwrap();
    let shares = upgrader.stage3(local, remote).unwrap();
    IrisCodeUpgrader::finalize(shares, masks, &DummySink).unwrap()
}

fn criterion_benchmark_upgrade(c: &mut Criterion) {
    let msg1 = TwoToThreeIrisCodeMessage {
        id: 0,
        party_id: 0,
        from: 0,
        data: (0..12800).map(|i| RepShare::new(i, i + 1)).collect(),
    };
    let msg2 = TwoToThreeIrisCodeMessage {
        id: 0,
        party_id: 0,
        from: 1,
        data: (0..12800)
            .map(|i| RepShare::new(1000 + i, i + 1001))
            .collect(),
    };
    let masks = ShamirSharesMessage {
        id: 0,
        party_id: 0,
        from: 0,
        data: (0..12800).collect(),
    };

    c.bench_function("full upgrade", |b| {
        b.iter(|| upgrade(msg1.clone(), msg2.clone(), masks.clone()));
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = criterion_benchmark_upgrade
);
criterion_main!(benches);
