use eyre::Result;
use iris_mpc_common::{
    galois_engine::degree4::{GaloisRingIrisCodeShare, GaloisRingTrimmedMaskCodeShare},
    IRIS_CODE_LENGTH, MASK_CODE_LENGTH, ROTATIONS,
};
use iris_mpc_cpu::{
    execution::hawk_main::iris_worker::{IrisWorkerPool, LocalIrisWorkerPool, QueryId, QuerySpec},
    hawkers::{aby3::aby3_store::DistanceMode, shared_irises::SharedIrises},
    protocol::shared_iris::GaloisRingSharedIris,
};
use std::{collections::HashMap, sync::Arc};

const TARGETS: usize = 257;
const WINDOW_ROTATIONS: usize = 11;

fn deterministic_iris(seed: u16) -> GaloisRingSharedIris {
    let mut code = [0u16; IRIS_CODE_LENGTH];
    let mut mask = [0u16; MASK_CODE_LENGTH];
    for (idx, value) in code.iter_mut().enumerate() {
        *value = (idx as u16)
            .wrapping_mul(17)
            .wrapping_add(seed.wrapping_mul(23));
    }
    for (idx, value) in mask.iter_mut().enumerate() {
        *value = (idx as u16)
            .wrapping_mul(29)
            .wrapping_add(seed.wrapping_mul(11));
    }
    GaloisRingSharedIris {
        code: GaloisRingIrisCodeShare::new(code, 0),
        mask: GaloisRingTrimmedMaskCodeShare::new(mask, 0),
    }
}

#[test]
fn fused_full_rotation_dot_matches_three_hnsw_windows() -> Result<()> {
    // Query preprocessing materializes large fixed arrays; give this regression
    // test the same generous stack used by the E2E harness.
    std::thread::Builder::new()
        .name("full_rotation_dot".to_string())
        .stack_size(32 * 1024 * 1024)
        .spawn(run_test)?
        .join()
        .expect("full-rotation dot test thread panicked")
}

fn run_test() -> Result<()> {
    let runtime = tokio::runtime::Runtime::new()?;
    // `preferred_scan_layout()` stores mixed planes where the UMMLA kernel is
    // available, so the fused scan below runs the mixed kernel while the
    // windowed comparison paths run the u16 kernel — a cross-kernel check.
    let layout = iris_mpc_cpu::protocol::shared_iris::preferred_scan_layout();
    let mut store = SharedIrises::new(
        HashMap::new(),
        iris_mpc_cpu::protocol::shared_iris::ResidentIris::from_arc(
            Arc::new(GaloisRingSharedIris::default_for_party(0)),
            layout,
        ),
    );
    let vector_ids = (0..TARGETS)
        .map(|idx| {
            store.append(iris_mpc_cpu::protocol::shared_iris::ResidentIris::from_arc(
                Arc::new(deterministic_iris(idx as u16 + 1)),
                layout,
            ))
        })
        .collect::<Vec<_>>();
    let pool = LocalIrisWorkerPool::new_local(store.to_arc(), layout, DistanceMode::MinRotation, 0);
    let query_id = QueryId::new();
    runtime.block_on(pool.cache_queries(vec![(query_id, Arc::new(deterministic_iris(0x5a5a)))]))?;

    let fused = runtime.block_on(
        pool.compute_dot_products_full_rotations(QuerySpec::new(query_id), vector_ids.clone()),
    )?;
    let windows = runtime.block_on(pool.compute_dot_products(vec![
        (
            QuerySpec::with_rotation(query_id, 5, false),
            vector_ids.clone(),
        ),
        (
            QuerySpec::with_rotation(query_id, 15, false),
            vector_ids.clone(),
        ),
        (QuerySpec::with_rotation(query_id, 25, false), vector_ids),
    ]))?;

    assert_eq!(fused.len(), TARGETS * ROTATIONS * 2);
    assert_eq!(windows.len(), 3);
    for window in &windows {
        assert_eq!(window.len(), TARGETS * WINDOW_ROTATIONS * 2);
    }

    for target in 0..TARGETS {
        let fused_record = &fused[target * ROTATIONS * 2..(target + 1) * ROTATIONS * 2];
        let mut reconstructed = Vec::with_capacity(ROTATIONS * 2);
        let first = &windows[0][target * WINDOW_ROTATIONS * 2..(target + 1) * WINDOW_ROTATIONS * 2];
        let center =
            &windows[1][target * WINDOW_ROTATIONS * 2..(target + 1) * WINDOW_ROTATIONS * 2];
        let last = &windows[2][target * WINDOW_ROTATIONS * 2..(target + 1) * WINDOW_ROTATIONS * 2];
        reconstructed.extend_from_slice(first);
        reconstructed.extend_from_slice(&center[2..]);
        reconstructed.extend_from_slice(&last[2..]);
        assert_eq!(fused_record, reconstructed, "target index {target}");
    }

    Ok(())
}
