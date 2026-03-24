//! Quick local test for the phase tracer. Generates a synthetic trace file
//! that can be opened in Perfetto UI (https://ui.perfetto.dev/).
//!
//! Usage:
//!   cargo run --example test_phase_trace -p iris-mpc-cpu --features phase_trace
//!   # then open /tmp/hawk_phase_trace_party0_batch0.json in Perfetto

use iris_mpc_cpu::execution::hawk_main::phase_tracer;
use iris_mpc_cpu::phase_trace;
use std::time::Duration;

#[tokio::main]
async fn main() {
    phase_tracer::init(0);

    // Simulate 4 sessions, each doing search phases
    let mut handles = Vec::new();
    for sess in 0..4 {
        for eye in 0..2 {
            let handle = tokio::spawn(phase_tracer::SESSION_CTX.scope(
                phase_tracer::SessionContext {
                    i_eye: eye,
                    i_session: sess,
                    orient: 'N',
                },
                async move {
                    // Simulate open_nodes → dot_product/mpc_lift/oblivious_min → prune → insert
                    {
                        phase_trace!("open_nodes", "n_unopened" => 10);

                        {
                            phase_trace!("dot_product", "n_vectors" => 48);
                            tokio::time::sleep(Duration::from_millis(5 + sess as u64 * 2)).await;
                        }

                        {
                            phase_trace!("mpc_lift");
                            tokio::time::sleep(Duration::from_millis(3)).await;
                        }

                        {
                            phase_trace!("oblivious_min");
                            tokio::time::sleep(Duration::from_millis(2)).await;
                        }
                    }

                    {
                        phase_trace!("prune_candidates", "n_candidates" => 32);
                        tokio::time::sleep(Duration::from_millis(3)).await;
                    }

                    {
                        phase_trace!("insert_and_trim", "n_insertions" => 8);
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                },
            ));
            handles.push(handle);
        }
    }

    for h in handles {
        h.await.unwrap();
    }

    phase_tracer::flush(1);

    eprintln!("Done! Open the trace file in https://ui.perfetto.dev/");
    eprintln!("File: /tmp/hawk_phase_trace_party0_batch1.json");
}
