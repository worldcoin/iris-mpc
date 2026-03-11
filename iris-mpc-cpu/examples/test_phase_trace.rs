//! Quick local test for the phase tracer. Generates a synthetic trace file
//! that can be opened in Perfetto UI (https://ui.perfetto.dev/).
//!
//! Usage:
//!   HAWK_PHASE_TRACE=1 cargo run --example test_phase_trace -p iris-mpc-cpu --features phase_trace
//!   # then open /tmp/hawk_phase_trace_party0_batch0.json in Perfetto

use iris_mpc_cpu::execution::hawk_main::phase_tracer;
use iris_mpc_cpu::phase_trace;
use std::time::Duration;

#[tokio::main]
async fn main() {
    // Initialize tracer (reads HAWK_PHASE_TRACE env var)
    phase_tracer::init(0);

    if !phase_tracer::is_enabled() {
        eprintln!("Set HAWK_PHASE_TRACE=1 to enable tracing");
        eprintln!("Example: HAWK_PHASE_TRACE=1 cargo run --example test_phase_trace -p iris-mpc-cpu --features phase_trace");
        return;
    }

    // Simulate 4 sessions, each doing 3 phases
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
                    // Simulate per_query
                    {
                        phase_trace!("per_query", "search", "i_request" => 0, "i_rotation" => 0);

                        // dot_product phase
                        {
                            phase_trace!("dot_product", "cpu", "n_vectors" => 48);
                            tokio::time::sleep(Duration::from_millis(5 + sess as u64 * 2)).await;
                        }

                        // mpc_lift phase
                        {
                            phase_trace!("mpc_lift", "mpc");
                            tokio::time::sleep(Duration::from_millis(3)).await;
                        }

                        // oblivious_min phase
                        {
                            phase_trace!("oblivious_min", "mpc");
                            tokio::time::sleep(Duration::from_millis(2)).await;
                        }
                    }

                    // Second query
                    {
                        phase_trace!("per_query", "search", "i_request" => 1, "i_rotation" => 0);

                        {
                            phase_trace!("dot_product", "cpu", "n_vectors" => 32);
                            tokio::time::sleep(Duration::from_millis(4 + sess as u64)).await;
                        }

                        {
                            phase_trace!("mpc_lift", "mpc");
                            tokio::time::sleep(Duration::from_millis(3)).await;
                        }

                        {
                            phase_trace!("oblivious_min", "mpc");
                            tokio::time::sleep(Duration::from_millis(2)).await;
                        }
                    }
                },
            ));
            handles.push(handle);
        }
    }

    for h in handles {
        h.await.unwrap();
    }

    // Flush to file
    phase_tracer::flush();

    eprintln!("Done! Open the trace file in https://ui.perfetto.dev/");
    eprintln!("File: /tmp/hawk_phase_trace_party0_batch0.json");
}
