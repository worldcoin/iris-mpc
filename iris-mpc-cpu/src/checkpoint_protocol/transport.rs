//! `ConsensusTransport` over a symmetric ring channel.
//!
//! Mirrors the shape of ampc-common PR #103's `ControlChannel`: each party
//! has a `next` and a `prev` neighbour, sends block until flushed, receives
//! block until a complete message arrives. Once #103 merges and the
//! ampc-common rev is bumped, the production wiring is:
//!
//! ```text
//! impl RingChannel for Box<dyn ControlChannel> { /* delegate */ }
//! ```
//!
//! and the sidecar / Hawk restart construct the transport via
//! `NetworkHandle::control_channel()`. Until then, an in-memory triangle
//! ring (defined in tests) drives end-to-end protocol testing without TCP.
//!
//! # Wire format
//!
//! Each proposal is bincode-serialized into a `WireFrame { cycle_nonce, msg }`
//! and sent as a single message on the channel. The nonce lets a party
//! detect cross-wires from a stale cycle (fatal). Variant mismatches (a peer
//! responding with `HeightProposal` to our `BaseProposal`) are also fatal.
//!
//! # Send-before-recv
//!
//! `exchange` issues both sends before either receive. This matches
//! `ControlChannel::sync`'s deadlock-avoidance pattern: as long as the wire
//! has enough buffering to absorb a single proposal, all parties can finish
//! their sends concurrently before any of them blocks on a recv.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::checkpoint_protocol::{ConsensusMessage, ConsensusTransport, CycleError, PeerResponses};

/// Symmetric ring primitive. Each party has a `next` and `prev` neighbour;
/// sends block until flushed, recvs block until a complete message arrives.
/// Structural mirror of ampc-common's `ControlChannel`; once #103 merges we
/// supply a delegating impl over `Box<dyn ControlChannel>`.
#[async_trait]
pub trait RingChannel: Send {
    async fn send_next(&mut self, bytes: Vec<u8>) -> Result<(), CycleError>;
    async fn send_prev(&mut self, bytes: Vec<u8>) -> Result<(), CycleError>;
    async fn recv_next(&mut self) -> Result<Vec<u8>, CycleError>;
    async fn recv_prev(&mut self) -> Result<Vec<u8>, CycleError>;
}

#[derive(Serialize, Deserialize)]
struct WireFrame {
    cycle_nonce: u128,
    msg: ConsensusMessage,
}

/// `ConsensusTransport` adapter over any `RingChannel`. The channel is held
/// behind a `tokio::sync::Mutex` so `exchange` can take `&self` while the
/// underlying primitive needs `&mut self` (TCP streams).
pub struct RingConsensusTransport<R: RingChannel> {
    channel: tokio::sync::Mutex<R>,
}

impl<R: RingChannel> RingConsensusTransport<R> {
    pub fn new(channel: R) -> Self {
        Self {
            channel: tokio::sync::Mutex::new(channel),
        }
    }
}

#[async_trait]
impl<R: RingChannel + Send> ConsensusTransport for RingConsensusTransport<R> {
    async fn exchange<T: Send + 'static>(
        &self,
        msg: ConsensusMessage,
        expect: fn(ConsensusMessage) -> Option<T>,
        cycle_nonce: u128,
        timeout: Duration,
    ) -> Result<PeerResponses<T>, CycleError> {
        let frame = WireFrame {
            cycle_nonce,
            msg: msg.clone(),
        };
        let bytes = bincode::serialize(&frame)
            .map_err(|e| CycleError::Fatal(format!("serialize WireFrame: {e}")))?;

        let mut ch = self.channel.lock().await;
        let result = tokio::time::timeout(timeout, async {
            // Sends first to avoid deadlock — see module doc.
            ch.send_next(bytes.clone()).await?;
            ch.send_prev(bytes).await?;
            let next = ch.recv_next().await?;
            let prev = ch.recv_prev().await?;
            Ok::<(Vec<u8>, Vec<u8>), CycleError>((next, prev))
        })
        .await;

        let (next_bytes, prev_bytes) = match result {
            Ok(Ok(pair)) => pair,
            Ok(Err(e)) => return Err(e),
            Err(_elapsed) => {
                return Err(CycleError::Transient(format!(
                    "ring exchange timed out after {timeout:?}"
                )))
            }
        };

        let parse = |bytes: Vec<u8>| -> Result<T, CycleError> {
            let frame: WireFrame = bincode::deserialize(&bytes)
                .map_err(|e| CycleError::Fatal(format!("deserialize WireFrame: {e}")))?;
            if frame.cycle_nonce != cycle_nonce {
                return Err(CycleError::Fatal(format!(
                    "nonce mismatch: expected {cycle_nonce} got {}",
                    frame.cycle_nonce
                )));
            }
            expect(frame.msg).ok_or_else(|| {
                CycleError::Fatal("peer returned wrong ConsensusMessage variant".into())
            })
        };

        Ok(PeerResponses {
            responses: vec![parse(next_bytes)?, parse(prev_bytes)?],
        })
    }
}

#[cfg(test)]
pub(crate) mod test_ring {
    //! In-process triangle ring. Three parties, six unidirectional mpsc
    //! channels: each party's `send_next` feeds the next party's
    //! `recv_prev`, and `send_prev` feeds the prev party's `recv_next`.
    //!
    //! Use [`triangle`] to build all three `InMemoryRing` instances at once.

    use super::*;
    use tokio::sync::mpsc;

    /// One leg of the in-memory ring — owns its tx/rx halves on each
    /// neighbour side.
    pub struct InMemoryRing {
        send_next: mpsc::Sender<Vec<u8>>,
        send_prev: mpsc::Sender<Vec<u8>>,
        recv_next: mpsc::Receiver<Vec<u8>>,
        recv_prev: mpsc::Receiver<Vec<u8>>,
    }

    #[async_trait]
    impl RingChannel for InMemoryRing {
        async fn send_next(&mut self, bytes: Vec<u8>) -> Result<(), CycleError> {
            self.send_next
                .send(bytes)
                .await
                .map_err(|e| CycleError::Transient(format!("send_next: {e}")))
        }
        async fn send_prev(&mut self, bytes: Vec<u8>) -> Result<(), CycleError> {
            self.send_prev
                .send(bytes)
                .await
                .map_err(|e| CycleError::Transient(format!("send_prev: {e}")))
        }
        async fn recv_next(&mut self) -> Result<Vec<u8>, CycleError> {
            self.recv_next
                .recv()
                .await
                .ok_or_else(|| CycleError::Transient("recv_next: channel closed".into()))
        }
        async fn recv_prev(&mut self) -> Result<Vec<u8>, CycleError> {
            self.recv_prev
                .recv()
                .await
                .ok_or_else(|| CycleError::Transient("recv_prev: channel closed".into()))
        }
    }

    /// Build a 3-party ring. Returned in role order [0, 1, 2]:
    /// party 0's next = 1, prev = 2; party 1's next = 2, prev = 0; etc.
    pub fn triangle(capacity: usize) -> [InMemoryRing; 3] {
        // Directed channel: producer → consumer.
        // 6 legs: 0→1, 1→2, 2→0, 1→0, 2→1, 0→2.
        let (t01, r01) = mpsc::channel(capacity);
        let (t12, r12) = mpsc::channel(capacity);
        let (t20, r20) = mpsc::channel(capacity);
        let (t10, r10) = mpsc::channel(capacity);
        let (t21, r21) = mpsc::channel(capacity);
        let (t02, r02) = mpsc::channel(capacity);

        // Party 0: next=1, prev=2
        let p0 = InMemoryRing {
            send_next: t01,
            recv_next: r10,
            send_prev: t02,
            recv_prev: r20,
        };
        // Party 1: next=2, prev=0
        let p1 = InMemoryRing {
            send_next: t12,
            recv_next: r21,
            send_prev: t10,
            recv_prev: r01,
        };
        // Party 2: next=0, prev=1
        let p2 = InMemoryRing {
            send_next: t20,
            recv_next: r02,
            send_prev: t21,
            recv_prev: r12,
        };
        [p0, p1, p2]
    }
}

#[cfg(test)]
mod tests {
    use super::test_ring::triangle;
    use super::*;
    use crate::checkpoint_protocol::{ConsensusMessage, ConsensusTransport};

    /// Two parties send proposals through the in-memory ring; each receives
    /// the other's. (Three parties needed for `triangle`, so we drive all
    /// three but the third just echoes back to keep the ring drained.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ring_exchange_round_trips_a_proposal() {
        let [p0, p1, p2] = triangle(8);
        let t0 = RingConsensusTransport::new(p0);
        let t1 = RingConsensusTransport::new(p1);
        let t2 = RingConsensusTransport::new(p2);

        let nonce = 0x1234_5678_u128;
        let proposal_0 = ConsensusMessage::HeightProposal { height: 100 };
        let proposal_1 = ConsensusMessage::HeightProposal { height: 200 };
        let proposal_2 = ConsensusMessage::HeightProposal { height: 300 };

        let timeout = std::time::Duration::from_secs(2);

        let h0 = tokio::spawn(async move {
            t0.exchange::<i64>(
                proposal_0,
                |m| match m {
                    ConsensusMessage::HeightProposal { height } => Some(height),
                    _ => None,
                },
                nonce,
                timeout,
            )
            .await
        });
        let h1 = tokio::spawn(async move {
            t1.exchange::<i64>(
                proposal_1,
                |m| match m {
                    ConsensusMessage::HeightProposal { height } => Some(height),
                    _ => None,
                },
                nonce,
                timeout,
            )
            .await
        });
        let h2 = tokio::spawn(async move {
            t2.exchange::<i64>(
                proposal_2,
                |m| match m {
                    ConsensusMessage::HeightProposal { height } => Some(height),
                    _ => None,
                },
                nonce,
                timeout,
            )
            .await
        });

        let r0 = h0.await.expect("p0 join").expect("p0 exchange");
        let r1 = h1.await.expect("p1 join").expect("p1 exchange");
        let r2 = h2.await.expect("p2 join").expect("p2 exchange");

        // Each party sees the other two parties' heights (set semantics — order
        // is next/prev which depends on party).
        let collect = |r: PeerResponses<i64>| -> std::collections::BTreeSet<i64> {
            r.responses.into_iter().collect()
        };
        let s0 = collect(r0);
        let s1 = collect(r1);
        let s2 = collect(r2);
        assert_eq!(s0, [200, 300].into_iter().collect());
        assert_eq!(s1, [100, 300].into_iter().collect());
        assert_eq!(s2, [100, 200].into_iter().collect());
    }

    /// Nonce mismatch surfaces as a fatal error.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ring_exchange_nonce_mismatch_is_fatal() {
        let [p0, p1, p2] = triangle(8);
        let t0 = RingConsensusTransport::new(p0);
        let t1 = RingConsensusTransport::new(p1);
        let t2 = RingConsensusTransport::new(p2);

        let timeout = std::time::Duration::from_secs(2);

        // Party 0 uses nonce A; parties 1 and 2 use nonce B.
        let h0 = tokio::spawn(async move {
            t0.exchange::<i64>(
                ConsensusMessage::HeightProposal { height: 1 },
                |m| match m {
                    ConsensusMessage::HeightProposal { height } => Some(height),
                    _ => None,
                },
                0xAA,
                timeout,
            )
            .await
        });
        let h1 = tokio::spawn(async move {
            t1.exchange::<i64>(
                ConsensusMessage::HeightProposal { height: 2 },
                |m| match m {
                    ConsensusMessage::HeightProposal { height } => Some(height),
                    _ => None,
                },
                0xBB,
                timeout,
            )
            .await
        });
        let h2 = tokio::spawn(async move {
            t2.exchange::<i64>(
                ConsensusMessage::HeightProposal { height: 3 },
                |m| match m {
                    ConsensusMessage::HeightProposal { height } => Some(height),
                    _ => None,
                },
                0xBB,
                timeout,
            )
            .await
        });

        let r0 = h0.await.expect("p0 join");
        let _ = h1.await.expect("p1 join"); // p1/p2 may succeed since they agree on nonce
        let _ = h2.await.expect("p2 join");

        match r0 {
            Err(CycleError::Fatal(msg)) => assert!(msg.contains("nonce mismatch"), "got: {msg}"),
            other => panic!("expected Fatal nonce mismatch, got {other:?}"),
        }
    }

    /// Wrong-variant response from peer is fatal.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ring_exchange_variant_mismatch_is_fatal() {
        let [p0, p1, p2] = triangle(8);
        let t0 = RingConsensusTransport::new(p0);
        let t1 = RingConsensusTransport::new(p1);
        let t2 = RingConsensusTransport::new(p2);

        let nonce = 7;
        let timeout = std::time::Duration::from_secs(2);

        // Party 0 expects HeightProposal but party 1 sends HashProposal.
        let h0 = tokio::spawn(async move {
            t0.exchange::<i64>(
                ConsensusMessage::HeightProposal { height: 1 },
                |m| match m {
                    ConsensusMessage::HeightProposal { height } => Some(height),
                    _ => None,
                },
                nonce,
                timeout,
            )
            .await
        });
        let h1 = tokio::spawn(async move {
            t1.exchange::<[u8; 32]>(
                ConsensusMessage::HashProposal { hash: [0xCC; 32] },
                |m| match m {
                    ConsensusMessage::HashProposal { hash } => Some(hash),
                    _ => None,
                },
                nonce,
                timeout,
            )
            .await
        });
        let h2 = tokio::spawn(async move {
            t2.exchange::<i64>(
                ConsensusMessage::HeightProposal { height: 3 },
                |m| match m {
                    ConsensusMessage::HeightProposal { height } => Some(height),
                    _ => None,
                },
                nonce,
                timeout,
            )
            .await
        });

        let r0 = h0.await.expect("p0 join");
        let _ = h1.await.expect("p1 join");
        let _ = h2.await.expect("p2 join");

        match r0 {
            Err(CycleError::Fatal(msg)) => assert!(msg.contains("wrong"), "got: {msg}"),
            other => panic!("expected Fatal variant mismatch, got {other:?}"),
        }
    }

    /// A party that never sends causes the others to time out — transient.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ring_exchange_times_out_when_peer_silent() {
        let [p0, p1, _p2] = triangle(8);
        // p2 is dropped: its sends never happen, its receives are dropped
        // immediately so p0/p1 get send errors after the channel buffer fills.
        // We use a small timeout to keep the test fast.
        let t0 = RingConsensusTransport::new(p0);
        let t1 = RingConsensusTransport::new(p1);

        let nonce = 1;
        let timeout = std::time::Duration::from_millis(150);

        let h0 = tokio::spawn(async move {
            t0.exchange::<i64>(
                ConsensusMessage::HeightProposal { height: 1 },
                |m| match m {
                    ConsensusMessage::HeightProposal { height } => Some(height),
                    _ => None,
                },
                nonce,
                timeout,
            )
            .await
        });
        let h1 = tokio::spawn(async move {
            t1.exchange::<i64>(
                ConsensusMessage::HeightProposal { height: 2 },
                |m| match m {
                    ConsensusMessage::HeightProposal { height } => Some(height),
                    _ => None,
                },
                nonce,
                timeout,
            )
            .await
        });

        let r0 = h0.await.expect("p0 join");
        let r1 = h1.await.expect("p1 join");

        // At least one party must observe a transient (timeout or channel
        // closed) — the missing third party prevents the exchange from
        // completing.
        let any_transient = matches!(r0, Err(CycleError::Transient(_)))
            || matches!(r1, Err(CycleError::Transient(_)));
        assert!(
            any_transient,
            "expected at least one transient; got r0={r0:?}, r1={r1:?}"
        );
    }
}
