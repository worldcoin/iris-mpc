//! `ConsensusTransport` over ampc-common's [`ControlChannel`]. Each proposal
//! is bincode-serialized into a `WireFrame { cycle_nonce, msg }`; nonce or
//! variant mismatch is fatal.
//!
//! `exchange` issues both sends before either receive — matches
//! `ControlChannel::sync`'s deadlock-avoidance pattern. With enough channel
//! buffering for one proposal, all parties' sends finish before any blocks
//! on a recv. `liveness_barrier` follows the same sends-before-recvs shape
//! across two rounds, so it needs buffering for two small messages per leg —
//! trivially satisfied by TCP socket buffers (production `ControlChannel`
//! sends block only until flushed, not until the peer consumes).

use std::time::Duration;

use ampc_actor_utils::network::mpc::handle::control_channel::ControlChannel;
use ampc_actor_utils::network::mpc::NetworkValue;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::checkpoint_protocol::{ConsensusMessage, ConsensusTransport, CycleError};

#[derive(Serialize, Deserialize)]
struct WireFrame {
    cycle_nonce: u128,
    msg: ConsensusMessage,
}

/// `ConsensusTransport` adapter over a [`ControlChannel`]. The channel is
/// held behind a `tokio::sync::Mutex` so `exchange` can take `&self` while
/// the underlying primitive needs `&mut self` (TCP streams).
pub struct RingConsensusTransport {
    channel: tokio::sync::Mutex<Box<dyn ControlChannel>>,
}

impl RingConsensusTransport {
    pub fn new(channel: Box<dyn ControlChannel>) -> Self {
        Self {
            channel: tokio::sync::Mutex::new(channel),
        }
    }
}

#[async_trait]
impl ConsensusTransport for RingConsensusTransport {
    async fn exchange<T: Send + 'static>(
        &self,
        msg: ConsensusMessage,
        expect: fn(ConsensusMessage) -> Option<T>,
        cycle_nonce: u128,
        timeout: Duration,
    ) -> Result<Vec<T>, CycleError> {
        let frame = WireFrame {
            cycle_nonce,
            msg: msg.clone(),
        };
        let bytes = bincode::serialize(&frame)
            .map_err(|e| CycleError::Fatal(format!("serialize WireFrame: {e}")))?;

        let mut ch = self.channel.lock().await;
        let result = tokio::time::timeout(timeout, async {
            // Sends first to avoid deadlock — see module doc.
            ch.send_next(NetworkValue::Bytes(bytes.clone()))
                .await
                .map_err(|e| CycleError::Transient(format!("control_channel.send_next: {e}")))?;
            ch.send_prev(NetworkValue::Bytes(bytes))
                .await
                .map_err(|e| CycleError::Transient(format!("control_channel.send_prev: {e}")))?;
            tracing::debug!(
                ?cycle_nonce,
                "ring exchange: proposals sent, awaiting peers"
            );
            let next = recv_bytes("recv_next", ch.recv_next().await).await?;
            let prev = recv_bytes("recv_prev", ch.recv_prev().await).await?;
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

        Ok(vec![parse(next_bytes)?, parse(prev_bytes)?])
    }

    async fn liveness_barrier(&self, timeout: Duration) -> Result<(), CycleError> {
        let my_nonce: u128 = rand::random();
        let mut ch = self.channel.lock().await;
        let result = tokio::time::timeout(timeout, async {
            // Round 1 (challenge): send our fresh nonce to both neighbours.
            // Sends precede receives — same deadlock-avoidance as `exchange`.
            let challenge = NetworkValue::Bytes(my_nonce.to_le_bytes().to_vec());
            ch.send_next(challenge.clone())
                .await
                .map_err(|e| CycleError::Transient(format!("liveness send_next: {e}")))?;
            ch.send_prev(challenge)
                .await
                .map_err(|e| CycleError::Transient(format!("liveness send_prev: {e}")))?;
            let next_nonce = recv_nonce("liveness recv_next", ch.recv_next().await)?;
            let prev_nonce = recv_nonce("liveness recv_prev", ch.recv_prev().await)?;

            // Round 2 (response): echo each neighbour's nonce back to it, and
            // expect our own nonce echoed in return. A peer that is dead — or
            // whose round-1 nonce merely sat buffered — cannot complete this
            // round, because it never received this call's challenge.
            ch.send_next(NetworkValue::Bytes(next_nonce.to_le_bytes().to_vec()))
                .await
                .map_err(|e| CycleError::Transient(format!("liveness echo_next: {e}")))?;
            ch.send_prev(NetworkValue::Bytes(prev_nonce.to_le_bytes().to_vec()))
                .await
                .map_err(|e| CycleError::Transient(format!("liveness echo_prev: {e}")))?;
            let echo_next = recv_nonce("liveness echo recv_next", ch.recv_next().await)?;
            let echo_prev = recv_nonce("liveness echo recv_prev", ch.recv_prev().await)?;
            Ok::<(u128, u128), CycleError>((echo_next, echo_prev))
        })
        .await;

        let (echo_next, echo_prev) = match result {
            Ok(Ok(pair)) => pair,
            Ok(Err(e)) => return Err(e),
            Err(_elapsed) => {
                return Err(CycleError::Transient(format!(
                    "liveness barrier timed out after {timeout:?}"
                )))
            }
        };
        // A wrong echo means a neighbour never saw this call's nonce — crossed
        // wires or a buggy peer, not mere lateness; treat as fatal.
        if echo_next != my_nonce {
            return Err(CycleError::Fatal(
                "liveness barrier: next neighbour echoed wrong nonce".into(),
            ));
        }
        if echo_prev != my_nonce {
            return Err(CycleError::Fatal(
                "liveness barrier: prev neighbour echoed wrong nonce".into(),
            ));
        }
        Ok(())
    }
}

/// Map a `ControlChannel` recv result to a payload byte vector. A non-Bytes
/// `NetworkValue` is fatal — the channel isn't shared with anything else.
async fn recv_bytes(
    label: &str,
    result: eyre::Result<NetworkValue>,
) -> Result<Vec<u8>, CycleError> {
    match result {
        Ok(NetworkValue::Bytes(b)) => Ok(b),
        Ok(other) => Err(CycleError::Fatal(format!(
            "control_channel.{label}: expected NetworkValue::Bytes, got {other:?}"
        ))),
        Err(e) => Err(CycleError::Transient(format!(
            "control_channel.{label}: {e}"
        ))),
    }
}

/// Parse a 16-byte little-endian `u128` nonce from a control-channel receive.
fn recv_nonce(label: &str, result: eyre::Result<NetworkValue>) -> Result<u128, CycleError> {
    match result {
        Ok(NetworkValue::Bytes(b)) => {
            let arr: [u8; 16] = b.as_slice().try_into().map_err(|_| {
                CycleError::Fatal(format!(
                    "{label}: expected 16-byte nonce, got {} bytes",
                    b.len()
                ))
            })?;
            Ok(u128::from_le_bytes(arr))
        }
        Ok(other) => Err(CycleError::Fatal(format!(
            "{label}: expected NetworkValue::Bytes, got {other:?}"
        ))),
        Err(e) => Err(CycleError::Transient(format!("{label}: {e}"))),
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
    use eyre::Result;
    use tokio::sync::mpsc;

    /// One leg of the in-memory ring — owns its tx/rx halves on each
    /// neighbour side. Implements [`ControlChannel`] directly so it plugs
    /// into [`RingConsensusTransport`] the same way a production TCP
    /// channel does.
    pub struct InMemoryRing {
        send_next: mpsc::Sender<NetworkValue>,
        send_prev: mpsc::Sender<NetworkValue>,
        recv_next: mpsc::Receiver<NetworkValue>,
        recv_prev: mpsc::Receiver<NetworkValue>,
    }

    #[async_trait]
    impl ControlChannel for InMemoryRing {
        async fn send_next(&mut self, value: NetworkValue) -> Result<()> {
            self.send_next
                .send(value)
                .await
                .map_err(|e| eyre::eyre!("send_next: {e}"))
        }
        async fn send_prev(&mut self, value: NetworkValue) -> Result<()> {
            self.send_prev
                .send(value)
                .await
                .map_err(|e| eyre::eyre!("send_prev: {e}"))
        }
        async fn recv_next(&mut self) -> Result<NetworkValue> {
            self.recv_next
                .recv()
                .await
                .ok_or_else(|| eyre::eyre!("recv_next: channel closed"))
        }
        async fn recv_prev(&mut self) -> Result<NetworkValue> {
            self.recv_prev
                .recv()
                .await
                .ok_or_else(|| eyre::eyre!("recv_prev: channel closed"))
        }
        async fn sync(&mut self) -> Result<()> {
            Ok(())
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
    use super::test_ring::{triangle, InMemoryRing};
    use super::*;
    use crate::checkpoint_protocol::{ConsensusMessage, ConsensusTransport};

    fn xport(ring: InMemoryRing) -> RingConsensusTransport {
        RingConsensusTransport::new(Box::new(ring))
    }

    /// Two parties send proposals through the in-memory ring; each receives
    /// the other's. (Three parties needed for `triangle`, so we drive all
    /// three but the third just echoes back to keep the ring drained.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ring_exchange_round_trips_a_proposal() {
        let [p0, p1, p2] = triangle(8);
        let t0 = xport(p0);
        let t1 = xport(p1);
        let t2 = xport(p2);

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
        let collect = |r: Vec<i64>| -> std::collections::BTreeSet<i64> { r.into_iter().collect() };
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
        let t0 = xport(p0);
        let t1 = xport(p1);
        let t2 = xport(p2);

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
        let t0 = xport(p0);
        let t1 = xport(p1);
        let t2 = xport(p2);

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
        let t0 = xport(p0);
        let t1 = xport(p1);

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

    /// All three parties live → every barrier completes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn liveness_barrier_passes_when_all_live() {
        let [p0, p1, p2] = triangle(8);
        let t0 = xport(p0);
        let t1 = xport(p1);
        let t2 = xport(p2);
        let timeout = Duration::from_secs(2);

        let h0 = tokio::spawn(async move { t0.liveness_barrier(timeout).await });
        let h1 = tokio::spawn(async move { t1.liveness_barrier(timeout).await });
        let h2 = tokio::spawn(async move { t2.liveness_barrier(timeout).await });

        assert!(h0.await.unwrap().is_ok());
        assert!(h1.await.unwrap().is_ok());
        assert!(h2.await.unwrap().is_ok());
    }

    /// A peer whose round-1 challenge is buffered for its neighbours but which
    /// never performs the round-2 echo must NOT let either neighbour pass — the
    /// whole point of the challenge/response. `p2` is kept alive (so the peers'
    /// sends still succeed and buffer) but stays silent after round 1.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn liveness_barrier_rejects_buffered_but_dead_peer() {
        let [p0, p1, mut p2] = triangle(8);
        let t0 = xport(p0);
        let t1 = xport(p1);
        let timeout = Duration::from_millis(300);

        // p2's round-1 traffic only: lands buffered in each neighbour's channel.
        let nonce = NetworkValue::Bytes(0u128.to_le_bytes().to_vec());
        p2.send_next(nonce.clone()).await.unwrap();
        p2.send_prev(nonce).await.unwrap();

        let h0 = tokio::spawn(async move { t0.liveness_barrier(timeout).await });
        let h1 = tokio::spawn(async move { t1.liveness_barrier(timeout).await });
        let r0 = h0.await.unwrap();
        let r1 = h1.await.unwrap();

        // Neither neighbour may report success: p2 never echoed their nonces.
        assert!(r0.is_err(), "p0 should not pass against a dead p2: {r0:?}");
        assert!(r1.is_err(), "p1 should not pass against a dead p2: {r1:?}");

        // Keep p2's receivers open until the peers have finished, so the failure
        // is the round-2 echo timeout — not a closed-channel send error.
        drop(p2);
    }
}
