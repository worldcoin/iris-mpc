use super::network_trait::{NetworkEstablisher, NetworkTrait};
use bytes::{Bytes, BytesMut};
use eyre::{eyre, Error};
use iris_mpc_common::id::PartyID;
use std::{
    collections::VecDeque,
    io,
    io::{Error as IOError, ErrorKind as IOErrorKind},
};
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

pub struct TestNetwork3p {
    p1_p2_sender:   UnboundedSender<Bytes>,
    p1_p3_sender:   UnboundedSender<Bytes>,
    p2_p3_sender:   UnboundedSender<Bytes>,
    p2_p1_sender:   UnboundedSender<Bytes>,
    p3_p1_sender:   UnboundedSender<Bytes>,
    p3_p2_sender:   UnboundedSender<Bytes>,
    p1_p2_receiver: UnboundedReceiver<Bytes>,
    p1_p3_receiver: UnboundedReceiver<Bytes>,
    p2_p3_receiver: UnboundedReceiver<Bytes>,
    p2_p1_receiver: UnboundedReceiver<Bytes>,
    p3_p1_receiver: UnboundedReceiver<Bytes>,
    p3_p2_receiver: UnboundedReceiver<Bytes>,
}

impl Default for TestNetwork3p {
    fn default() -> Self {
        Self::new()
    }
}

impl TestNetwork3p {
    pub fn new() -> Self {
        // AT Most 1 message is buffered before they are read so this should be fine
        let p1_p2 = mpsc::unbounded_channel();
        let p1_p3 = mpsc::unbounded_channel();
        let p2_p3 = mpsc::unbounded_channel();
        let p2_p1 = mpsc::unbounded_channel();
        let p3_p1 = mpsc::unbounded_channel();
        let p3_p2 = mpsc::unbounded_channel();

        Self {
            p1_p2_sender:   p1_p2.0,
            p1_p3_sender:   p1_p3.0,
            p2_p1_sender:   p2_p1.0,
            p2_p3_sender:   p2_p3.0,
            p3_p1_sender:   p3_p1.0,
            p3_p2_sender:   p3_p2.0,
            p1_p2_receiver: p1_p2.1,
            p1_p3_receiver: p1_p3.1,
            p2_p1_receiver: p2_p1.1,
            p2_p3_receiver: p2_p3.1,
            p3_p1_receiver: p3_p1.1,
            p3_p2_receiver: p3_p2.1,
        }
    }

    pub fn get_party_networks(self) -> [PartyTestNetwork; 3] {
        let party1 = PartyTestNetwork {
            id:        PartyID::ID0,
            send_prev: self.p1_p3_sender,
            recv_prev: self.p3_p1_receiver,
            send_next: self.p1_p2_sender,
            recv_next: self.p2_p1_receiver,
            stats:     [0; 4],
        };

        let party2 = PartyTestNetwork {
            id:        PartyID::ID1,
            send_prev: self.p2_p1_sender,
            recv_prev: self.p1_p2_receiver,
            send_next: self.p2_p3_sender,
            recv_next: self.p3_p2_receiver,
            stats:     [0; 4],
        };

        let party3 = PartyTestNetwork {
            id:        PartyID::ID2,
            send_prev: self.p3_p2_sender,
            recv_prev: self.p2_p3_receiver,
            send_next: self.p3_p1_sender,
            recv_next: self.p1_p3_receiver,
            stats:     [0; 4],
        };

        [party1, party2, party3]
    }
}

pub struct TestNetworkEstablisher {
    id:           PartyID,
    test_network: VecDeque<PartyTestNetwork>,
}

pub struct PartyTestNetwork {
    id:        PartyID,
    send_prev: UnboundedSender<Bytes>,
    send_next: UnboundedSender<Bytes>,
    recv_prev: UnboundedReceiver<Bytes>,
    recv_next: UnboundedReceiver<Bytes>,
    stats:     [usize; 4], // [sent_prev, sent_next, recv_prev, recv_next]
}

impl From<VecDeque<PartyTestNetwork>> for TestNetworkEstablisher {
    fn from(net: VecDeque<PartyTestNetwork>) -> Self {
        Self {
            id:           net.front().unwrap().id,
            test_network: net,
        }
    }
}

impl PartyTestNetwork {
    pub const NUM_PARTIES: usize = 3;
}

impl NetworkEstablisher<PartyTestNetwork> for TestNetworkEstablisher {
    fn get_id(&self) -> PartyID {
        self.id
    }

    fn get_num_parties(&self) -> usize {
        3
    }

    async fn open_channel(&mut self) -> Result<PartyTestNetwork, Error> {
        self.test_network
            .pop_front()
            .ok_or(eyre!("test config error".to_owned()))
    }

    async fn shutdown(self) -> Result<(), Error> {
        Ok(())
    }

    fn get_send_receive(&self, _: usize) -> std::io::Result<(u64, u64)> {
        unreachable!()
    }

    fn print_connection_stats(&self, _: &mut impl std::io::Write) -> std::io::Result<()> {
        unreachable!()
    }
}

impl NetworkTrait for PartyTestNetwork {
    async fn shutdown(self) -> Result<(), IOError> {
        Ok(())
    }

    async fn send(&mut self, id: PartyID, data: Bytes) -> std::io::Result<()> {
        tracing::trace!("send_id {}->{}: {:?}", self.id, id, data);
        let res = if id == self.id.next_id() {
            self.stats[1] += data.len();
            self.send_next
                .send(data)
                .map_err(|_| IOError::new(IOErrorKind::Other, "Send failed"))
        } else if id == self.id.prev_id() {
            self.stats[0] += data.len();
            self.send_prev
                .send(data)
                .map_err(|_| IOError::new(IOErrorKind::Other, "Send failed"))
        } else {
            Err(IOError::new(io::ErrorKind::Other, "Invalid ID"))
        };

        tracing::trace!("send_id {}->{}: done", self.id, id);
        res
    }

    async fn receive(&mut self, id: PartyID) -> std::io::Result<BytesMut> {
        tracing::trace!("recv_id {}<-{}: ", self.id, id);
        let buf = if id == self.id.prev_id() {
            let data = self
                .recv_prev
                .recv()
                .await
                .ok_or_else(|| IOError::new(IOErrorKind::Other, "Receive failed"))?;
            self.stats[2] += data.len();
            data
        } else if id == self.id.next_id() {
            let data = self
                .recv_next
                .recv()
                .await
                .ok_or_else(|| IOError::new(IOErrorKind::Other, "Receive failed"))?;
            self.stats[3] += data.len();
            data
        } else {
            return Err(io::Error::new(io::ErrorKind::Other, "Invalid ID"));
        };
        tracing::trace!("recv_id {}<-{}: done", self.id, id);

        Ok(BytesMut::from(buf.as_ref()))
    }

    async fn broadcast(&mut self, data: Bytes) -> Result<Vec<BytesMut>, io::Error> {
        let mut result = Vec::with_capacity(3);
        for id in 0..3 {
            if id != usize::from(self.id) {
                self.send(PartyID::try_from(id).unwrap(), data.clone())
                    .await?;
            }
        }
        for id in 0..3 {
            if id == usize::from(self.id) {
                result.push(BytesMut::from(data.as_ref()));
            } else {
                result.push(self.receive(PartyID::try_from(id).unwrap()).await?);
            }
        }
        Ok(result)
    }

    fn get_id(&self) -> PartyID {
        self.id
    }

    async fn send_next_id(&mut self, data: Bytes) -> Result<(), IOError> {
        tracing::trace!("send {}->{}: {:?}", self.id, self.id.next_id(), data);
        self.stats[1] += data.len();
        let res = self
            .send_next
            .send(data)
            .map_err(|_| IOError::new(IOErrorKind::Other, "Send failed"));
        tracing::trace!("send {}->{}: done", self.id, self.id.next_id());
        res
    }

    async fn send_prev_id(&mut self, data: Bytes) -> Result<(), IOError> {
        tracing::trace!("send {}->{}: {:?}", self.id, self.id.prev_id(), data);
        self.stats[0] += data.len();
        let res = self
            .send_prev
            .send(data)
            .map_err(|_| IOError::new(IOErrorKind::Other, "Send failed"));
        tracing::trace!("send {}->{}: done", self.id, self.id.prev_id());
        res
    }

    async fn receive_prev_id(&mut self) -> Result<bytes::BytesMut, IOError> {
        tracing::trace!("recv {}<-{}: ", self.id, self.id.prev_id());
        let buf = self
            .recv_prev
            .recv()
            .await
            .ok_or_else(|| IOError::new(IOErrorKind::Other, "Receive failed"))?;
        self.stats[2] += buf.len();

        tracing::trace!("recv {}<-{}: done", self.id, self.id.prev_id());
        Ok(BytesMut::from(buf.as_ref()))
    }

    async fn receive_next_id(&mut self) -> Result<bytes::BytesMut, IOError> {
        tracing::trace!("recv {}<-{}: ", self.id, self.id.next_id());
        let buf = self
            .recv_next
            .recv()
            .await
            .ok_or_else(|| IOError::new(IOErrorKind::Other, "Receive failed"))?;
        self.stats[3] += buf.len();

        tracing::trace!("recv {}<-{}: done", self.id, self.id.next_id());
        Ok(BytesMut::from(buf.as_ref()))
    }

    fn blocking_send(
        &mut self,
        id: PartyID,
        data: Bytes,
    ) -> Result<(), super::network_trait::IoError> {
        tracing::trace!("send_id {}->{}: {:?}", self.id, id, data);
        let res = if id == self.id.next_id() {
            self.blocking_send_next_id(data)
        } else {
            self.blocking_send_prev_id(data)
        };
        tracing::trace!("send_id {}->{}: done", self.id, id);
        res
    }

    fn blocking_send_next_id(&mut self, data: Bytes) -> Result<(), super::network_trait::IoError> {
        tracing::trace!("send {}->{}: {:?}", self.id, self.id.next_id(), data);
        self.stats[1] += data.len();
        let res = self
            .send_next
            .send(data)
            .map_err(|_| IOError::new(IOErrorKind::Other, "Send failed"));
        tracing::trace!("send {}->{}: done", self.id, self.id.next_id());
        res
    }

    fn blocking_send_prev_id(&mut self, data: Bytes) -> Result<(), super::network_trait::IoError> {
        tracing::trace!("send {}->{}: {:?}", self.id, self.id.prev_id(), data);
        self.stats[0] += data.len();
        let res = self
            .send_prev
            .send(data)
            .map_err(|_| IOError::new(IOErrorKind::Other, "Send failed"));
        tracing::trace!("send {}->{}: done", self.id, self.id.prev_id());
        res
    }

    fn blocking_receive(&mut self, id: PartyID) -> Result<BytesMut, super::network_trait::IoError> {
        tracing::trace!("recv_id {}<-{}: ", self.id, id);
        let buf = if id == self.id.next_id() {
            self.blocking_receive_next_id()
        } else {
            self.blocking_receive_prev_id()
        };
        tracing::trace!("recv_id {}<-{}: done", self.id, id);
        buf
    }

    fn blocking_receive_prev_id(&mut self) -> Result<BytesMut, super::network_trait::IoError> {
        tracing::trace!("recv {}<-{}: ", self.id, self.id.prev_id());
        let buf = self
            .recv_prev
            .blocking_recv()
            .ok_or_else(|| IOError::new(IOErrorKind::Other, "Receive failed"))?;
        self.stats[2] += buf.len();

        tracing::trace!("recv {}<-{}: done", self.id, self.id.prev_id());
        Ok(BytesMut::from(buf.as_ref()))
    }

    fn blocking_receive_next_id(&mut self) -> Result<BytesMut, super::network_trait::IoError> {
        tracing::trace!("recv {}<-{}: ", self.id, self.id.next_id());
        let buf = self
            .recv_next
            .blocking_recv()
            .ok_or_else(|| IOError::new(IOErrorKind::Other, "Receive failed"))?;
        self.stats[3] += buf.len();

        tracing::trace!("recv {}<-{}: done", self.id, self.id.next_id());
        Ok(BytesMut::from(buf.as_ref()))
    }

    fn blocking_broadcast(
        &mut self,
        data: Bytes,
    ) -> Result<Vec<BytesMut>, super::network_trait::IoError> {
        let mut result = Vec::with_capacity(3);
        for id in 0..3 {
            if id != usize::from(self.id) {
                self.blocking_send(PartyID::try_from(id).unwrap(), data.clone())?
            }
        }
        for id in 0..3 {
            if id == usize::from(self.id) {
                result.push(BytesMut::from(data.as_ref()));
            } else {
                result.push(self.blocking_receive(PartyID::try_from(id).unwrap())?);
            }
        }
        Ok(result)
    }
}
