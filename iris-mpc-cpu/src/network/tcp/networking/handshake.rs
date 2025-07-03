use crate::{execution::player::Identity, network::tcp::data::StreamId};
use eyre::{eyre, Result};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};

pub async fn outbound(
    stream: &mut TcpStream,
    own_id: &Identity,
    stream_id: &StreamId,
) -> Result<()> {
    // Perform handshake: send our StreamId, expect to read it back
    if let Err(e) = stream.write_u32(stream_id.0).await {
        return Err(eyre!("Failed to write stream_id during handshake: {:?}", e));
    }
    let echoed_id = match stream.read_u32().await {
        Ok(id) => id,
        Err(e) => {
            return Err(eyre!("Failed to read stream_id during handshake: {:?}", e));
        }
    };
    if echoed_id != stream_id.0 {
        return Err(eyre!(
            "Handshake failed: expected stream_id {}, got {}",
            stream_id.0,
            echoed_id
        ));
    }
    let own_id_bytes = own_id.0.as_bytes();
    let own_id_len = own_id_bytes.len() as u32;
    if let Err(e) = stream.write_u32(own_id_len).await {
        return Err(eyre!(
            "Failed to write own_id length during handshake: {:?}",
            e
        ));
    }
    if let Err(e) = stream.write_all(own_id_bytes).await {
        return Err(eyre!(
            "Failed to write own_id bytes during handshake: {:?}",
            e
        ));
    }
    Ok(())
}

pub async fn inbound(stream: &mut TcpStream) -> Result<(Identity, StreamId)> {
    let stream_id = match stream.read_u32().await {
        Ok(id) => id,
        Err(e) => {
            return Err(eyre!("Failed to read stream_id: {:?}", e));
        }
    };
    if let Err(e) = stream.write_u32(stream_id).await {
        return Err(eyre!("Failed to write stream_id back: {:?}", e));
    }
    let peer_id_length = match stream.read_u32().await {
        Ok(id) => id,
        Err(e) => {
            return Err(eyre!("Failed to read peer_id length: {:?}", e));
        }
    };
    let mut peer_id_bytes = vec![0u8; peer_id_length as usize];
    if let Err(e) = stream.read_exact(&mut peer_id_bytes).await {
        return Err(eyre!("Failed to read peer_id bytes: {:?}", e));
    }
    let peer_id = match String::from_utf8(peer_id_bytes) {
        Ok(s) => Identity(s),
        Err(e) => {
            return Err(eyre!("Failed to parse peer_id bytes as UTF-8: {:?}", e));
        }
    };
    Ok((peer_id, stream_id.into()))
}
