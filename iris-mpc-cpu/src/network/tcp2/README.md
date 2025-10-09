> TCP networking stack for iris-mpc-cpu
---

# Overview

The TCP networking stack supports either TCP or TLS. The following traits make that possible:
- `NetworkConnection`: both `TcpStream` and `TlsStream` support `AsyncRead` and `AsyncWrite`. Both can be passed to `tokio::io::split()` to get a read/write half. 
- `Client`: used to initiate a connection.
- `Server`: used to listen for connections.
- `NetworkHandle`: simplifies `HawkActor`. `TcpNetworkHandle` is generic over `NetworkConnection` but `HawkActor` doesn't need to be. 

## `handle.rs`

This module handles the following:
- session-level multiplexing over TCP sockets
- creation of `TcpSession`s
- TCP reconnection

In this module, a `connection_id` refers to a `TcpStream`. For each peer ID, the streams are numbered from zero to `connection_parallelism` (which is a configuration parameter).

For each `(peer, connection_id)` there is a control task (in `session/multiplexer.rs`) to forward traffic between the sessions and sockets 

The program lifecycle is as follows:
1. `TcpNetworkHandle::new()` spawns an `accept_loop()` which listens for connections.
2. `NetworkHandle::make_sessions()` will create the connections and then turn them into sessions.
3. Connections retry until cancelled but once they are established, if any connection becomes disconnected, all connections will be disconnected. The user will have to call `make_sessions()` again.

# Technical Details

## Message Format
This section is related to `network/value.rs` - namely the `NetworkValue` and `DescriptorByte` enums.

- Each message begins with a 4-byte little-endian `SessionId`, followed by a 1-byte `DescriptorByte`. For certain descriptors, the length is known. When the length is variable, the next field is a 4-byte payload length.

`NetworkValue` does not know about the session id - this field is inserted by the outbound forwarding task and is used by the inbound forwarding task to send the message to the correct `TcpSession`.

## Establishing Connections
When `PeerConnectionBuilder::include_peer()` is called
- the connection builder compares its Identity with the peers Identity. The peer with the greater ID initiates the connection. 
- both peers have to call `PeerConnectionBuilder::include_peer()` for the connection process to work, but the peer with the lower id receives the connection passively via the accept loop.
- when the accept loop accepts an inbound TCP connection, a handshake process is used to establish who initiated the connection and what ID to use for the TcpStream.
- it is crucial that `set_nodelay(true)` is called on the `TcpStream`. Without this, the TCP networking stack will be at least 10x slower than the previous gRPC networking stack, even though the gRPC networking stack used 3 additional framing protocols between `NetworkValue` the TCP.
