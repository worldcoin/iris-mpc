> TCP networking stack for iris-mpc-cpu
---

# Overview

## `connection_builder.rs`

This module manages peer-to-peer TCP connections by coordinating connection setup, acceptance, and reconnection.

The logic resides in the `Worker` struct. It contains HashMaps for pending connections, requested connections, and requested reconnections. It also contains a mapping from peer ID to socket address.

In a loop, the worker struct `tokio::select!`s from the following:
- incoming TCP connections (sent from an `accept_loop`)
- incoming commands (sent from either the `PeerConnectionBuilder` or `Reconnector` structs)
- a channel used to cancel the worker.

To start the connection process, `PeerConnectionBuilder::include_peer()` is called for each peer the user wishes to connect to. This sends a `Cmd::Connect` to the worker task. Each peer is added to `requested_connections`. 

`PeerConnectionBuilder::build()` will wait for all requested connections to complete and return a tuple of (`Reconnector`, `PeerConnections`). These are used as input to `TcpNetworkHandle:new()`.

The `Reconnector` can only request that a previously established connection be re-created. After `PeerConnectionBuilder::build()`, the worker task only allows connections from peers that were saved in `requested_connections`.

## `handle.rs`

This module handles the following:
- session-level multiplexing over TCP sockets
- creation of `TcpSession`s
- TCP reconnection

In this module, a `stream_id` refers to a `TcpStream`. For each peer ID, the streams are numbered from zero to `connection_parallelism` (which is a configuration parameter).

For each `(peer, stream_id)` there is a control task to manage the following:
- an inbound and outbound forwarder
- reconnections - has to obtain a new TcpStream and give the read and write half to the respective forwarders 
- session creation - has to create a mapping of `(peer_id, session_id)` to channel and give these channels to their respective forwarders.

The program lifecycle is as follows:
1. `TcpNetworkHandle::new()` is called with a `PeerConnections` map and spawns one task per `(peer, stream_id)`.
2. These tasks wait until they receive a `Cmd::NewSessions` message before doing any forwarding.
3. When `make_sessions()` is called, channel pairs are created. One side of the channels goes to a `TcpSession` and the other side is stored in a HashMap and sent to the respective forwarders via `Cmd::NewSessions`
4. If a connection fails, it is automatically re-established using the `Reconnector`. Session mappings are kept up to date.

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
