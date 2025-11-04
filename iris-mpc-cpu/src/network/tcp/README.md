> TCP networking stack for iris-mpc-cpu
---

# Overview

The TCP networking stack provides the communication channels needed for the 3-party MPC protocol. 
Each party has an Identity and public address (url:port). Each party must connect to the other and over each connection,
establish some number of communication channels, called sessions. 

The parties work on batches jobs over these sessions. The networking stack needs to do the following:

1. Establish connections between the parties
2. Set up the communication channels (sessions) ove each connection
3. Indicate if connectivity was lost. This is expected to cause a job to fail. 
4. Allow for re-creation of connections and sessions.
5. Shutdown gracefully. 

The entry point into this module is `build_network_handle()`. It returns a `Box<dyn NetworkHandle>`, which wraps a `TcpNetworkHandle`. This struct is generic over the types of connections used for inbound and outbound traffic: could be either `TCP` or `TLS`. Normally inbound/outbound traffic uses the same protocol but for testing purposes, it is possible to use different protocols for inbound and outbound traffic.


## Important Traits
- `NetworkConnection`: implements `AsyncRead` and `AsyncWrite`. These allow it to be passed to `tokio::io::split()` to get a read/write half. 
- `Client`: used to initiate a connection.
- `Server`: used to listen for connections.
- `NetworkHandle`: provides `make_sessions()`
- `Networking`: defined in `iris-mpc-cpu/src/network`. Is implemented by `TcpSession`.

## `handle` module
Contains the `TcpNetworkHandle` struct, which does the following:
- implements `NetworkHandle`
- creates connections
- spawns a task that accepts inbound connections

## `connection` module
- contains code to establish inbound/outbound connections.
- `connection/mod.rs` contains the `Connector` struct, which will attempt to connect to a peer by calling `connect()` until it either succeeds or is cancelled. 
- The peer with the greater ID initiates the connection, using `Client::connect()`. The other party uses the `listener::Server` module to send a request to the `accept_loop()`. The `accept_loop()` accepts all incoming connections but unless it has previously been told to accept a connection from a certain peer, incoming connections are then dropped. 
- When the `Server` accepts a peer's connection, the peers will perform a handshake, where the initiator sends its peer ID. If this ID matches one that the `Server` was commanded to accept, the `Server` initiates a second handshake. 

## `session` module
- contains the `TcpSession` struct, which despite the name also is used for the `TLS` protocol.
- creates channels used to forward data between the sockets and the `TcpSession`s. 
- spawns tasks to multiplex sessions over a socket. see `multiplexer.rs`.

## Message Format
This section is related to `network/value.rs` - namely the `NetworkValue` and `DescriptorByte` enums.

- Each message begins with a 4-byte little-endian `SessionId`, followed by a 1-byte `DescriptorByte`. For certain descriptors, the length is known. When the length is variable, the next field is a 4-byte payload length.

`NetworkValue` does not know about the session id - this field is inserted by the outbound forwarding task and is used by the inbound forwarding task to send the message to the correct `TcpSession`.

## Establishing Connections
- when the accept loop accepts an inbound TCP connection, a handshake process is used to establish who initiated the connection and what ID to use for the TcpStream.
- it is crucial that `set_nodelay(true)` is called on the `TcpStream`. Without this, the TCP networking stack will be at least 10x slower than the previous gRPC networking stack, even though the gRPC networking stack used 3 additional framing protocols between `NetworkValue` the TCP.
