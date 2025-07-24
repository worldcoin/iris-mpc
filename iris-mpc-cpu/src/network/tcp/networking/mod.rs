use eyre::{bail, Result};
use std::io;
use std::os::unix::io::AsRawFd;
use tokio::net::TcpStream;

pub mod client;
pub mod connection_builder;
mod handshake;
pub mod server;

/// set no_delay and keepalive
fn configure_tcp_stream(stream: &TcpStream) -> Result<()> {
    // idle time before keepalives get sent. NGINX default is 60 seconds. want to be less than that.
    const KEEPALIVE_TIME_SECS: libc::c_int = 30;
    // how often to send keepalives
    const KEEPALIVE_INTERVAL_SECS: libc::c_int = 30;
    // how many unanswered probes before the connection is closed
    const KEEPALIVE_PROBES: libc::c_int = 4;

    stream.set_nodelay(true)?;

    let fd = stream.as_raw_fd();
    unsafe {
        let ret = libc::setsockopt(
            fd,
            libc::SOL_TCP,
            libc::TCP_KEEPIDLE,
            &KEEPALIVE_TIME_SECS as *const _ as *const libc::c_void,
            std::mem::size_of_val(&KEEPALIVE_TIME_SECS) as libc::socklen_t,
        );
        if ret != 0 {
            let err = io::Error::last_os_error();
            bail!("Failed to set TCP_KEEPIDLE: {}", err);
        }

        let ret = libc::setsockopt(
            fd,
            libc::SOL_TCP,
            libc::TCP_KEEPINTVL,
            &KEEPALIVE_INTERVAL_SECS as *const _ as *const libc::c_void,
            std::mem::size_of_val(&KEEPALIVE_INTERVAL_SECS) as libc::socklen_t,
        );
        if ret != 0 {
            let err = io::Error::last_os_error();
            bail!("Failed to set TCP_KEEPINTVL: {}", err);
        }

        let ret = libc::setsockopt(
            fd,
            libc::SOL_TCP,
            libc::TCP_KEEPCNT,
            &KEEPALIVE_PROBES as *const _ as *const libc::c_void,
            std::mem::size_of_val(&KEEPALIVE_PROBES) as libc::socklen_t,
        );
        if ret != 0 {
            let err = io::Error::last_os_error();
            bail!("Failed to set TCP_KEEPCNT: {}", err);
        }
    }

    Ok(())
}
