use cudarc::driver::{result::{event, stream}, CudaDevice};

fn main(){
    let device = CudaDevice::new(0).unwrap();

    // event::record(event, stream)
    // stream::wait_event(stream, event, flags)
}