pub fn get_check_addresses(
    hostnames: Vec<String>,
    ports: Vec<String>,
    endpoint: &str,
) -> Vec<String> {
    hostnames
        .iter()
        .zip(ports.iter())
        .map(|(host, port)| format!("http://{}:{}/{}", host, port, endpoint))
        .collect::<Vec<String>>()
}
