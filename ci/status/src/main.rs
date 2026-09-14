#[allow(dead_code)]
mod prefetch {
    include!(concat!(env!("OUT_DIR"), "/prefetch.rs"));
}

fn main() {
    let home = std::path::PathBuf::from(std::env::args().nth(1).expect("studio home"));
    println!("{}", serde_json::to_string(&prefetch::status(&home)).unwrap());
}
