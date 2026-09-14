fn main() {
    let repo = std::env::var("PR_REPO").expect("PR_REPO");
    let src = std::path::Path::new(&repo).join("studio/src-tauri/src/prefetch.rs");
    println!("cargo:rerun-if-env-changed=PR_REPO");
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("prefetch.rs");
    let text = std::fs::read_to_string(src).unwrap();
    // Inner doc comments cannot sit inside include!; nothing else changes.
    let body: String = text.lines().filter(|l| !l.starts_with("//!")).map(|l| format!("{l}\n")).collect();
    std::fs::write(out, body).unwrap();
}
