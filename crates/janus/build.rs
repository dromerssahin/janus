use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../Cargo.lock");
    if let Ok(target) = std::env::var("TARGET") {
        println!("cargo:rustc-env=JANUS_TARGET={target}");
    }
    if let Ok(rustc) = std::env::var("RUSTC") {
        if let Ok(output) = Command::new(rustc).arg("--version").output() {
            if output.status.success() {
                println!(
                    "cargo:rustc-env=JANUS_RUSTC={}",
                    String::from_utf8_lossy(&output.stdout).trim()
                );
            }
        }
    }
}
