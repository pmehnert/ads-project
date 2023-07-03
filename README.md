# Advanced Data Structures Project

Project for the [Advanced Data Structures](algo2.iti.kit.edu/4521.php) lecture during summer term 2023.

## Build Requirements

Building the project requires an up-to-date version of Rust (tested with 1.70.0).
No further external dependencies or any crates are required to build the project.

## Build Instructions

To code can be built and run using Cargo.

```sh
cargo run --release [pd|rmq] <input-file> <output-file>
```

The binary can also be executed explicitly.

```sh
cargo build --release
./target/release/ads-project [pd|rmq] <input-file> <output-file>
```

Note that the code is built with `-Ctarget-cpu=native` (cf. `-march=native`) by default to 
  make use of x86 instruction set extensions like AVX and BMI (see `.cargo/config.toml`).

## Documentation

For an overview of what is implemented, see the `rustdoc` documentation.

```sh
cargo doc --open
```
