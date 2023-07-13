# Advanced Data Structures Project

Project for the [Advanced Data Structures](https://algo2.iti.kit.edu/4521.php) lecture during summer term 2023.

- Implements accelerated predecessor queries using the quasi-succinct Elias-Fano coding.
- Implements accelerated Range Minimum Queries (RMQs) using the naive `O(n²)` approach,
  the _sparse table_ `O(n log n)` approach, and an `O(n)` approach using cartesian trees.

## Build Requirements

Building the project requires an up-to-date version of Rust (tested with 1.70.0).
No further external dependencies or any crates are required to build the project.

## Build Instructions

The code can be built and run using Cargo.

```
cargo run --release (pd|rmq) <input-file> <output-file>
```

The binary can also be built and executed explicity as follows:

```
cargo build --release
./target/release/ads-project (pd|rmq) <input-file> <output-file>
```

Note that the code is built with `-Ctarget-cpu=native` (cf. GCC's `-march=native`) by default to 
  make use of x86 instruction set extensions like AVX and BMI (see `.cargo/config.toml`).

## Documentation

For an overview of what is implemented, see the `rustdoc` documentation.

```
cargo doc --open
```
