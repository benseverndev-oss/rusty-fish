//! Deterministic movegen throughput benchmark.
//!
//! `perft` walks the full legal-move tree to a fixed depth, so its node count is
//! an exact correctness check and its wall-clock time is a stable measure of
//! make/unmake + legal-move-generation throughput — no game-playing or search
//! randomness involved. Run with `cargo run --release --example perft_bench`.

use engine_core::Board;
use std::time::Instant;

fn bench(name: &str, fen: &str, depth: u32) {
    let mut board = Board::from_fen(fen).expect("valid fen");
    let start = Instant::now();
    let nodes = board.perft(depth);
    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64();
    let nps = nodes as f64 / secs;
    println!(
        "{name:<10} depth {depth}: {nodes:>12} nodes in {secs:>7.3}s  =  {:>8.2} Mnps",
        nps / 1_000_000.0
    );
}

fn main() {
    let depth: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    println!("perft throughput benchmark (depth {depth})");
    bench(
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        depth,
    );
    bench(
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        depth,
    );
}
