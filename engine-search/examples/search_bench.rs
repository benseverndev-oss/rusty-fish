//! Fixed-depth search throughput benchmark (NNUE eval on).
//!
//! Unlike perft, this exercises the whole node cost of a real search — move
//! generation *and* NNUE evaluation, move ordering, and the transposition table
//! — so its nodes/sec is the number that actually tracks playing strength. Run
//! with `cargo run --release --example search_bench -- [depth]`.

use engine_core::Board;
use engine_search::{bundled_network, SearchLimits, Searcher};
use std::time::Instant;

const POSITIONS: &[(&str, &str)] = &[
    (
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ),
    (
        "midgame",
        "r1bq1rk1/pp1nbppp/2p1pn2/3p4/3P4/2NBPN2/PPQ1BPPP/R3K2R w KQ - 4 8",
    ),
    (
        "endgame",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ),
];

fn main() {
    let depth: u8 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    println!("search throughput benchmark (fixed depth {depth}, NNUE on)");

    let net = bundled_network();
    let mut total_nodes = 0u64;
    let mut total_secs = 0f64;

    for (name, fen) in POSITIONS {
        let board = Board::from_fen(fen).expect("valid fen");
        let mut searcher = Searcher::default();
        searcher.set_nnue(Some(net.clone()));
        let limits = SearchLimits {
            depth: Some(depth),
            ..Default::default()
        };
        let start = Instant::now();
        let result = searcher.search(&board, limits);
        let secs = start.elapsed().as_secs_f64();
        let nps = result.nodes as f64 / secs;
        println!(
            "{name:<10} {:>11} nodes in {secs:>7.3}s  =  {:>7.3} Mnps  (bestmove {})",
            result.nodes,
            nps / 1_000_000.0,
            result.best_move.map(|m| m.to_uci()).unwrap_or_else(|| "-".into()),
        );
        total_nodes += result.nodes;
        total_secs += secs;
    }

    println!(
        "TOTAL      {:>11} nodes in {total_secs:>7.3}s  =  {:>7.3} Mnps",
        total_nodes,
        total_nodes as f64 / total_secs / 1_000_000.0
    );
}
