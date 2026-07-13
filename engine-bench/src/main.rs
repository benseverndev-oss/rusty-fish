use engine_bench::{measure_throughput, throughput_tsv_report};

const BENCHMARKS: &[(&str, u8)] = &[
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        4,
    ),
    (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        4,
    ),
];

fn main() -> Result<(), String> {
    let samples = BENCHMARKS
        .iter()
        .map(|(fen, depth)| measure_throughput(fen, *depth))
        .collect::<Result<Vec<_>, _>>()?;
    print!("{}", throughput_tsv_report(&samples));
    Ok(())
}
