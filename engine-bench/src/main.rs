use engine_bench::{
    DEFAULT_TACTICAL_SUITE, ExternalMatchConfig, MatchConfig, SprtConfig, external_tsv_report,
    measure_throughput, run_external_opponent_match, run_fixed_opponent_match, run_tactical_suite,
    sprt_tsv_report, summarize, tactical_tsv_report, throughput_tsv_report,
};

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

const GAUNTLET_POSITIONS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
];

const EXTERNAL_SPRT_POSITIONS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3",
    "r1bq1rk1/pp1nbppp/2p1pn2/3p4/3P4/2NBPN2/PPQ1BPPP/R3K2R w KQ - 4 8",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "r1bq1rk1/1pp1bppp/p1np1n2/8/2BPP3/2N1BN2/PPQ2PPP/R3K2R w KQ - 2 10",
    "2r2rk1/pp1qbppp/2np1n2/8/2BPP3/2N1BN2/PPQ2PPP/2RR2K1 w - - 4 13",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PPQ1NPPP/R1B2RK1 w - - 2 9",
    "2r2rk1/pp1bqppp/2np1n2/8/2BPP3/2N1BN2/PPQ2PPP/2RR2K1 w - - 6 14",
    "r2q1rk1/pp1nbppp/2p1pn2/3p4/3P4/2NBPN2/PPQ1BPPP/2R2RK1 w - - 6 10",
    "r1bq1rk1/pp1nbppp/2p1pn2/3p4/3P4/2NBPN2/PPQ1BPPP/R3K2R b KQ - 5 8",
    "2r2rk1/pp1qbppp/2np1n2/8/2BPP3/2N1BN2/PPQ2PPP/R3K2R b KQ - 3 12",
    "r3r1k1/pp1qbppp/2np1n2/8/2BPP3/2N1BN2/PPQ2PPP/2RR2K1 w - - 8 15",
    "r1bq1rk1/pp1nbppp/2p1pn2/3p4/3P4/2NBPN2/PPQ1BPPP/2R2RK1 b - - 7 10",
    "2r2rk1/pp1qbppp/2np1n2/8/2BPP3/2N1BN2/PPQ2PPP/R3K2R w KQ - 5 13",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PPQ1NPPP/R1B2RK1 b - - 3 9",
    "r3r1k1/pp1qbppp/2np1n2/8/2BPP3/2N1BN2/PPQ2PPP/2RR2K1 b - - 7 14",
];

fn main() -> Result<(), String> {
    if std::env::args().nth(1).as_deref() == Some("tactical") {
        let results = run_tactical_suite(DEFAULT_TACTICAL_SUITE)?;
        print!("{}", tactical_tsv_report(&results));
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("gauntlet") {
        let records = run_fixed_opponent_match(GAUNTLET_POSITIONS, MatchConfig::default())?;
        print!("{}", sprt_tsv_report(summarize(&records), SprtConfig::default()));
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("external-sprt") {
        let config = ExternalMatchConfig::default();
        let records = run_external_opponent_match(EXTERNAL_SPRT_POSITIONS, &config)?;
        eprint!("{}", external_tsv_report(&records, &config));
        print!("{}", sprt_tsv_report(summarize(&records), SprtConfig::default()));
        return Ok(());
    }

    let samples = BENCHMARKS
        .iter()
        .map(|(fen, depth)| measure_throughput(fen, *depth))
        .collect::<Result<Vec<_>, _>>()?;
    print!("{}", throughput_tsv_report(&samples));
    Ok(())
}
