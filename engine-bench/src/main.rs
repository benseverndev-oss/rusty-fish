use std::{collections::{BTreeMap, BTreeSet}, path::Path, sync::Arc, time::Duration};

use engine_bench::{
    DEFAULT_TACTICAL_SUITE, ExternalMatchConfig, MatchConfig, MatchScore, SpsaConfig, SprtConfig,
    external_tsv_report, measure_throughput, random_opening_fens, run_external_opponent_match,
    run_fixed_opponent_match, run_nnue_gauntlet, run_nnue_gauntlet_with_move_time,
    run_spsa_campaign, run_tactical_suite,
    spsa_tsv_report, sprt, sprt_tsv_report, summarize, tactical_tsv_report, throughput_tsv_report,
};
use engine_bench::train::{generate_training_samples, train_nnue, TrainConfig};
use engine_bench::dataset::{
    DatasetManifest, PositionRecord, TEST_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT,
    canonical_fen, deduplicate_and_split, sha256_hex, write_manifest,
};
use engine_bench::stockfish::{StockfishConfig, calibrate_node_budget, label_positions};
use engine_search::{Nnue, SearchParams, active_features};
use engine_core::Board;

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
    if std::env::args().nth(1).as_deref() == Some("stockfish-calibrate") {
        return stockfish_calibrate();
    }
    if std::env::args().nth(1).as_deref() == Some("stockfish-label") {
        return stockfish_label();
    }
    if std::env::args().nth(1).as_deref() == Some("dataset-build") {
        return dataset_build();
    }
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
    if std::env::args().nth(1).as_deref() == Some("spsa") {
        let mut config = SpsaConfig::default();
        if let Some(iterations) = std::env::args().nth(2).and_then(|arg| arg.parse::<usize>().ok()) {
            config.iterations = iterations;
        }
        let report = run_spsa_campaign(EXTERNAL_SPRT_POSITIONS, SearchParams::default(), config)?;
        print!("{}", spsa_tsv_report(&report));
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("train") {
        let path = std::env::args()
            .nth(2)
            .unwrap_or_else(|| "artifacts/rusty-fish.rfnn".to_string());
        let plies = std::env::args()
            .nth(3)
            .and_then(|arg| arg.parse::<u32>().ok())
            .unwrap_or(48);
        // Optional 4th arg: label with a depth-N search instead of static eval.
        // A value of 0 means static labels.
        let label_depth = std::env::args()
            .nth(4)
            .and_then(|arg| arg.parse::<u8>().ok())
            .filter(|depth| *depth > 0);
        let mut config = TrainConfig::default();
        // Optional 5th/6th args tune the campaign: epochs and learning rate.
        if let Some(epochs) = std::env::args().nth(5).and_then(|arg| arg.parse::<usize>().ok()) {
            config.epochs = epochs.max(1);
        }
        if let Some(learning_rate) = std::env::args().nth(6).and_then(|arg| arg.parse::<f32>().ok())
        {
            if learning_rate > 0.0 {
                config.learning_rate = learning_rate;
            }
        }
        let samples =
            generate_training_samples(EXTERNAL_SPRT_POSITIONS, plies, config.seed, label_depth)?;
        let (network, report) = train_nnue(&samples, config)?;
        std::fs::write(&path, network.to_bytes())
            .map_err(|error| format!("failed to write network {path}: {error}"))?;
        let teacher = match label_depth {
            Some(depth) => format!("depth-{depth} search"),
            None => "static eval".to_string(),
        };
        eprintln!(
            "trained on {} samples ({teacher} labels, {} epochs, lr {}): loss {:.2} -> {:.2}; wrote {path}",
            report.samples,
            config.epochs,
            config.learning_rate,
            report.initial_loss,
            report.final_loss
        );
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("nnue-sprt") {
        let path = std::env::args()
            .nth(2)
            .ok_or_else(|| "usage: nnue-sprt <network> [depth]".to_string())?;
        let depth = std::env::args()
            .nth(3)
            .and_then(|arg| arg.parse::<u8>().ok())
            .unwrap_or(4);
        let net = Nnue::from_file(&path)?;
        let config = MatchConfig {
            candidate_depth: depth,
            baseline_depth: depth,
            max_plies: 120,
        };
        let records = run_nnue_gauntlet(EXTERNAL_SPRT_POSITIONS, Arc::new(net), config)?;
        let score = summarize(&records);
        print!("{}", sprt_tsv_report(score, SprtConfig::default()));
        let decision = sprt(score, SprtConfig::default()).map(|result| result.decision);
        eprintln!(
            "nnue-sprt: candidate (NNUE {path}) vs baseline (hand-crafted) at depth {depth}: \
             {}W {}D {}L; decision = {decision:?}",
            score.wins, score.draws, score.losses,
        );
        return Ok(());
    }

    // --- Sharded primitives for parallel (e.g. Modal) orchestration ---
    if std::env::args().nth(1).as_deref() == Some("gen-data") {
        // gen-data <plies> <label_depth> <seed>: emit labelled samples as TSV
        // (target, own-feature CSV, opp-feature CSV) for an external trainer.
        let plies = arg_u32(2).unwrap_or(48);
        let label_depth = arg_u32(3).and_then(|d| u8::try_from(d).ok()).filter(|d| *d > 0);
        let seed = arg_u64(4).unwrap_or(1);
        let samples = generate_training_samples(EXTERNAL_SPRT_POSITIONS, plies, seed, label_depth)?;
        for sample in &samples {
            println!(
                "{}\t{}\t{}",
                sample.target,
                join_usize(&sample.own),
                join_usize(&sample.opp)
            );
        }
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("gen-openings") {
        // gen-openings <count> <plies> <seed>: emit random opening FENs.
        let count = arg_u32(2).unwrap_or(64) as usize;
        let plies = arg_u32(3).unwrap_or(8);
        let seed = arg_u64(4).unwrap_or(1);
        for fen in random_opening_fens(count, plies, seed) {
            println!("{fen}");
        }
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("gate-file") {
        // gate-file <net> <depth> <openings_file> [move_time_ms]: play NNUE
        // candidate vs hand-crafted baseline over the file's openings; emit
        // "W\tD\tL". The deadline keeps one pathological search from stalling
        // a full campaign shard.
        let path = std::env::args()
            .nth(2)
            .ok_or_else(|| "usage: gate-file <net> <depth> <openings_file> [move_time_ms]".to_string())?;
        let depth = arg_u32(3).and_then(|d| u8::try_from(d).ok()).unwrap_or(4);
        let openings_path = std::env::args()
            .nth(4)
            .ok_or_else(|| "usage: gate-file <net> <depth> <openings_file>".to_string())?;
        let net = Nnue::from_file(&path)?;
        let contents = std::fs::read_to_string(&openings_path)
            .map_err(|error| format!("failed to read openings {openings_path}: {error}"))?;
        let fens: Vec<&str> = contents.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        let config = MatchConfig {
            candidate_depth: depth,
            baseline_depth: depth,
            max_plies: 160,
        };
        let move_time = Duration::from_millis(arg_u64(5).unwrap_or(100));
        let records = run_nnue_gauntlet_with_move_time(
            &fens,
            std::sync::Arc::new(net),
            config,
            move_time,
        )?;
        let score = summarize(&records);
        println!("{}\t{}\t{}", score.wins, score.draws, score.losses);
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("sprt") {
        // sprt <wins> <draws> <losses>: SPRT verdict from aggregated counts.
        let wins = arg_u32(2).unwrap_or(0);
        let draws = arg_u32(3).unwrap_or(0);
        let losses = arg_u32(4).unwrap_or(0);
        let score = MatchScore { wins, draws, losses };
        print!("{}", sprt_tsv_report(score, SprtConfig::default()));
        let decision = sprt(score, SprtConfig::default()).map(|result| result.decision);
        eprintln!(
            "sprt: {wins}W {draws}D {losses}L; elo {}; decision = {decision:?}",
            score.elo_difference().map_or_else(|| "n/a".to_string(), |elo| format!("{elo:.1}")),
        );
        return Ok(());
    }

    let samples = BENCHMARKS
        .iter()
        .map(|(fen, depth)| measure_throughput(fen, *depth))
        .collect::<Result<Vec<_>, _>>()?;
    print!("{}", throughput_tsv_report(&samples));
    Ok(())
}

fn stockfish_calibrate() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 { return Err("usage: stockfish-calibrate <manifest> <stockfish> <sha256> <out_config>".into()); }
    let fens = manifest_fens(Path::new(&args[2]), None)?;
    let mut config = StockfishConfig {
        binary: args[3].clone().into(), binary_sha256: args[4].clone(), hash_mb: 16,
        node_budget: 25_000, response_timeout: Duration::from_secs(30),
    };
    config.node_budget = calibrate_node_budget(&config, &fens)?;
    write_stockfish_config(Path::new(&args[5]), &config)
}

fn stockfish_label() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 { return Err("usage: stockfish-label <manifest> <split> <stockfish_config> <out_tsv>".into()); }
    let fens = manifest_fens(Path::new(&args[2]), Some(&args[3]))?;
    let config = read_stockfish_config(Path::new(&args[4]))?;
    let labels = label_positions(&config, &fens)?;
    let mut output = String::new();
    for label in labels {
        let fen = canonical_label_fen(&label.fen)?;
        let board = Board::from_fen(&fen)?;
        let own = join_usize(&active_features(&board, board.side_to_move));
        let opp = join_usize(&active_features(&board, board.side_to_move.opposite()));
        output.push_str(&format!("{}\t{}\t{}\t{}\t{}\n", label.score_cp, own, opp, fen, label.nodes));
    }
    let out = Path::new(&args[5]);
    if out.exists() { return Err(format!("refusing to overwrite label output {}", out.display())); }
    std::fs::write(out, output).map_err(|error| format!("failed to write {}: {error}", out.display()))
}

fn manifest_fens(manifest_path: &Path, requested_split: Option<&str>) -> Result<Vec<String>, String> {
    let manifest = engine_bench::dataset::read_manifest(manifest_path)?;
    let split = requested_split.unwrap_or(TRAIN_SPLIT);
    let index = match split { TRAIN_SPLIT => 0, VALIDATION_SPLIT => 1, TEST_SPLIT => 2, _ => return Err(format!("unknown dataset split: {split}")) };
    let shard = manifest_path.parent().ok_or_else(|| "manifest has no parent directory".to_string())?.join(format!("{split}.tsv"));
    let bytes = std::fs::read(&shard).map_err(|error| format!("failed to read split {}: {error}", shard.display()))?;
    if sha256_hex(&bytes) != manifest.shard_sha256[index] { return Err(format!("split FEN hash does not match manifest: {split}")); }
    let text = std::str::from_utf8(&bytes).map_err(|_| format!("split is not UTF-8: {}", shard.display()))?;
    if text.lines().next() != Some("fen\tsource") { return Err(format!("invalid split header: {split}")); }
    text.lines().skip(1).map(|line| line.split_once('\t').map(|(fen, _)| fen.to_string()).ok_or_else(|| format!("invalid split row: {line}"))).collect()
}

fn write_stockfish_config(path: &Path, config: &StockfishConfig) -> Result<(), String> {
    if path.exists() { return Err(format!("refusing to overwrite Stockfish config {}", path.display())); }
    let body = format!("stockfish_config\t1\nbinary\t{}\nbinary_sha256\t{}\nhash_mb\t{}\nnode_budget\t{}\nresponse_timeout_ms\t{}\n", config.binary.display(), config.binary_sha256, config.hash_mb, config.node_budget, config.response_timeout.as_millis());
    std::fs::write(path, body).map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn read_stockfish_config(path: &Path) -> Result<StockfishConfig, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let mut lines = text.lines();
    if lines.next() != Some("stockfish_config\t1") { return Err("unsupported Stockfish config format".into()); }
    let mut values = BTreeMap::new();
    for line in lines { let (key, value) = line.split_once('\t').ok_or_else(|| format!("invalid Stockfish config line: {line}"))?; if values.insert(key, value).is_some() { return Err(format!("duplicate Stockfish config field: {key}")); } }
    Ok(StockfishConfig {
        binary: values.remove("binary").ok_or_else(|| "Stockfish config missing binary".to_string())?.into(),
        binary_sha256: values.remove("binary_sha256").ok_or_else(|| "Stockfish config missing binary_sha256".to_string())?.to_string(),
        hash_mb: values.remove("hash_mb").ok_or_else(|| "Stockfish config missing hash_mb".to_string())?.parse().map_err(|_| "invalid hash_mb".to_string())?,
        node_budget: values.remove("node_budget").ok_or_else(|| "Stockfish config missing node_budget".to_string())?.parse().map_err(|_| "invalid node_budget".to_string())?,
        response_timeout: Duration::from_millis(values.remove("response_timeout_ms").ok_or_else(|| "Stockfish config missing response_timeout_ms".to_string())?.parse().map_err(|_| "invalid response_timeout_ms".to_string())?),
    })
}

fn canonical_label_fen(fen: &str) -> Result<String, String> {
    canonical_fen(fen)
}

fn dataset_build() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 8 && args.len() != 9 {
        return Err("usage: dataset-build <run_id> <out_dir> <random_count> <opening_count> <quiet_count> <seed> [--smoke]".to_string());
    }
    let run_id = &args[2];
    let out_dir = Path::new(&args[3]);
    let random_count = args[4].parse::<usize>().map_err(|_| "random_count must be an integer".to_string())?;
    let opening_count = args[5].parse::<usize>().map_err(|_| "opening_count must be an integer".to_string())?;
    let quiet_count = args[6].parse::<usize>().map_err(|_| "quiet_count must be an integer".to_string())?;
    let seed = args[7].parse::<u64>().map_err(|_| "seed must be an integer".to_string())?;
    let smoke = args.get(8).is_some_and(|arg| arg == "--smoke");
    if args.len() == 9 && !smoke {
        return Err("the only optional dataset-build flag is --smoke".to_string());
    }
    let total = random_count.saturating_add(opening_count).saturating_add(quiet_count);
    if (!smoke && (random_count, opening_count, quiet_count) != (400_000, 400_000, 200_000))
        || (smoke && total > 1_000)
    {
        return Err("dataset-build requires counts 400000 400000 200000; --smoke permits a total of at most 1000".to_string());
    }

    let mut records = Vec::with_capacity(total);
    append_records(&mut records, "random", random_count, 8, seed);
    append_records(&mut records, "opening", opening_count, 16, seed ^ 0x9E37_79B9_7F4A_7C15);
    append_records(&mut records, "quiet", quiet_count, 24, seed ^ 0xD1B5_4A32_D192_ED03);
    let splits = deduplicate_and_split(records)?;
    let actual_source_counts = splits.values().flatten().fold(BTreeMap::new(), |mut counts, record| {
        *counts.entry(record.source.clone()).or_insert(0_usize) += 1;
        counts
    });
    let expected_source_counts = BTreeMap::from([
        ("random".to_string(), random_count),
        ("opening".to_string(), opening_count),
        ("quiet".to_string(), quiet_count),
    ]);
    if actual_source_counts != expected_source_counts || splits.values().map(Vec::len).sum::<usize>() != total {
        return Err("dataset generation did not preserve the requested unique source composition".into());
    }
    reserve_output_directory(out_dir)?;

    let mut source_counts = BTreeMap::new();
    let mut split_counts = BTreeMap::new();
    let mut shard_sha256 = Vec::new();
    let mut dataset_bytes = Vec::new();
    for split in [TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT] {
        let records = splits.get(split).expect("all dataset splits are initialized");
        let mut shard = String::from("fen\tsource\n");
        for record in records {
            shard.push_str(&record.fen);
            shard.push('\t');
            shard.push_str(&record.source);
            shard.push('\n');
            *source_counts.entry(record.source.clone()).or_insert(0) += 1;
        }
        split_counts.insert(split.to_string(), records.len());
        shard_sha256.push(sha256_hex(shard.as_bytes()));
        dataset_bytes.extend_from_slice(shard.as_bytes());
        std::fs::write(out_dir.join(format!("{split}.tsv")), shard)
            .map_err(|error| format!("failed to write {split} shard: {error}"))?;
    }
    let manifest = DatasetManifest {
        run_id: run_id.to_string(),
        source_counts,
        split_counts,
        shard_sha256,
        dataset_sha256: sha256_hex(&dataset_bytes),
        stockfish_config_sha256: None,
    };
    write_manifest(&out_dir.join("manifest.tsv"), &manifest)
}

fn reserve_output_directory(path: &Path) -> Result<(), String> {
    std::fs::create_dir(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::AlreadyExists {
            format!("refusing to modify existing or reserved dataset output {}", path.display())
        } else {
            format!("failed to reserve {}: {error}", path.display())
        }
    })
}

fn append_records(records: &mut Vec<PositionRecord>, source: &str, count: usize, plies: u32, seed: u64) {
    let target_len = records.len().saturating_add(count);
    let mut seen: BTreeSet<String> = records.iter().map(|record| record.fen.clone()).collect();
    let mut batch = 0_u64;
    while records.len() < target_len {
        let remaining = target_len - records.len();
        for fen in source_fens(source, remaining.saturating_mul(2).max(16), plies, seed.wrapping_add(batch)) {
            if let Ok(fen) = canonical_fen(&fen)
                && seen.insert(fen.clone())
            {
                records.push(PositionRecord {
                    fen,
                    source: source.to_string(),
                });
                if records.len() == target_len {
                    return;
                }
            }
        }
        batch = batch.wrapping_add(1);
    }
}

fn source_fens(source: &str, count: usize, plies: u32, seed: u64) -> Vec<String> {
    match source {
        // Random samples use varying legal-walk lengths; opening samples deliberately use the
        // established opening generator; quiet samples are admitted only after a quiet move.
        "random" => random_opening_fens(count, (seed as u32 % plies.max(1)).max(1), seed),
        "opening" => random_opening_fens(count, plies.max(12), seed),
        "quiet" => quiet_walk_fens(count, plies, seed),
        _ => Vec::new(),
    }
}

fn quiet_walk_fens(count: usize, plies: u32, mut seed: u64) -> Vec<String> {
    let mut output = Vec::with_capacity(count);
    for _ in 0..count.saturating_mul(4).max(16) {
        let mut board = Board::startpos();
        for _ in 0..plies.max(1) {
            let moves = board.generate_legal_move_list();
            let quiet: Vec<_> = moves.as_slice().iter().copied().filter(|mv| board.piece_at(mv.to).is_none() && mv.promotion.is_none()).collect();
            if quiet.is_empty() { break; }
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mv = quiet[(seed as usize) % quiet.len()];
            if board.make_move(mv).is_err() { break; }
        }
        if !board.in_check(board.side_to_move) { output.push(board.to_fen()); }
        if output.len() == count { break; }
    }
    output
}

fn arg_u32(index: usize) -> Option<u32> {
    std::env::args().nth(index).and_then(|arg| arg.parse::<u32>().ok())
}

fn arg_u64(index: usize) -> Option<u64> {
    std::env::args().nth(index).and_then(|arg| arg.parse::<u64>().ok())
}

fn join_usize(values: &[usize]) -> String {
    values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::{append_records, canonical_label_fen, deduplicate_and_split, reserve_output_directory};

    #[test]
    fn generated_dataset_records_exclude_terminal_positions() {
        let mut records = Vec::new();
        append_records(&mut records, "random", 400, 8, 1);
        append_records(&mut records, "opening", 400, 16, 1 ^ 0x9E37_79B9_7F4A_7C15);
        append_records(&mut records, "quiet", 200, 24, 1 ^ 0xD1B5_4A32_D192_ED03);
        assert_eq!(records.len(), 1_000);
        assert!(deduplicate_and_split(records).is_ok());
    }

    #[test]
    fn output_directory_reservation_rejects_an_existing_path() {
        let path = std::env::temp_dir().join(format!("rusty-fish-reservation-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        reserve_output_directory(&path).unwrap();
        assert!(reserve_output_directory(&path).is_err());
        std::fs::remove_dir(path).unwrap();
    }

    #[test]
    fn label_fens_are_canonicalized_and_validated() {
        let canonical = canonical_label_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 00 01").unwrap();
        assert_eq!(canonical, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(canonical_label_fen("8/8/8/8/8/8/8/4K3 w - - 0 1").is_err());
    }
}
