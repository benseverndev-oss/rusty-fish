use std::{sync::Arc, time::Duration};

use engine_bench::{
    BaselineMode, DEFAULT_TACTICAL_SUITE, EvalSpsaConfig, ExternalMatchConfig, MatchConfig,
    MatchScore, SpsaConfig, SprtConfig,
    eval_params_from_tsv, eval_params_to_tsv, external_tsv_report, measure_throughput,
    random_opening_fens, run_eval_gate_fens, run_eval_spsa_campaign, run_external_opponent_match,
    run_fixed_opponent_match, run_mobility_gate, run_mobility_gate_fens, run_nnue_gauntlet,
    run_nnue_gauntlet_with_move_time_and_baseline,
    run_spsa_campaign, run_tactical_suite,
    spsa_tsv_report, sprt, sprt_tsv_report, summarize, tactical_tsv_report, throughput_tsv_report,
    gen_wdl_data_samples_from_reader, WdlSampleConfig,
    gen_eval_positions_from_reader, run_label_sf, run_label_fens,
    search_params_from_tsv, run_search_gate_fens,
};
use engine_bench::bench_harness::{
    BenchCompareConfig, BenchReportConfig, BudgetMode, EngineConfig, bench_compare_tsv_report,
    bench_full_report_text, bench_sweep_tsv_report, compare_openings, run_bench_compare,
    run_bench_report, run_bench_sweep,
};
use engine_bench::train::{generate_training_samples, train_nnue, TrainConfig};
use engine_search::{EvalParams, Nnue, SearchParams};

/// Shared middlegame sampling window for both `gen-wdl-data` and
/// `gen-eval-positions`. Single-sourced so the Stockfish teacher labels the
/// exact same position distribution the WDL data trains on — tuning one
/// without the other would silently drift the two apart.
const SAMPLE_MIN_PLY: u32 = 8;
const SAMPLE_END_TRIM: u32 = 5;

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
        // external-sprt [--opponent-elo N] [--openings N] [--movetime MS]:
        // play the bundled-net engine vs the external UCI opponent
        // (RUSTY_FISH_EXTERNAL_UCI), both colours per opening. --opponent-elo
        // weakens the opponent via UCI_LimitStrength to find the strength
        // crossover; --openings N uses N generated openings instead of the fixed
        // suite (more games = tighter estimate); --movetime sets both sides' budget.
        let mut config = ExternalMatchConfig::default();
        let mut openings: Option<usize> = None;
        let mut args = std::env::args().skip(2);
        while let Some(arg) = args.next() {
            let mut value = || {
                args.next()
                    .ok_or_else(|| format!("{arg} needs a value"))
            };
            match arg.as_str() {
                "--opponent-elo" => {
                    config.opponent_elo = Some(value()?.parse::<u32>().map_err(|_| "invalid --opponent-elo".to_string())?);
                }
                "--openings" => {
                    openings = Some(value()?.parse::<usize>().map_err(|_| "invalid --openings".to_string())?);
                }
                "--movetime" => {
                    let ms = value()?.parse::<u64>().map_err(|_| "invalid --movetime".to_string())?;
                    config.candidate_movetime = std::time::Duration::from_millis(ms);
                    config.opponent_movetime = std::time::Duration::from_millis(ms);
                }
                other => return Err(format!("unexpected argument: {other}")),
            }
        }
        // Own the FENs (generated or the fixed suite) so `&[&str]` refs stay valid.
        let owned: Vec<String> = match openings {
            Some(count) => random_opening_fens(count, 8, 0x5EED),
            None => EXTERNAL_SPRT_POSITIONS.iter().map(|s| s.to_string()).collect(),
        };
        let positions: Vec<&str> = owned.iter().map(String::as_str).collect();
        let records = run_external_opponent_match(&positions, &config)?;
        eprint!("{}", external_tsv_report(&records, &config));
        print!("{}", sprt_tsv_report(summarize(&records), SprtConfig::default()));
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("search-gate") {
        // search-gate <tuned_tsv_file> [openings] [depth]: out-of-sample validation
        // for a SPSA-tuned SearchParams — play the tuned params vs the default over
        // freshly generated openings (both colours) and emit an SPRT verdict. The
        // tuned TSV is the 8-value search-param vector (the last row of `spsa`).
        // Honours RUSTY_FISH_SPSA_NNUE (tune/validate against the shipped NNUE eval).
        let tuned_path = std::env::args()
            .nth(2)
            .ok_or_else(|| "usage: search-gate <tuned_tsv_file> [openings] [depth]".to_string())?;
        let openings = std::env::args().nth(3).and_then(|arg| arg.parse::<usize>().ok()).unwrap_or(64);
        let depth = std::env::args().nth(4).and_then(|arg| arg.parse::<u8>().ok()).unwrap_or(6);
        let tuned_tsv = std::fs::read_to_string(&tuned_path)
            .map_err(|error| format!("failed to read tuned params {tuned_path}: {error}"))?;
        let candidate = search_params_from_tsv(&tuned_tsv)?;
        // Fresh openings (distinct seed from the SPSA training set) => out of sample.
        let fens = random_opening_fens(openings, 8, 0xA11CE_5EED_C0DE);
        let records = run_search_gate_fens(&fens, candidate, SearchParams::default(), depth, 160)?;
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
    if std::env::args().nth(1).as_deref() == Some("spsa-eval") {
        // spsa-eval [iterations] [openings] [movetime_ms]: SPSA-tune the eval
        // weights via self-play (theta+ vs theta-, both mobility-on), then print
        // the tuned EvalParams to STDOUT as the 18-value vector TSV that
        // `eval-gate-file` parses, so the campaign output feeds the gate directly.
        // A per-iteration trace goes to stderr.
        let mut config = EvalSpsaConfig::default();
        if let Some(iterations) = arg_u32(2) {
            config.iterations = iterations as usize;
        }
        let openings = arg_u32(3).unwrap_or(64) as usize;
        if let Some(movetime_ms) = arg_u64(4) {
            config.move_time = Duration::from_millis(movetime_ms);
        }
        let fens = random_opening_fens(openings, 8, 0x5EED);
        let fen_refs: Vec<&str> = fens.iter().map(String::as_str).collect();
        let report = run_eval_spsa_campaign(&fen_refs, EvalParams::default(), config)?;
        for record in &report.iterations {
            eprintln!(
                "iter {:>3}: {}W {}D {}L score {:.3} | {}",
                record.iteration,
                record.score.wins,
                record.score.draws,
                record.score.losses,
                record.candidate_score_fraction,
                eval_params_to_tsv(&record.params),
            );
        }
        println!("{}", eval_params_to_tsv(&report.tuned));
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
        // gate-file <net> <depth> <openings_file> [move_time_ms] [champion|handcrafted]:
        // play the NNUE candidate over the file's openings; emit "W\tD\tL". The
        // baseline defaults to the bundled champion net (net-vs-net); pass
        // `handcrafted` to gate against the hand-crafted eval instead. The
        // deadline keeps one pathological search from stalling a full campaign
        // shard.
        let path = std::env::args()
            .nth(2)
            .ok_or_else(|| "usage: gate-file <net> <depth> <openings_file> [move_time_ms] [champion|handcrafted]".to_string())?;
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
        let mode = if std::env::args().any(|a| a == "handcrafted") {
            BaselineMode::Handcrafted
        } else {
            BaselineMode::Champion
        };
        let records = run_nnue_gauntlet_with_move_time_and_baseline(
            &fens,
            std::sync::Arc::new(net),
            config,
            move_time,
            mode,
        )?;
        let score = summarize(&records);
        println!("{}\t{}\t{}", score.wins, score.draws, score.losses);
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("mobility-gate-file") {
        // mobility-gate-file <openings_file> [movetime_ms]: play mobility-on vs
        // off over the file's openings, color-swapped; emit "W\tD\tL". The
        // shardable form so a Modal fan-out can play a slice per container.
        let openings_path = std::env::args()
            .nth(2)
            .ok_or_else(|| "usage: mobility-gate-file <openings_file> [movetime_ms]".to_string())?;
        let move_time = Duration::from_millis(arg_u64(3).unwrap_or(15));
        let contents = std::fs::read_to_string(&openings_path)
            .map_err(|error| format!("failed to read openings {openings_path}: {error}"))?;
        let fens: Vec<&str> = contents
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect();
        let records = run_mobility_gate_fens(&fens, move_time, 80)?;
        let score = summarize(&records);
        println!("{}\t{}\t{}", score.wins, score.draws, score.losses);
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("eval-gate-file") {
        // eval-gate-file <openings_file> <tuned_eval_tsv_file> [movetime_ms]:
        // read the tuned EvalParams from a TSV file (the 18-value vector, one
        // line) and play it (mobility on) vs the default eval (mobility off)
        // over the file's openings, color-swapped; emit "W\tD\tL". The shardable
        // form so a Modal fan-out can play a slice per container.
        let openings_path = std::env::args().nth(2).ok_or_else(|| {
            "usage: eval-gate-file <openings_file> <tuned_eval_tsv_file> [movetime_ms]".to_string()
        })?;
        let tuned_path = std::env::args().nth(3).ok_or_else(|| {
            "usage: eval-gate-file <openings_file> <tuned_eval_tsv_file> [movetime_ms]".to_string()
        })?;
        let move_time = Duration::from_millis(arg_u64(4).unwrap_or(15));
        let openings = std::fs::read_to_string(&openings_path)
            .map_err(|error| format!("failed to read openings {openings_path}: {error}"))?;
        let fens: Vec<&str> = openings
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect();
        let tuned_tsv = std::fs::read_to_string(&tuned_path)
            .map_err(|error| format!("failed to read tuned eval {tuned_path}: {error}"))?;
        let candidate = eval_params_from_tsv(&tuned_tsv)?;
        let records = run_eval_gate_fens(&fens, candidate, EvalParams::default(), move_time, 80)?;
        let score = summarize(&records);
        println!("{}\t{}\t{}", score.wins, score.draws, score.losses);
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("mobility-gate") {
        // mobility-gate [openings] [movetime_ms]: self-play SPRT, mobility on vs
        // off. Movetime bounds each move, so runtime <= openings*2*80*movetime.
        let openings = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(600);
        let move_ms: u64 = std::env::args().nth(3).and_then(|a| a.parse().ok()).unwrap_or(15);
        let records =
            run_mobility_gate(openings, 0xC0FFEE, Duration::from_millis(move_ms), 80)?;
        print!("{}", sprt_tsv_report(summarize(&records), SprtConfig::default()));
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

    if std::env::args().nth(1).as_deref() == Some("gen-wdl-data") {
        // gen-wdl-data <pgn_or_-> [--shard i/n] [--per-game N]: sample middlegame
        // positions from a Lichess PGN (path or `-` for stdin) and label each with
        // the side-to-move-relative game outcome. Emits one sample per line as
        // "target\town_csv\topp_csv" — the format train_nnue.py reads.
        let mut source: Option<String> = None;
        let mut shard = (0usize, 1usize);
        let mut per_game = 12usize;
        let mut args = std::env::args().skip(2);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--shard" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "usage: gen-wdl-data <pgn_or_-> [--shard i/n] [--per-game N]".to_string())?;
                    let (i, n) = value
                        .split_once('/')
                        .ok_or_else(|| format!("invalid --shard value (want i/n): {value}"))?;
                    let i = i
                        .parse::<usize>()
                        .map_err(|_| format!("invalid --shard index: {i}"))?;
                    let n = n
                        .parse::<usize>()
                        .map_err(|_| format!("invalid --shard count: {n}"))?;
                    if n == 0 || i >= n {
                        return Err(format!("invalid --shard {i}/{n}: need 0 <= i < n"));
                    }
                    shard = (i, n);
                }
                "--per-game" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "usage: gen-wdl-data <pgn_or_-> [--shard i/n] [--per-game N]".to_string())?;
                    per_game = value
                        .parse::<usize>()
                        .map_err(|_| format!("invalid --per-game value: {value}"))?;
                    if per_game == 0 {
                        return Err("invalid --per-game 0: need N >= 1".to_string());
                    }
                }
                _ => {
                    if source.is_some() {
                        return Err(format!("unexpected argument: {arg}"));
                    }
                    source = Some(arg);
                }
            }
        }
        let source = source
            .ok_or_else(|| "usage: gen-wdl-data <pgn_or_-> [--shard i/n] [--per-game N]".to_string())?;
        let config = WdlSampleConfig {
            min_ply: SAMPLE_MIN_PLY,
            end_trim: SAMPLE_END_TRIM,
            per_game,
            shard,
        };
        // Stream the PGN rather than buffering it: the real Lichess export is
        // ~16 GB decompressed and is piped via `zstdcat ... | gen-wdl-data -`,
        // so `read_to_string` would OOM the container. `Reader` is generic over
        // `io::Read`, and both `StdinLock` and `BufReader<File>` implement it.
        let samples = if source == "-" {
            gen_wdl_data_samples_from_reader(std::io::stdin().lock(), config)?
        } else {
            let file = std::fs::File::open(&source)
                .map_err(|error| format!("failed to open PGN {source}: {error}"))?;
            gen_wdl_data_samples_from_reader(std::io::BufReader::new(file), config)?
        };
        for sample in samples {
            println!(
                "{}\t{}\t{}",
                sample.target,
                join_usize(&sample.own),
                join_usize(&sample.opp)
            );
        }
        return Ok(());
    }

    if std::env::args().nth(1).as_deref() == Some("gen-eval-positions") {
        // gen-eval-positions <pgn_or_-> [--shard i/n] [--per-game N]: sample the
        // same middlegame positions as gen-wdl-data (path or `-` for stdin) but
        // emit each as "fen\town_csv\topp_csv" — the input label-sf reads to
        // re-evaluate each FEN with Stockfish.
        let mut source: Option<String> = None;
        let mut shard = (0usize, 1usize);
        let mut per_game = 12usize;
        let mut args = std::env::args().skip(2);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--shard" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "usage: gen-eval-positions <pgn_or_-> [--shard i/n] [--per-game N]".to_string())?;
                    let (i, n) = value
                        .split_once('/')
                        .ok_or_else(|| format!("invalid --shard value (want i/n): {value}"))?;
                    let i = i
                        .parse::<usize>()
                        .map_err(|_| format!("invalid --shard index: {i}"))?;
                    let n = n
                        .parse::<usize>()
                        .map_err(|_| format!("invalid --shard count: {n}"))?;
                    if n == 0 || i >= n {
                        return Err(format!("invalid --shard {i}/{n}: need 0 <= i < n"));
                    }
                    shard = (i, n);
                }
                "--per-game" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "usage: gen-eval-positions <pgn_or_-> [--shard i/n] [--per-game N]".to_string())?;
                    per_game = value
                        .parse::<usize>()
                        .map_err(|_| format!("invalid --per-game value: {value}"))?;
                    if per_game == 0 {
                        return Err("invalid --per-game 0: need N >= 1".to_string());
                    }
                }
                _ => {
                    if source.is_some() {
                        return Err(format!("unexpected argument: {arg}"));
                    }
                    source = Some(arg);
                }
            }
        }
        let source = source
            .ok_or_else(|| "usage: gen-eval-positions <pgn_or_-> [--shard i/n] [--per-game N]".to_string())?;
        let config = WdlSampleConfig {
            min_ply: SAMPLE_MIN_PLY,
            end_trim: SAMPLE_END_TRIM,
            per_game,
            shard,
        };
        // Stream the PGN rather than buffering it (see gen-wdl-data): the real
        // Lichess export is ~16 GB decompressed and is piped via
        // `zstdcat ... | gen-eval-positions -`.
        let samples = if source == "-" {
            gen_eval_positions_from_reader(std::io::stdin().lock(), config)?
        } else {
            let file = std::fs::File::open(&source)
                .map_err(|error| format!("failed to open PGN {source}: {error}"))?;
            gen_eval_positions_from_reader(std::io::BufReader::new(file), config)?
        };
        for sample in samples {
            println!(
                "{}\t{}\t{}",
                sample.fen,
                join_usize(&sample.own),
                join_usize(&sample.opp)
            );
        }
        return Ok(());
    }

    if std::env::args().nth(1).as_deref() == Some("label-sf") {
        // label-sf <positions_or_-> <nodes> [--engine PATH]: drive one persistent
        // Stockfish process (default /usr/games/stockfish, the Debian package
        // location on Modal) to relabel `fen\town\topp` lines (path or `-` for
        // stdin) with a fixed-node cp eval, printing `cp\town\topp` — the format
        // the trainer reads. One malformed line or scoring error is skipped, not
        // fatal, so a bad position can't kill a shard of tens of thousands.
        let mut source: Option<String> = None;
        let mut nodes: Option<u64> = None;
        let mut engine_path = "/usr/games/stockfish".to_string();
        let mut args = std::env::args().skip(2);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--engine" => {
                    engine_path = args
                        .next()
                        .ok_or_else(|| "usage: label-sf <positions_or_-> <nodes> [--engine PATH]".to_string())?;
                }
                _ => {
                    if source.is_none() {
                        source = Some(arg);
                    } else if nodes.is_none() {
                        nodes = Some(
                            arg.parse::<u64>()
                                .map_err(|_| format!("invalid nodes value: {arg}"))?,
                        );
                    } else {
                        return Err(format!("unexpected argument: {arg}"));
                    }
                }
            }
        }
        let source = source
            .ok_or_else(|| "usage: label-sf <positions_or_-> <nodes> [--engine PATH]".to_string())?;
        let nodes = nodes
            .ok_or_else(|| "usage: label-sf <positions_or_-> <nodes> [--engine PATH]".to_string())?;
        if nodes == 0 {
            return Err("invalid nodes 0: need nodes >= 1".to_string());
        }
        // Stream the positions rather than buffering (see gen-eval-positions):
        // a shard is piped via `gen-eval-positions - | label-sf - <nodes>`.
        if source == "-" {
            run_label_sf(std::io::stdin().lock(), nodes, &engine_path)?;
        } else {
            let file = std::fs::File::open(&source)
                .map_err(|error| format!("failed to open positions {source}: {error}"))?;
            run_label_sf(std::io::BufReader::new(file), nodes, &engine_path)?;
        }
        return Ok(());
    }

    if std::env::args().nth(1).as_deref() == Some("label-fens") {
        // label-fens <fen_eval_or_-> [--stm-relative]: convert a pre-evaluated
        // public dataset — `FEN<sep>eval` rows, tab- or comma-separated (path or
        // `-` for stdin) — into `cp\town\topp` training rows, taking the cp label
        // from the file instead of a Stockfish pass. This replaces the whole
        // `gen-eval-positions | label-sf` self-play + fixed-node fan-out for data
        // that is already scored. Evals are treated as White-relative (the common
        // public convention) and flipped to side-to-move POV; pass --stm-relative
        // when the source is already side-to-move. Malformed rows are skipped.
        let mut source: Option<String> = None;
        let mut white_relative = true;
        let mut args = std::env::args().skip(2);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--stm-relative" => white_relative = false,
                "--white-relative" => white_relative = true,
                _ => {
                    if source.is_some() {
                        return Err(format!("unexpected argument: {arg}"));
                    }
                    source = Some(arg);
                }
            }
        }
        let source = source
            .ok_or_else(|| "usage: label-fens <fen_eval_or_-> [--stm-relative]".to_string())?;
        if source == "-" {
            run_label_fens(std::io::stdin().lock(), white_relative)?;
        } else {
            let file = std::fs::File::open(&source)
                .map_err(|error| format!("failed to open positions {source}: {error}"))?;
            run_label_fens(std::io::BufReader::new(file), white_relative)?;
        }
        return Ok(());
    }
    // --- Rigorous A/B compare harness (equal-nodes, multi-TC, unified report) ---
    // Each subcommand pits two `EngineConfig`s over color-swapped openings under a
    // selectable budget. For now both configs are the default hand-crafted engine
    // (a baseline-vs-baseline sanity check that must report ~0 Elo); a future
    // learned-search toggle slots in as just another `EngineConfig`.
    if std::env::args().nth(1).as_deref() == Some("bench-compare") {
        // bench-compare <nodes|movetime|depth> <value> [openings] [max_plies]
        let mode = parse_budget_mode(2, 3)?;
        let openings = arg_u32(4).unwrap_or(16) as usize;
        let max_plies = arg_u32(5).unwrap_or(120);
        let candidate = EngineConfig::handcrafted("candidate");
        let baseline = EngineConfig::handcrafted("baseline");
        let fens = compare_openings(openings, BENCH_COMPARE_SEED);
        let report = run_bench_compare(
            &fens,
            &candidate,
            &baseline,
            BenchCompareConfig { mode, max_plies, sprt: SprtConfig::default() },
        )?;
        print!("{}", bench_compare_tsv_report(&report));
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("bench-sweep") {
        // bench-sweep [openings] [max_plies]: run the compare across several
        // budgets so a change can be checked for time-control dependence.
        let openings = arg_u32(2).unwrap_or(16) as usize;
        let max_plies = arg_u32(3).unwrap_or(120);
        let candidate = EngineConfig::handcrafted("candidate");
        let baseline = EngineConfig::handcrafted("baseline");
        let fens = compare_openings(openings, BENCH_COMPARE_SEED);
        let modes = [
            BudgetMode::Nodes(10_000),
            BudgetMode::Nodes(50_000),
            BudgetMode::Movetime(Duration::from_millis(20)),
            BudgetMode::Movetime(Duration::from_millis(50)),
        ];
        let reports =
            run_bench_sweep(&fens, &candidate, &baseline, &modes, max_plies, SprtConfig::default())?;
        print!("{}", bench_sweep_tsv_report(&reports));
        return Ok(());
    }
    if std::env::args().nth(1).as_deref() == Some("bench-report") {
        // bench-report [openings] [max_plies]: the "evaluate under all axes" entry
        // point — equal-nodes + equal-movetime matches, the multi-TC sweep, the
        // tactical suite, and a throughput measurement, in one consolidated block.
        let openings = arg_u32(2).unwrap_or(16) as usize;
        let max_plies = arg_u32(3).unwrap_or(120);
        let candidate = EngineConfig::handcrafted("candidate");
        let baseline = EngineConfig::handcrafted("baseline");
        let fens = compare_openings(openings, BENCH_COMPARE_SEED);
        let mut config = BenchReportConfig::default();
        config.max_plies = max_plies;
        let report = run_bench_report(&fens, &candidate, &baseline, config)?;
        print!("{}", bench_full_report_text(&report));
        return Ok(());
    }

    let samples = BENCHMARKS
        .iter()
        .map(|(fen, depth)| measure_throughput(fen, *depth))
        .collect::<Result<Vec<_>, _>>()?;
    print!("{}", throughput_tsv_report(&samples));
    Ok(())
}

/// Fixed seed for the compare openings so a `bench-compare`/`-sweep`/`-report`
/// run reproduces the same match given the same opening count.
const BENCH_COMPARE_SEED: u64 = 0xB0BA_CAFE;

/// Parses a `<nodes|movetime|depth> <value>` pair from the given argument
/// indices into a [`BudgetMode`].
fn parse_budget_mode(mode_index: usize, value_index: usize) -> Result<BudgetMode, String> {
    let kind = std::env::args().nth(mode_index).ok_or_else(|| {
        "usage: bench-compare <nodes|movetime|depth> <value> [openings] [max_plies]".to_string()
    })?;
    match kind.as_str() {
        "nodes" => {
            let value = arg_u64(value_index)
                .ok_or_else(|| "bench-compare nodes <N>: missing node budget".to_string())?;
            Ok(BudgetMode::Nodes(value))
        }
        "movetime" => {
            let value = arg_u64(value_index)
                .ok_or_else(|| "bench-compare movetime <MS>: missing movetime".to_string())?;
            Ok(BudgetMode::Movetime(Duration::from_millis(value)))
        }
        "depth" => {
            let value = arg_u32(value_index)
                .and_then(|d| u8::try_from(d).ok())
                .ok_or_else(|| "bench-compare depth <D>: missing/invalid depth".to_string())?;
            Ok(BudgetMode::Depth(value))
        }
        other => Err(format!(
            "unknown budget mode `{other}` (want nodes|movetime|depth)"
        )),
    }
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
