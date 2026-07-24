use std::io::{BufRead, BufReader, Write};
use std::ops::ControlFlow;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

use engine_core::{Board, Color, GameStatus};
use engine_search::{
    EvalParams, Nnue, SearchLimits, SearchParams, Searcher, TELEMETRY_TSV_HEADER, TaperedScore,
    active_features,
};
use pgn_reader::shakmaty::{Chess, Position, uci::UciMove};
use pgn_reader::{RawTag, Reader, SanPlus, Visitor};

pub mod bench_harness;
pub mod train;

#[derive(Clone, Debug)]
pub struct ThroughputSample {
    pub fen: String,
    pub depth: u8,
    pub nodes: u64,
    pub elapsed: Duration,
    pub nodes_per_second: u64,
}

pub fn measure_throughput(fen: &str, depth: u8) -> Result<ThroughputSample, String> {
    let board = Board::from_fen(fen)?;
    let mut searcher = Searcher::default();
    let result = searcher.search(
        &board,
        SearchLimits {
            depth: Some(depth),
            ..SearchLimits::default()
        },
    );
    let elapsed_nanos = result.elapsed.as_nanos().max(1);
    let nodes_per_second = (u128::from(result.nodes) * 1_000_000_000 / elapsed_nanos) as u64;

    Ok(ThroughputSample {
        fen: fen.to_string(),
        depth,
        nodes: result.nodes,
        elapsed: result.elapsed,
        nodes_per_second,
    })
}

pub fn throughput_tsv_report(samples: &[ThroughputSample]) -> String {
    let mut report =
        "engine_version\tdepth\tnodes\telapsed_ms\tnodes_per_second\tfen\n".to_string();
    for sample in samples {
        report.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\n",
            env!("CARGO_PKG_VERSION"),
            sample.depth,
            sample.nodes,
            sample.elapsed.as_millis(),
            sample.nodes_per_second,
            sample.fen,
        ));
    }
    report
}

#[derive(Clone, Copy, Debug)]
pub struct TacticalCase {
    pub name: &'static str,
    pub fen: &'static str,
    pub expected_move: &'static str,
    pub depth: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TacticalResult {
    pub name: &'static str,
    pub expected_move: &'static str,
    pub actual_move: Option<String>,
    pub depth: u8,
    pub solved: bool,
}

pub const DEFAULT_TACTICAL_SUITE: &[TacticalCase] = &[
    TacticalCase {
        name: "mate_in_one",
        fen: "6k1/5ppp/8/8/8/5Q2/6PP/6K1 w - - 0 1",
        expected_move: "f3a8",
        depth: 2,
    },
    TacticalCase {
        name: "hanging_queen",
        fen: "4k3/8/8/8/4q3/8/4Q3/4K3 w - - 0 1",
        expected_move: "e2e4",
        depth: 2,
    },
    TacticalCase {
        name: "check_evasion",
        fen: "4k3/8/8/8/8/8/4q3/4K3 w - - 0 1",
        expected_move: "e1e2",
        depth: 2,
    },
];

pub fn run_tactical_suite(cases: &[TacticalCase]) -> Result<Vec<TacticalResult>, String> {
    cases
        .iter()
        .map(|case| {
            let board = Board::from_fen(case.fen)?;
            let mut searcher = Searcher::default();
            let actual_move = searcher
                .search(
                    &board,
                    SearchLimits {
                        depth: Some(case.depth),
                        ..SearchLimits::default()
                    },
                )
                .best_move
                .map(|mv| mv.to_uci());
            Ok(TacticalResult {
                name: case.name,
                expected_move: case.expected_move,
                solved: actual_move.as_deref() == Some(case.expected_move),
                actual_move,
                depth: case.depth,
            })
        })
        .collect()
}

pub fn tactical_solve_rate(results: &[TacticalResult]) -> Option<f64> {
    (!results.is_empty()).then(|| {
        results.iter().filter(|result| result.solved).count() as f64 / results.len() as f64
    })
}

pub fn tactical_tsv_report(results: &[TacticalResult]) -> String {
    let solve_rate = tactical_solve_rate(results).unwrap_or(0.0);
    let mut report =
        "engine_version\tcase\tdepth\texpected_move\tactual_move\tsolved\tsolve_rate\n".to_string();
    for result in results {
        report.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{solve_rate:.6}\n",
            env!("CARGO_PKG_VERSION"),
            result.name,
            result.depth,
            result.expected_move,
            result.actual_move.as_deref().unwrap_or(""),
            result.solved,
        ));
    }
    report
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MatchScore {
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
}

impl MatchScore {
    pub fn games(self) -> u32 {
        self.wins + self.draws + self.losses
    }

    pub fn score_fraction(self) -> Option<f64> {
        let games = self.games();
        (games > 0).then(|| (f64::from(self.wins) + f64::from(self.draws) * 0.5) / f64::from(games))
    }

    pub fn elo_difference(self) -> Option<f64> {
        match self.score_fraction() {
            None => Some(0.0),
            Some(score) if score <= 0.0 || score >= 1.0 => None,
            Some(score) => Some(400.0 * (score / (1.0 - score)).log10()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SprtConfig {
    pub elo0: f64,
    pub elo1: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl Default for SprtConfig {
    fn default() -> Self {
        Self {
            elo0: 0.0,
            elo1: 5.0,
            alpha: 0.05,
            beta: 0.05,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SprtDecision {
    AcceptH0,
    AcceptH1,
    Continue,
}

#[derive(Clone, Copy, Debug)]
pub struct SprtResult {
    pub log_likelihood_ratio: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub decision: SprtDecision,
}

pub fn sprt(score: MatchScore, config: SprtConfig) -> Option<SprtResult> {
    let games = score.games();
    (games > 0).then(|| {
        let draw_rate = f64::from(score.draws) / f64::from(games);
        let probabilities = |elo: f64| {
            let expected_score = 1.0 / (1.0 + 10_f64.powf(-elo / 400.0));
            let win = (expected_score - draw_rate * 0.5).clamp(1e-12, 1.0 - 1e-12);
            let loss = (1.0 - draw_rate - win).clamp(1e-12, 1.0 - 1e-12);
            (win, draw_rate.clamp(1e-12, 1.0 - 1e-12), loss)
        };
        let (win0, draw0, loss0) = probabilities(config.elo0);
        let (win1, draw1, loss1) = probabilities(config.elo1);
        let llr = f64::from(score.wins) * (win1 / win0).ln()
            + f64::from(score.draws) * (draw1 / draw0).ln()
            + f64::from(score.losses) * (loss1 / loss0).ln();
        let lower_bound = (config.beta / (1.0 - config.alpha)).ln();
        let upper_bound = ((1.0 - config.beta) / config.alpha).ln();
        let decision = if llr <= lower_bound {
            SprtDecision::AcceptH0
        } else if llr >= upper_bound {
            SprtDecision::AcceptH1
        } else {
            SprtDecision::Continue
        };
        SprtResult {
            log_likelihood_ratio: llr,
            lower_bound,
            upper_bound,
            decision,
        }
    })
}

pub fn sprt_tsv_report(score: MatchScore, config: SprtConfig) -> String {
    let result = sprt(score, config);
    format!(
        "engine_version\twins\tdraws\tlosses\telo_estimate\telo0\telo1\talpha\tbeta\tllr\tlower_bound\tupper_bound\tdecision\n{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
        env!("CARGO_PKG_VERSION"),
        score.wins,
        score.draws,
        score.losses,
        score.elo_difference().map_or_else(|| "".to_string(), |elo| format!("{elo:.2}")),
        config.elo0,
        config.elo1,
        config.alpha,
        config.beta,
        result.map_or_else(|| "".to_string(), |result| format!("{:.6}", result.log_likelihood_ratio)),
        result.map_or_else(|| "".to_string(), |result| format!("{:.6}", result.lower_bound)),
        result.map_or_else(|| "".to_string(), |result| format!("{:.6}", result.upper_bound)),
        result.map_or_else(|| "".to_string(), |result| format!("{:?}", result.decision)),
    )
}

#[derive(Clone, Copy, Debug)]
pub struct MatchConfig {
    pub candidate_depth: u8,
    pub baseline_depth: u8,
    pub max_plies: u32,
}

#[derive(Clone, Debug)]
pub struct ExternalMatchConfig {
    pub uci_path: Option<String>,
    pub candidate_movetime: Duration,
    pub candidate_move_overhead: Duration,
    pub opponent_movetime: Duration,
    pub max_plies: u32,
    pub response_timeout: Duration,
    /// When set, the opponent is weakened to approximately this Elo via
    /// `UCI_LimitStrength` / `UCI_Elo` (e.g. to benchmark the candidate against a
    /// deliberately weakened Stockfish and find the strength crossover). `None`
    /// plays the opponent at full strength.
    pub opponent_elo: Option<u32>,
}

impl Default for ExternalMatchConfig {
    fn default() -> Self {
        Self {
            uci_path: std::env::var("RUSTY_FISH_EXTERNAL_UCI").ok(),
            candidate_movetime: Duration::from_millis(100),
            candidate_move_overhead: Duration::from_millis(10),
            opponent_movetime: Duration::from_millis(100),
            max_plies: 160,
            response_timeout: Duration::from_secs(10),
            opponent_elo: None,
        }
    }
}

impl ExternalMatchConfig {
    pub fn validate(&self) -> Result<(), String> {
        match self.uci_path.as_deref().filter(|path| !path.trim().is_empty()) {
            Some(_) => Ok(()),
            None => Err("RUSTY_FISH_EXTERNAL_UCI must name an external UCI executable".to_string()),
        }
    }
}

pub fn external_match_game_count(position_count: usize) -> usize {
    position_count.saturating_mul(2)
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            candidate_depth: 4,
            baseline_depth: 3,
            max_plies: 160,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameOutcome {
    Win,
    Draw,
    Loss,
}

#[derive(Clone, Debug)]
pub struct GameRecord {
    pub fen: String,
    pub candidate_color: Color,
    pub outcome: GameOutcome,
    pub plies: u32,
}

pub fn run_fixed_opponent_match(
    positions: &[&str],
    config: MatchConfig,
) -> Result<Vec<GameRecord>, String> {
    let mut records = Vec::with_capacity(positions.len() * 2);
    for fen in positions {
        for candidate_color in [Color::White, Color::Black] {
            records.push(play_game(fen, candidate_color, config)?);
        }
    }
    Ok(records)
}

/// Which opponent the NNUE candidate is gated against. `Champion` (the default)
/// plays the candidate net against the bundled champion net that
/// `Searcher::default()` installs — a net-vs-net gate. `Handcrafted` disables
/// NNUE on the baseline so the candidate plays the hand-crafted evaluation.
#[derive(Clone, Copy)]
pub enum BaselineMode {
    Champion,
    Handcrafted,
}

/// Builds the gauntlet baseline searcher for `mode`. `Searcher::default()`
/// installs the bundled champion net; `Handcrafted` strips it back to the
/// hand-crafted evaluation.
fn baseline_searcher(mode: BaselineMode) -> Searcher {
    let mut baseline = Searcher::default(); // Default installs the bundled champion net.
    if matches!(mode, BaselineMode::Handcrafted) {
        baseline.set_nnue(None);
    }
    baseline
}

/// Plays the NNUE-equipped engine (candidate) against the bundled champion net
/// (baseline) over each position and both colours. This is the SPRT gate that
/// decides whether a trained network actually beats the current engine.
pub fn run_nnue_gauntlet(
    positions: &[&str],
    net: Arc<Nnue>,
    config: MatchConfig,
) -> Result<Vec<GameRecord>, String> {
    run_nnue_gauntlet_with_optional_move_time(positions, net, config, None, BaselineMode::Champion)
}

/// Plays a bounded NNUE gauntlet. Every search receives the same per-move
/// deadline so an unusually expensive position cannot stall a whole campaign.
pub fn run_nnue_gauntlet_with_move_time(
    positions: &[&str],
    net: Arc<Nnue>,
    config: MatchConfig,
    move_time: Duration,
) -> Result<Vec<GameRecord>, String> {
    run_nnue_gauntlet_with_optional_move_time(
        positions,
        net,
        config,
        Some(move_time),
        BaselineMode::Champion,
    )
}

/// Plays a bounded NNUE gauntlet against the chosen `BaselineMode` (champion net
/// or hand-crafted eval). This is the entry `gate-file` calls so the gate can
/// pick its baseline.
pub fn run_nnue_gauntlet_with_move_time_and_baseline(
    positions: &[&str],
    net: Arc<Nnue>,
    config: MatchConfig,
    move_time: Duration,
    mode: BaselineMode,
) -> Result<Vec<GameRecord>, String> {
    run_nnue_gauntlet_with_optional_move_time(positions, net, config, Some(move_time), mode)
}

fn run_nnue_gauntlet_with_optional_move_time(
    positions: &[&str],
    net: Arc<Nnue>,
    config: MatchConfig,
    move_time: Option<Duration>,
    mode: BaselineMode,
) -> Result<Vec<GameRecord>, String> {
    let mut records = Vec::with_capacity(positions.len() * 2);
    for fen in positions {
        for candidate_color in [Color::White, Color::Black] {
            records.push(play_nnue_game(fen, candidate_color, &net, config, move_time, mode)?);
        }
    }
    Ok(records)
}

/// Generates `count` distinct opening positions by walking random legal moves
/// from the start position. Used to give a parallel SPRT gate enough opening
/// diversity for a decisive verdict (the engines are deterministic, so each
/// opening yields one game per colour).
pub fn random_opening_fens(count: usize, plies: u32, seed: u64) -> Vec<String> {
    let mut fens = Vec::with_capacity(count);
    for index in 0..count {
        let mut rng = SplitMix64::new(seed ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut board = Board::startpos();
        for _ in 0..plies {
            let moves = board.generate_legal_move_list();
            if moves.is_empty() {
                break;
            }
            let choice = (rng.next_u64() % moves.as_slice().len() as u64) as usize;
            let mv = moves.as_slice()[choice];
            if board.make_move(mv).is_err() {
                break;
            }
        }
        fens.push(board.to_fen());
    }
    fens
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn play_nnue_game(
    fen: &str,
    candidate_color: Color,
    net: &Arc<Nnue>,
    config: MatchConfig,
    move_time: Option<Duration>,
    mode: BaselineMode,
) -> Result<GameRecord, String> {
    let mut board = Board::from_fen(fen)?;
    let mut candidate = Searcher::default();
    candidate.set_nnue(Some(Arc::clone(net)));
    let mut baseline = baseline_searcher(mode);
    for ply in 0..config.max_plies {
        let (depth, searcher) = if board.side_to_move == candidate_color {
            (config.candidate_depth, &mut candidate)
        } else {
            (config.baseline_depth, &mut baseline)
        };
        let result = searcher.search(
            &board,
            SearchLimits {
                depth: Some(depth),
                movetime: move_time,
                ..SearchLimits::default()
            },
        );
        let Some(mv) = result.best_move else {
            return Ok(GameRecord {
                fen: fen.to_string(),
                candidate_color,
                outcome: outcome_from_status(board.game_status(), candidate_color),
                plies: ply,
            });
        };
        board.make_move(mv)?;
    }
    Ok(GameRecord {
        fen: fen.to_string(),
        candidate_color,
        outcome: GameOutcome::Draw,
        plies: config.max_plies,
    })
}

pub fn run_external_opponent_match(
    positions: &[&str],
    config: &ExternalMatchConfig,
) -> Result<Vec<GameRecord>, String> {
    config.validate()?;
    let mut records = Vec::with_capacity(external_match_game_count(positions.len()));
    for fen in positions {
        for candidate_color in [Color::White, Color::Black] {
            records.push(play_external_game(fen, candidate_color, config)?);
        }
    }
    Ok(records)
}

pub fn summarize(records: &[GameRecord]) -> MatchScore {
    records
        .iter()
        .fold(MatchScore::default(), |mut score, record| {
            match record.outcome {
                GameOutcome::Win => score.wins += 1,
                GameOutcome::Draw => score.draws += 1,
                GameOutcome::Loss => score.losses += 1,
            }
            score
        })
}

pub fn tsv_report(records: &[GameRecord], config: MatchConfig) -> String {
    let mut report = format!(
        "engine_version\tcandidate_depth\tbaseline_depth\tmax_plies\tfen\tcandidate_color\toutcome\tplies\n"
    );
    for record in records {
        report.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{:?}\t{:?}\t{}\n",
            env!("CARGO_PKG_VERSION"),
            config.candidate_depth,
            config.baseline_depth,
            config.max_plies,
            record.fen,
            record.candidate_color,
            record.outcome,
            record.plies,
        ));
    }
    report
}

pub fn external_tsv_report(records: &[GameRecord], config: &ExternalMatchConfig) -> String {
    let opponent = config.uci_path.as_deref().unwrap_or("");
    let mut report = "engine_version\topponent_uci\tcandidate_movetime_ms\topponent_movetime_ms\tmax_plies\tfen\tcandidate_color\toutcome\tplies\n".to_string();
    for record in records {
        report.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{:?}\t{:?}\t{}\n",
            env!("CARGO_PKG_VERSION"),
            opponent,
            config.candidate_movetime.as_millis(),
            config.opponent_movetime.as_millis(),
            config.max_plies,
            record.fen,
            record.candidate_color,
            record.outcome,
            record.plies,
        ));
    }
    report
}

fn play_game(fen: &str, candidate_color: Color, config: MatchConfig) -> Result<GameRecord, String> {
    play_parameter_game(
        fen,
        candidate_color,
        SearchParams::default(),
        EvalParams::default(),
        SearchParams::default(),
        EvalParams::default(),
        config,
    )
}

/// Plays one self-play game between two parameter sets. The candidate searches
/// with `candidate_params`/`candidate_eval`, the baseline with
/// `baseline_params`/`baseline_eval`; this is the objective the SPSA tuner
/// optimises. Per-side `EvalParams` let a match pit two eval configurations.
pub fn play_parameter_game(
    fen: &str,
    candidate_color: Color,
    candidate_params: SearchParams,
    candidate_eval: EvalParams,
    baseline_params: SearchParams,
    baseline_eval: EvalParams,
    config: MatchConfig,
) -> Result<GameRecord, String> {
    let mut board = Board::from_fen(fen)?;
    let mut candidate = Searcher::default();
    if !spsa_keep_nnue() {
        candidate.set_nnue(None);
    }
    candidate.set_search_params(candidate_params);
    candidate.set_eval_params(candidate_eval);
    let mut baseline = Searcher::default();
    if !spsa_keep_nnue() {
        baseline.set_nnue(None);
    }
    baseline.set_search_params(baseline_params);
    baseline.set_eval_params(baseline_eval);
    for ply in 0..config.max_plies {
        let (depth, searcher) = if board.side_to_move == candidate_color {
            (config.candidate_depth, &mut candidate)
        } else {
            (config.baseline_depth, &mut baseline)
        };
        let result = searcher.search(
            &board,
            SearchLimits {
                depth: Some(depth),
                ..SearchLimits::default()
            },
        );
        let Some(mv) = result.best_move else {
            return Ok(GameRecord {
                fen: fen.to_string(),
                candidate_color,
                outcome: outcome_from_status(board.game_status(), candidate_color),
                plies: ply,
            });
        };
        board.make_move(mv)?;
    }
    Ok(GameRecord {
        fen: fen.to_string(),
        candidate_color,
        outcome: GameOutcome::Draw,
        plies: config.max_plies,
    })
}

/// A gate searcher: the given params plus a trimmed move overhead so a small
/// `movetime` budget is actually spent searching rather than swallowed by the
/// default 25 ms overhead (there is no GUI latency to reserve for here).
/// The search-param SPSA and its gate compare with the hand-crafted eval by
/// default (fast, deterministic). But once an NNUE is the shipped default, the
/// search params should be tuned against the eval the engine actually uses:
/// setting `RUSTY_FISH_SPSA_NNUE=1` keeps the bundled net on for these matches.
fn spsa_keep_nnue() -> bool {
    std::env::var("RUSTY_FISH_SPSA_NNUE").is_ok_and(|value| !value.is_empty() && value != "0")
}

fn gate_searcher(params: SearchParams, eval: EvalParams) -> Searcher {
    let mut searcher = Searcher::default();
    if !spsa_keep_nnue() {
        searcher.set_nnue(None);
    }
    searcher.set_search_params(params);
    searcher.set_eval_params(eval);
    let mut options = searcher.options().clone();
    options.move_overhead = Duration::from_millis(3);
    searcher.set_options(options);
    searcher
}

/// Plays one gate game where both sides think for a fixed `move_time` per move.
/// Movetime (not depth) bounds each move, so the whole game's cost is bounded by
/// `max_plies * move_time` regardless of how sharp the position is — which is
/// what makes the gate's total runtime predictable.
fn play_mobility_game(
    fen: &str,
    candidate_color: Color,
    candidate_params: SearchParams,
    candidate_eval: EvalParams,
    baseline_params: SearchParams,
    baseline_eval: EvalParams,
    move_time: Duration,
    max_plies: u32,
) -> Result<GameRecord, String> {
    let mut board = Board::from_fen(fen)?;
    let mut candidate = gate_searcher(candidate_params, candidate_eval);
    let mut baseline = gate_searcher(baseline_params, baseline_eval);
    for ply in 0..max_plies {
        let searcher = if board.side_to_move == candidate_color {
            &mut candidate
        } else {
            &mut baseline
        };
        let result = searcher.search(
            &board,
            SearchLimits {
                movetime: Some(move_time),
                ..SearchLimits::default()
            },
        );
        let Some(mv) = result.best_move else {
            return Ok(GameRecord {
                fen: fen.to_string(),
                candidate_color,
                outcome: outcome_from_status(board.game_status(), candidate_color),
                plies: ply,
            });
        };
        board.make_move(mv)?;
    }
    Ok(GameRecord {
        fen: fen.to_string(),
        candidate_color,
        outcome: GameOutcome::Draw,
        plies: max_plies,
    })
}

/// Plays mobility-on (`mobility_scale = 100`) against mobility-off (`= 0`) over
/// `openings` generated openings, color-swapped. Both sides think for the same
/// `move_time` per move (fair) and games are capped at `max_plies`, so the total
/// cost is bounded by `openings * 2 * max_plies * move_time`. Everything but the
/// mobility scale is identical, so the SPRT isolates the term.
pub fn run_mobility_gate(
    openings: usize,
    seed: u64,
    move_time: Duration,
    max_plies: u32,
) -> Result<Vec<GameRecord>, String> {
    let fens = random_opening_fens(openings, 8, seed);
    run_mobility_gate_fens(&fens, move_time, max_plies)
}

/// The mobility gate over an explicit set of opening FENs — the shardable core
/// so a Modal fan-out can play a slice per container. Plays each FEN
/// color-swapped, mobility-on (`mobility_scale = 100`) against mobility-off, at
/// the same `move_time` per move, capped at `max_plies`.
pub fn run_mobility_gate_fens<S: AsRef<str>>(
    fens: &[S],
    move_time: Duration,
    max_plies: u32,
) -> Result<Vec<GameRecord>, String> {
    let candidate = SearchParams { mobility_scale: 100, ..SearchParams::default() };
    // Pin the baseline OFF explicitly: the shipped default is now mobility-on, so
    // relying on `SearchParams::default()` here would make this 100-vs-100.
    let baseline = SearchParams { mobility_scale: 0, ..SearchParams::default() };
    let mut records = Vec::with_capacity(fens.len() * 2);
    for fen in fens {
        for candidate_color in [Color::White, Color::Black] {
            records.push(play_mobility_game(
                fen.as_ref(),
                candidate_color,
                candidate,
                EvalParams::default(),
                baseline,
                EvalParams::default(),
                move_time,
                max_plies,
            )?);
        }
    }
    Ok(records)
}

/// The eval gate over an explicit set of opening FENs — the shardable core so a
/// Modal fan-out can play a slice per container. Plays the `candidate`
/// `EvalParams` (mobility on, `mobility_scale = 100`) against the `baseline`
/// `EvalParams` (mobility off) over each FEN color-swapped, at the same
/// `move_time` per move, capped at `max_plies`. Because the candidate also turns
/// mobility on while the baseline leaves it off, the SPRT measures the *combined*
/// tuned-eval-plus-mobility-on change versus the current default — not a pure
/// eval-only A/B (that would require both sides to share one mobility setting).
pub fn run_eval_gate_fens<S: AsRef<str>>(
    fens: &[S],
    candidate: EvalParams,
    baseline: EvalParams,
    move_time: Duration,
    max_plies: u32,
) -> Result<Vec<GameRecord>, String> {
    let candidate_params = SearchParams { mobility_scale: 100, ..SearchParams::default() };
    // Pin the baseline OFF explicitly: the shipped default is now mobility-on, so
    // relying on `SearchParams::default()` here would leave mobility on for both.
    let baseline_params = SearchParams { mobility_scale: 0, ..SearchParams::default() };
    let mut records = Vec::with_capacity(fens.len() * 2);
    for fen in fens {
        for candidate_color in [Color::White, Color::Black] {
            records.push(play_mobility_game(
                fen.as_ref(),
                candidate_color,
                candidate_params,
                candidate,
                baseline_params,
                baseline,
                move_time,
                max_plies,
            )?);
        }
    }
    Ok(records)
}

/// Plays `candidate` search params vs `baseline` over `fens` (both colours) at a
/// fixed depth — the out-of-sample validation for a SPSA-tuned SearchParams (the
/// campaign optimises in-sample over a small opening set; this re-checks on fresh
/// openings before the params are shipped). Honours `RUSTY_FISH_SPSA_NNUE` via
/// [`play_parameter_game`], so it can validate against the shipped NNUE eval.
pub fn run_search_gate_fens<S: AsRef<str>>(
    fens: &[S],
    candidate: SearchParams,
    baseline: SearchParams,
    depth: u8,
    max_plies: u32,
) -> Result<Vec<GameRecord>, String> {
    let config = MatchConfig { candidate_depth: depth, baseline_depth: depth, max_plies };
    let mut records = Vec::with_capacity(fens.len() * 2);
    for fen in fens {
        for candidate_color in [Color::White, Color::Black] {
            records.push(play_parameter_game(
                fen.as_ref(),
                candidate_color,
                candidate,
                EvalParams::default(),
                baseline,
                EvalParams::default(),
                config,
            )?);
        }
    }
    Ok(records)
}

fn play_external_game(
    fen: &str,
    candidate_color: Color,
    config: &ExternalMatchConfig,
) -> Result<GameRecord, String> {
    let opponent_path = config
        .uci_path
        .as_deref()
        .ok_or_else(|| "external UCI path disappeared after validation".to_string())?;
    let mut board = Board::from_fen(fen)?;
    let mut candidate = Searcher::default();
    // Trim the candidate's move overhead: this is an automated harness with no
    // GUI latency to reserve for, so the default 25 ms would under-spend the
    // time budget and break parity with the opponent's full movetime.
    let mut candidate_options = candidate.options().clone();
    candidate_options.move_overhead = config.candidate_move_overhead;
    candidate.set_options(candidate_options);
    let mut opponent = UciProcess::start(opponent_path, config.response_timeout)?;
    if let Some(elo) = config.opponent_elo {
        opponent.set_uci_elo(elo)?;
    }
    for ply in 0..config.max_plies {
        let mv = if board.side_to_move == candidate_color {
            candidate
                .search(
                    &board,
                    SearchLimits {
                        movetime: Some(config.candidate_movetime),
                        ..SearchLimits::default()
                    },
                )
                .best_move
        } else {
            match opponent.best_move(&board, config.opponent_movetime)? {
                Some(reply) => Some(board.parse_uci_move(&reply)?),
                None => None,
            }
        };
        let Some(mv) = mv else {
            return Ok(GameRecord {
                fen: fen.to_string(),
                candidate_color,
                outcome: outcome_from_status(board.game_status(), candidate_color),
                plies: ply,
            });
        };
        board.make_move(mv)?;
    }
    Ok(GameRecord {
        fen: fen.to_string(),
        candidate_color,
        outcome: GameOutcome::Draw,
        plies: config.max_plies,
    })
}

struct UciProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: Receiver<Result<String, String>>,
    response_timeout: Duration,
}

impl UciProcess {
    fn start(path: &str, response_timeout: Duration) -> Result<Self, String> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| format!("failed to start external UCI engine {path}: {error}"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "external UCI engine has no stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "external UCI engine has no stdout".to_string())?;
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let line = line.map_err(|error| error.to_string());
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let mut process = Self {
            child,
            stdin,
            stdout: rx,
            response_timeout,
        };
        process.send("uci")?;
        process.wait_for("uciok")?;
        process.send("setoption name Threads value 1")?;
        process.send("setoption name Hash value 16")?;
        process.send("isready")?;
        process.wait_for("readyok")?;
        Ok(process)
    }

    /// Returns the engine's chosen move, or `None` when the engine reports that
    /// the side to move has no legal move (a terminal position).
    fn best_move(&mut self, board: &Board, movetime: Duration) -> Result<Option<String>, String> {
        self.send(&format!("position fen {}", board.to_fen()))?;
        self.send(&format!("go movetime {}", movetime.as_millis()))?;
        loop {
            let line = self.next_line()?;
            if let Some(best_move) = line.strip_prefix("bestmove ") {
                let best_move = best_move.split_whitespace().next().unwrap_or_default();
                return Ok(classify_bestmove_token(best_move));
            }
        }
    }

    /// Fixed-node Stockfish evaluation of `fen`, side-to-move-relative
    /// centipawns. Sends `ucinewgame` first to clear the transposition table so
    /// a fixed node budget is reproducible regardless of the scan order, then
    /// keeps the last `score` reported before `bestmove`.
    fn score_position(&mut self, fen: &str, nodes: u64) -> Result<i32, String> {
        self.send("ucinewgame")?;
        self.send("isready")?;
        self.wait_for("readyok")?;
        self.send(&format!("position fen {fen}"))?;
        self.send(&format!("go nodes {nodes}"))?;
        let mut last = None;
        loop {
            let line = self.next_line()?;
            if let Some(cp) = parse_uci_score_cp(&line) {
                last = Some(cp);
            }
            if line.starts_with("bestmove") {
                return last.ok_or_else(|| format!("no score for fen {fen}"));
            }
        }
    }

    fn send(&mut self, command: &str) -> Result<(), String> {
        writeln!(self.stdin, "{command}")
            .map_err(|error| format!("failed to send UCI command `{command}`: {error}"))?;
        self.stdin
            .flush()
            .map_err(|error| format!("failed to flush UCI command `{command}`: {error}"))
    }

    /// Weaken the engine to approximately `elo` via `UCI_LimitStrength` + `UCI_Elo`
    /// (both supported by Stockfish; the engine clamps to its own min/max Elo).
    /// Waits for `readyok` so the options are applied before the game starts.
    pub fn set_uci_elo(&mut self, elo: u32) -> Result<(), String> {
        self.send("setoption name UCI_LimitStrength value true")?;
        self.send(&format!("setoption name UCI_Elo value {elo}"))?;
        self.send("isready")?;
        self.wait_for("readyok")
    }

    fn wait_for(&mut self, expected: &str) -> Result<(), String> {
        loop {
            if self.next_line()? == expected {
                return Ok(());
            }
        }
    }

    fn next_line(&self) -> Result<String, String> {
        self.stdout
            .recv_timeout(self.response_timeout)
            .map_err(|error| format!("timed out waiting for external UCI response: {error}"))?
    }
}

impl Drop for UciProcess {
    fn drop(&mut self) {
        let _ = self.send("quit");
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Centipawn magnitude a `score mate N` line collapses to. Well past the
/// trainer's eval clamp / sigmoid saturation, so a forced mate reads as a
/// decisive-but-finite label rather than an infinity the trainer can't fit.
pub const MATE_CP: i32 = 30000;

/// Extracts the side-to-move centipawn score from a single UCI `info` line.
///
/// Returns `Some(N)` for `score cp N`, `Some(MATE_CP * sign(N))` for
/// `score mate N` (a strictly positive mate distance -> `+MATE_CP`; `mate 0`
/// means the side to move is already mated, so it maps to `-MATE_CP`), and `None` for
/// any line without a `score cp`/`score mate` pair (the caller keeps the last
/// score it saw). A trailing `lowerbound`/`upperbound` qualifier sits after the
/// value, so scanning the token right after `cp`/`mate` naturally ignores it.
fn parse_uci_score_cp(line: &str) -> Option<i32> {
    let mut tokens = line.split_whitespace();
    while let Some(token) = tokens.next() {
        if token != "score" {
            continue;
        }
        match tokens.next() {
            Some("cp") => {
                if let Some(value) = tokens.next().and_then(|value| value.parse::<i32>().ok()) {
                    return Some(value);
                }
            }
            Some("mate") => {
                if let Some(value) = tokens.next().and_then(|value| value.parse::<i32>().ok()) {
                    // `score mate 0` means the side to move is already mated (a
                    // loss), so only a strictly positive mate distance is a win.
                    return Some(if value > 0 { MATE_CP } else { -MATE_CP });
                }
            }
            _ => {}
        }
    }
    None
}

/// Classifies the token following `bestmove `. Returns `None` when the engine
/// reports no legal move (`0000` in the UCI spec, `(none)` as Stockfish spells
/// it, or an empty token) — meaning the position is terminal and the game ends.
fn classify_bestmove_token(token: &str) -> Option<String> {
    if token == "0000" || token == "(none)" || token.is_empty() {
        None
    } else {
        Some(token.to_string())
    }
}

fn outcome_from_status(status: GameStatus, candidate_color: Color) -> GameOutcome {
    match status {
        GameStatus::Checkmate(mated) if mated == candidate_color => GameOutcome::Loss,
        GameStatus::Checkmate(_) => GameOutcome::Win,
        GameStatus::Ongoing
        | GameStatus::Stalemate
        | GameStatus::DrawByFiftyMoveRule
        | GameStatus::DrawByRepetition => GameOutcome::Draw,
    }
}

/// Number of tunable dimensions. Must match [`SearchParams`]'s scalar fields and
/// the [`SPSA_SPECS`] table below.
pub const SPSA_DIMENSIONS: usize = 8;

/// Describes one tunable dimension: its name, inclusive bounds, and the SPSA
/// perturbation magnitude (in parameter units).
#[derive(Clone, Copy, Debug)]
pub struct SpsaSpec {
    pub name: &'static str,
    pub min: f64,
    pub max: f64,
    pub step: f64,
}

/// The tunable dimensions, in the same order as the vector produced by
/// [`search_params_to_vector`].
pub const SPSA_SPECS: [SpsaSpec; SPSA_DIMENSIONS] = [
    SpsaSpec { name: "aspiration_window", min: 10.0, max: 200.0, step: 8.0 },
    SpsaSpec { name: "razor_margin_base", min: 40.0, max: 400.0, step: 16.0 },
    SpsaSpec { name: "razor_margin_scale", min: 10.0, max: 200.0, step: 12.0 },
    SpsaSpec { name: "reverse_futility_base", min: 40.0, max: 400.0, step: 16.0 },
    SpsaSpec { name: "reverse_futility_scale", min: 10.0, max: 200.0, step: 12.0 },
    SpsaSpec { name: "late_move_pruning_base", min: 1.0, max: 10.0, step: 1.0 },
    SpsaSpec { name: "late_move_pruning_scale", min: 1.0, max: 6.0, step: 1.0 },
    SpsaSpec { name: "null_move_reduction", min: 2.0, max: 5.0, step: 1.0 },
];

/// Projects a [`SearchParams`] onto the tunable vector.
pub fn search_params_to_vector(params: &SearchParams) -> [f64; SPSA_DIMENSIONS] {
    [
        params.aspiration_window as f64,
        params.razor_margin_base as f64,
        params.razor_margin_scale as f64,
        params.reverse_futility_base as f64,
        params.reverse_futility_scale as f64,
        params.late_move_pruning_base as f64,
        params.late_move_pruning_scale as f64,
        f64::from(params.null_move_reduction),
    ]
}

/// Reconstructs a [`SearchParams`] from a tunable vector, clamping to bounds and
/// rounding to each field's integer type.
pub fn vector_to_search_params(vector: &[f64; SPSA_DIMENSIONS]) -> SearchParams {
    let clamped = clamp_vector(vector, &SPSA_SPECS);
    SearchParams {
        aspiration_window: clamped[0].round() as i32,
        razor_margin_base: clamped[1].round() as i32,
        razor_margin_scale: clamped[2].round() as i32,
        reverse_futility_base: clamped[3].round() as i32,
        reverse_futility_scale: clamped[4].round() as i32,
        late_move_pruning_base: clamped[5].round() as usize,
        late_move_pruning_scale: clamped[6].round() as usize,
        null_move_reduction: clamped[7].round() as u8,
        mobility_scale: 0,
        // LMR correction thresholds are excluded from the SPSA vector (tuned by
        // direct gated A/B); keep their defaults.
        ..SearchParams::default()
    }
}

/// Narrows a generalized SPSA vector back to the fixed-width search-param array
/// so it can feed [`vector_to_search_params`].
fn to_array(v: &[f64]) -> [f64; SPSA_DIMENSIONS] {
    v.try_into().expect("len")
}

/// Number of tunable eval dimensions: 4 piece values x (mg, eg) = 8, 4 mobility
/// weights x (mg, eg) = 8, plus `bishop_pair` and `passed_pawn_base`. Kept next
/// to [`EVAL_SPSA_SPECS`] and the projection functions so the table and vector
/// lengths cannot drift.
pub const EVAL_DIMENSIONS: usize = 18;

/// The fixed, documented order of the eval SPSA vector. **This order is the
/// interchange contract** shared by [`eval_params_to_vector`],
/// [`vector_to_eval_params`], and the `eval-gate-file` / `spsa-eval` TSV: any
/// producer and consumer must agree on it. The 18 slots are, in order:
///
/// | Index | Weight |
/// |-------|--------|
/// | 0  | `knight.middlegame`          |
/// | 1  | `knight.endgame`             |
/// | 2  | `bishop.middlegame`          |
/// | 3  | `bishop.endgame`             |
/// | 4  | `rook.middlegame`            |
/// | 5  | `rook.endgame`               |
/// | 6  | `queen.middlegame`           |
/// | 7  | `queen.endgame`              |
/// | 8  | `knight_mobility.middlegame` |
/// | 9  | `knight_mobility.endgame`    |
/// | 10 | `bishop_mobility.middlegame` |
/// | 11 | `bishop_mobility.endgame`    |
/// | 12 | `rook_mobility.middlegame`   |
/// | 13 | `rook_mobility.endgame`      |
/// | 14 | `queen_mobility.middlegame`  |
/// | 15 | `queen_mobility.endgame`     |
/// | 16 | `bishop_pair`                |
/// | 17 | `passed_pawn_base`           |
pub const EVAL_SPSA_SPECS: [SpsaSpec; EVAL_DIMENSIONS] = [
    // Piece values: default +/- 120, step 12 (mg and eg share the same window).
    SpsaSpec { name: "knight_mg", min: 200.0, max: 440.0, step: 12.0 },
    SpsaSpec { name: "knight_eg", min: 200.0, max: 440.0, step: 12.0 },
    SpsaSpec { name: "bishop_mg", min: 210.0, max: 450.0, step: 12.0 },
    SpsaSpec { name: "bishop_eg", min: 210.0, max: 450.0, step: 12.0 },
    SpsaSpec { name: "rook_mg", min: 380.0, max: 620.0, step: 12.0 },
    SpsaSpec { name: "rook_eg", min: 380.0, max: 620.0, step: 12.0 },
    SpsaSpec { name: "queen_mg", min: 780.0, max: 1020.0, step: 12.0 },
    SpsaSpec { name: "queen_eg", min: 780.0, max: 1020.0, step: 12.0 },
    // Mobility weights: 0..12, step 1.
    SpsaSpec { name: "knight_mobility_mg", min: 0.0, max: 12.0, step: 1.0 },
    SpsaSpec { name: "knight_mobility_eg", min: 0.0, max: 12.0, step: 1.0 },
    SpsaSpec { name: "bishop_mobility_mg", min: 0.0, max: 12.0, step: 1.0 },
    SpsaSpec { name: "bishop_mobility_eg", min: 0.0, max: 12.0, step: 1.0 },
    SpsaSpec { name: "rook_mobility_mg", min: 0.0, max: 12.0, step: 1.0 },
    SpsaSpec { name: "rook_mobility_eg", min: 0.0, max: 12.0, step: 1.0 },
    SpsaSpec { name: "queen_mobility_mg", min: 0.0, max: 12.0, step: 1.0 },
    SpsaSpec { name: "queen_mobility_eg", min: 0.0, max: 12.0, step: 1.0 },
    SpsaSpec { name: "bishop_pair", min: 0.0, max: 80.0, step: 6.0 },
    SpsaSpec { name: "passed_pawn_base", min: 0.0, max: 60.0, step: 6.0 },
];

/// Projects an [`EvalParams`] onto the fixed 18-wide eval SPSA vector, in the
/// order documented on [`EVAL_SPSA_SPECS`].
pub fn eval_params_to_vector(params: &EvalParams) -> [f64; EVAL_DIMENSIONS] {
    [
        params.knight.middlegame as f64,
        params.knight.endgame as f64,
        params.bishop.middlegame as f64,
        params.bishop.endgame as f64,
        params.rook.middlegame as f64,
        params.rook.endgame as f64,
        params.queen.middlegame as f64,
        params.queen.endgame as f64,
        params.knight_mobility.middlegame as f64,
        params.knight_mobility.endgame as f64,
        params.bishop_mobility.middlegame as f64,
        params.bishop_mobility.endgame as f64,
        params.rook_mobility.middlegame as f64,
        params.rook_mobility.endgame as f64,
        params.queen_mobility.middlegame as f64,
        params.queen_mobility.endgame as f64,
        params.bishop_pair as f64,
        params.passed_pawn_base as f64,
    ]
}

/// Reconstructs an [`EvalParams`] from the 18-wide eval SPSA vector, clamping
/// each slot to its [`EVAL_SPSA_SPECS`] bounds and rounding to `i32`.
pub fn vector_to_eval_params(vector: &[f64; EVAL_DIMENSIONS]) -> EvalParams {
    let clamped = clamp_vector(vector, &EVAL_SPSA_SPECS);
    let at = |index: usize| clamped[index].round() as i32;
    EvalParams {
        knight: TaperedScore::new(at(0), at(1)),
        bishop: TaperedScore::new(at(2), at(3)),
        rook: TaperedScore::new(at(4), at(5)),
        queen: TaperedScore::new(at(6), at(7)),
        knight_mobility: TaperedScore::new(at(8), at(9)),
        bishop_mobility: TaperedScore::new(at(10), at(11)),
        rook_mobility: TaperedScore::new(at(12), at(13)),
        queen_mobility: TaperedScore::new(at(14), at(15)),
        bishop_pair: at(16),
        passed_pawn_base: at(17),
    }
}

/// Narrows a generalized SPSA vector back to the fixed-width eval array so it
/// can feed [`vector_to_eval_params`].
pub fn to_eval_array(v: &[f64]) -> [f64; EVAL_DIMENSIONS] {
    v.try_into().expect("len")
}

/// Serializes an [`EvalParams`] to the one-line TSV the `eval-gate-file` /
/// `spsa-eval` commands interchange: the 18 vector values, tab-separated, in the
/// [`EVAL_SPSA_SPECS`] order.
/// Parses `SPSA_DIMENSIONS` whitespace-separated numbers (the tuned search-param
/// vector — e.g. the last row of `spsa`'s report) back into a [`SearchParams`].
/// `mobility_scale` is not part of the SPSA vector; it comes back as the
/// [`vector_to_search_params`] default (irrelevant under NNUE, which ignores it).
pub fn search_params_from_tsv(contents: &str) -> Result<SearchParams, String> {
    let values: Vec<f64> = contents
        .split_whitespace()
        .map(|token| {
            token
                .parse::<f64>()
                .map_err(|error| format!("invalid search-param TSV value `{token}`: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != SPSA_DIMENSIONS {
        return Err(format!(
            "search-param TSV must have {SPSA_DIMENSIONS} values, found {}",
            values.len()
        ));
    }
    Ok(vector_to_search_params(&to_array(&values)))
}

pub fn eval_params_to_tsv(params: &EvalParams) -> String {
    eval_params_to_vector(params)
        .iter()
        .map(|value| format!("{value:.0}"))
        .collect::<Vec<_>>()
        .join("\t")
}

/// Parses the [`eval_params_to_tsv`] one-line TSV back into an [`EvalParams`].
/// Accepts tab- or whitespace-separated values; requires exactly
/// [`EVAL_DIMENSIONS`] of them.
pub fn eval_params_from_tsv(contents: &str) -> Result<EvalParams, String> {
    let values: Vec<f64> = contents
        .split_whitespace()
        .map(|token| {
            token
                .parse::<f64>()
                .map_err(|error| format!("invalid eval TSV value `{token}`: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() != EVAL_DIMENSIONS {
        return Err(format!(
            "eval TSV must have {EVAL_DIMENSIONS} values, found {}",
            values.len()
        ));
    }
    Ok(vector_to_eval_params(&to_eval_array(&values)))
}

fn clamp_vector(vector: &[f64], specs: &[SpsaSpec]) -> Vec<f64> {
    vector
        .iter()
        .zip(specs)
        .map(|(v, s)| v.clamp(s.min, s.max))
        .collect()
}

fn perturb(theta: &[f64], direction: &[f64], sign: f64, specs: &[SpsaSpec]) -> Vec<f64> {
    let stepped: Vec<f64> = theta
        .iter()
        .zip(direction)
        .zip(specs)
        .map(|((t, d), s)| t + sign * d * s.step)
        .collect();
    clamp_vector(&stepped, specs)
}

/// A small deterministic xorshift64* PRNG. Seeding it identically reproduces the
/// same perturbation directions, so campaigns are reproducible and testable.
#[derive(Clone, Debug)]
pub struct SpsaRng {
    state: u64,
}

impl SpsaRng {
    pub fn new(seed: u64) -> Self {
        // Force a non-zero state; xorshift is undefined at zero.
        Self {
            state: (seed ^ 0x9E37_79B9_7F4A_7C15) | 1,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Draws a Rademacher (+/-1) perturbation direction for every dimension.
    pub fn direction(&mut self, dimensions: usize) -> Vec<f64> {
        (0..dimensions)
            .map(|_| if self.next_u64() & 1 == 0 { -1.0 } else { 1.0 })
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SpsaConfig {
    pub iterations: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub match_config: MatchConfig,
}

impl Default for SpsaConfig {
    fn default() -> Self {
        Self {
            iterations: 32,
            learning_rate: 1.0,
            seed: 0x0DDB_1A5E_5BAD_5EED,
            // Both sides search at the same depth so the comparison is fair.
            match_config: MatchConfig {
                candidate_depth: 4,
                baseline_depth: 4,
                max_plies: 120,
            },
        }
    }
}

/// One SPSA gradient step. Moves each parameter along the perturbation direction
/// in proportion to how much the perturbed candidate (`theta + step`) outscored
/// its mirror (`theta - step`); `candidate_score` is that match's score fraction
/// in `[0, 1]`. The result is clamped to the spec bounds.
pub fn spsa_update(
    theta: &[f64],
    direction: &[f64],
    candidate_score: f64,
    learning_rate: f64,
    specs: &[SpsaSpec],
) -> Vec<f64> {
    let gradient = 2.0 * candidate_score - 1.0;
    let next: Vec<f64> = theta
        .iter()
        .zip(direction)
        .zip(specs)
        .map(|((t, d), s)| t + learning_rate * gradient * d * s.step)
        .collect();
    clamp_vector(&next, specs)
}

#[derive(Clone, Debug)]
pub struct SpsaIterationRecord {
    pub iteration: usize,
    pub score: MatchScore,
    pub candidate_score_fraction: f64,
    pub params: SearchParams,
}

#[derive(Clone, Debug)]
pub struct SpsaReport {
    pub initial: SearchParams,
    pub tuned: SearchParams,
    pub iterations: Vec<SpsaIterationRecord>,
}

/// Runs a deterministic SPSA campaign over the supplied positions, starting from
/// `initial`. Each iteration perturbs the parameter vector, plays a `theta+` vs
/// `theta-` self-play match across all positions and both colours, and steps the
/// parameters. Returns the per-iteration record and the final tuned parameters.
pub fn run_spsa_campaign(
    positions: &[&str],
    initial: SearchParams,
    config: SpsaConfig,
) -> Result<SpsaReport, String> {
    let mut theta = search_params_to_vector(&initial).to_vec();
    let mut rng = SpsaRng::new(config.seed);
    let mut iterations = Vec::with_capacity(config.iterations);

    for iteration in 0..config.iterations {
        let direction = rng.direction(SPSA_DIMENSIONS);
        let plus_vector = perturb(&theta, &direction, 1.0, &SPSA_SPECS);
        let minus_vector = perturb(&theta, &direction, -1.0, &SPSA_SPECS);
        let plus = vector_to_search_params(&to_array(&plus_vector));
        let minus = vector_to_search_params(&to_array(&minus_vector));

        let mut score = MatchScore::default();
        for fen in positions {
            for candidate_color in [Color::White, Color::Black] {
                let record = play_parameter_game(
                    fen,
                    candidate_color,
                    plus,
                    EvalParams::default(),
                    minus,
                    EvalParams::default(),
                    config.match_config,
                )?;
                match record.outcome {
                    GameOutcome::Win => score.wins += 1,
                    GameOutcome::Draw => score.draws += 1,
                    GameOutcome::Loss => score.losses += 1,
                }
            }
        }

        let fraction = score.score_fraction().unwrap_or(0.5);
        theta = spsa_update(&theta, &direction, fraction, config.learning_rate, &SPSA_SPECS);
        iterations.push(SpsaIterationRecord {
            iteration,
            score,
            candidate_score_fraction: fraction,
            params: vector_to_search_params(&to_array(&theta)),
        });
    }

    Ok(SpsaReport {
        initial,
        tuned: vector_to_search_params(&to_array(&theta)),
        iterations,
    })
}

/// Configuration for an eval SPSA campaign ([`run_eval_spsa_campaign`]). Unlike
/// [`SpsaConfig`], the per-iteration match is bounded by a per-move `move_time`
/// (not a search depth) and a `max_plies` cap, so the whole campaign's cost is
/// `iterations * positions * 2 * max_plies * move_time` — mirroring the movetime
/// discipline of the eval gate ([`run_eval_gate_fens`]).
#[derive(Clone, Copy, Debug)]
pub struct EvalSpsaConfig {
    pub iterations: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub move_time: Duration,
    pub max_plies: u32,
}

impl Default for EvalSpsaConfig {
    fn default() -> Self {
        Self {
            iterations: 32,
            learning_rate: 1.0,
            seed: 0x0DDB_1A5E_5BAD_5EED,
            move_time: Duration::from_millis(50),
            max_plies: 120,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EvalSpsaIterationRecord {
    pub iteration: usize,
    pub score: MatchScore,
    pub candidate_score_fraction: f64,
    pub params: EvalParams,
}

#[derive(Clone, Debug)]
pub struct EvalSpsaReport {
    pub initial: EvalParams,
    pub tuned: EvalParams,
    pub iterations: Vec<EvalSpsaIterationRecord>,
}

/// Runs an eval SPSA campaign over the supplied positions, starting
/// from `initial`. The perturbation schedule is deterministic (fixed-seed
/// [`SpsaRng`] and deterministic openings), but the **match outcomes are not**:
/// each move is movetime-bounded, so per-move search depth — and thus the tuned
/// result — varies with CPU speed and load. Do not freeze the tuned output in a
/// test the way the search campaign's `search_param_spsa_matches_the_frozen_...`
/// test does; assert only that it stays in-bounds. Mirrors [`run_spsa_campaign`] but tunes the eval vector
/// (`EVAL_SPSA_SPECS`, [`EVAL_DIMENSIONS`] wide): each iteration perturbs the eval
/// vector, plays a `theta+` vs `theta-` self-play match across all positions and
/// both colours, and steps the eval params. **Both sides run mobility-on**
/// (`mobility_scale = 100`) so the tuned mobility weights are actually exercised;
/// the only difference between the two players is their [`EvalParams`]. Each move
/// is bounded by `config.move_time` (not depth), games capped at
/// `config.max_plies`. Returns the per-iteration record and the final tuned
/// `EvalParams`.
pub fn run_eval_spsa_campaign(
    positions: &[&str],
    initial: EvalParams,
    config: EvalSpsaConfig,
) -> Result<EvalSpsaReport, String> {
    let mobility_on = SearchParams { mobility_scale: 100, ..SearchParams::default() };
    let mut theta = eval_params_to_vector(&initial).to_vec();
    let mut rng = SpsaRng::new(config.seed);
    let mut iterations = Vec::with_capacity(config.iterations);

    for iteration in 0..config.iterations {
        let direction = rng.direction(EVAL_DIMENSIONS);
        let plus_vector = perturb(&theta, &direction, 1.0, &EVAL_SPSA_SPECS);
        let minus_vector = perturb(&theta, &direction, -1.0, &EVAL_SPSA_SPECS);
        let plus = vector_to_eval_params(&to_eval_array(&plus_vector));
        let minus = vector_to_eval_params(&to_eval_array(&minus_vector));

        // theta+ (candidate) vs theta- (baseline), both mobility-on, color-swapped
        // over every position — the eval analog of run_eval_gate_fens's match.
        let mut records = Vec::with_capacity(positions.len() * 2);
        for fen in positions {
            for candidate_color in [Color::White, Color::Black] {
                records.push(play_mobility_game(
                    fen,
                    candidate_color,
                    mobility_on,
                    plus,
                    mobility_on,
                    minus,
                    config.move_time,
                    config.max_plies,
                )?);
            }
        }
        let score = summarize(&records);
        let fraction = score.score_fraction().unwrap_or(0.5);
        theta = spsa_update(&theta, &direction, fraction, config.learning_rate, &EVAL_SPSA_SPECS);
        iterations.push(EvalSpsaIterationRecord {
            iteration,
            score,
            candidate_score_fraction: fraction,
            params: vector_to_eval_params(&to_eval_array(&theta)),
        });
    }

    Ok(EvalSpsaReport {
        initial,
        tuned: vector_to_eval_params(&to_eval_array(&theta)),
        iterations,
    })
}

pub fn spsa_tsv_report(report: &SpsaReport) -> String {
    let mut out = String::from("engine_version\titeration\twins\tdraws\tlosses\tcandidate_score");
    for spec in SPSA_SPECS.iter() {
        out.push('\t');
        out.push_str(spec.name);
    }
    out.push('\n');
    for record in &report.iterations {
        let vector = search_params_to_vector(&record.params);
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{:.6}",
            env!("CARGO_PKG_VERSION"),
            record.iteration,
            record.score.wins,
            record.score.draws,
            record.score.losses,
            record.candidate_score_fraction,
        ));
        for value in vector.iter() {
            out.push_str(&format!("\t{value:.0}"));
        }
        out.push('\n');
    }
    out
}

// --- WDL sample generation from Lichess games -------------------------------

/// The other colour. `engine_core::Color` has no `Not` impl, so mirror
/// `train.rs`'s free helper rather than writing `!stm`.
fn opposite(color: Color) -> Color {
    match color {
        Color::White => Color::Black,
        Color::Black => Color::White,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WdlSampleConfig {
    /// Skip the opening: only positions at or past this ply are eligible.
    pub min_ply: u32,
    /// Skip the last N plies (the decided endgame) from each game.
    pub end_trim: u32,
    /// Maximum sampled positions per game.
    pub per_game: usize,
    /// `(i, n)`: keep games where `stream_index % n == i`, counted over every
    /// game in stream order *before* the rating filter so shards partition
    /// disjointly regardless of which games the filter drops.
    pub shard: (usize, usize),
}

/// One labelled WDL position. `target` is a win-probability in `{0.0, 0.5, 1.0}`
/// (side-to-move-relative game outcome). It is deliberately not `Eq`/`Hash`
/// (f32) — the disjoint-partition test compares the printed TSV lines instead.
#[derive(Clone)]
pub struct WdlSample {
    pub target: f32,
    pub own: Vec<usize>,
    pub opp: Vec<usize>,
    pub ply: u32,
}

#[derive(Default)]
struct WdlTags {
    event: String,
    variant: String,
    white_elo: u32,
    black_elo: u32,
    result: String,
}

struct WdlGame {
    chess: Chess,
    board: Board,
    /// `(features_own, features_opp, side_to_move, ply)` for every eligible ply.
    positions: Vec<(Vec<usize>, Vec<usize>, Color, u32)>,
    ply: u32,
    valid: bool,
    result: String,
}

struct WdlBuilder {
    config: WdlSampleConfig,
    stream_index: usize,
    out: Vec<WdlSample>,
}

impl WdlBuilder {
    /// The same rating/standard filter book-tool applies (min rating 2200).
    fn accepts_tags(tags: &WdlTags) -> bool {
        tags.white_elo >= 2200
            && tags.black_elo >= 2200
            && tags.event.to_ascii_lowercase().contains("rated")
            && (tags.variant.is_empty() || tags.variant.eq_ignore_ascii_case("standard"))
    }
}

impl Visitor for WdlBuilder {
    type Tags = WdlTags;
    type Movetext = WdlGame;
    type Output = ();

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        ControlFlow::Continue(WdlTags::default())
    }

    fn tag(
        &mut self,
        tags: &mut WdlTags,
        name: &[u8],
        value: RawTag<'_>,
    ) -> ControlFlow<Self::Output> {
        let value = std::str::from_utf8(value.as_bytes()).unwrap_or_default();
        match name {
            b"Event" => tags.event = value.to_string(),
            b"Variant" => tags.variant = value.to_string(),
            b"WhiteElo" => tags.white_elo = value.parse().unwrap_or(0),
            b"BlackElo" => tags.black_elo = value.parse().unwrap_or(0),
            b"Result" => tags.result = value.to_string(),
            _ => {}
        }
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, tags: WdlTags) -> ControlFlow<Self::Output, Self::Movetext> {
        let (i, n) = self.config.shard;
        // Count every game in stream order and gate on the index *before* the
        // rating filter, so `0/n, 1/n, ...` partition the games disjointly.
        let in_shard = n == 0 || self.stream_index % n == i;
        self.stream_index += 1;
        let valid = in_shard && Self::accepts_tags(&tags);
        ControlFlow::Continue(WdlGame {
            chess: Chess::default(),
            board: Board::startpos(),
            positions: Vec::new(),
            ply: 0,
            valid,
            result: tags.result,
        })
    }

    fn san(&mut self, game: &mut WdlGame, san: SanPlus) -> ControlFlow<Self::Output> {
        if !game.valid {
            return ControlFlow::Continue(());
        }
        let Ok(mv) = san.san.to_move(&game.chess) else {
            game.valid = false;
            return ControlFlow::Continue(());
        };
        let stm = game.board.side_to_move;
        // Capture the pre-move features for the eligible plies; only commit them
        // once the move has applied cleanly to `board`.
        let eligible = game.ply >= self.config.min_ply;
        let features = eligible
            .then(|| (active_features(&game.board, stm), active_features(&game.board, opposite(stm))));
        let uci = UciMove::from_standard(mv.clone()).to_string();
        match game
            .board
            .parse_uci_move(&uci)
            .and_then(|parsed| game.board.make_move(parsed))
        {
            Ok(_) => {
                if let Some((own, opp)) = features {
                    game.positions.push((own, opp, stm, game.ply));
                }
                game.chess.play_unchecked(mv);
                game.ply += 1;
            }
            Err(_) => game.valid = false,
        }
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, game: WdlGame) -> Self::Output {
        if !game.valid || !matches!(game.result.as_str(), "1-0" | "0-1" | "1/2-1/2") {
            return;
        }
        // Trim the last `end_trim` plies (mechanical), then evenly subsample to
        // `per_game` positions (deterministic — no RNG).
        let last_ply = game.ply;
        let eligible: Vec<_> = game
            .positions
            .into_iter()
            .filter(|(_, _, _, ply)| *ply + self.config.end_trim < last_ply)
            .collect();
        let picked = evenly_spaced(&eligible, self.config.per_game);
        for (own, opp, stm, ply) in picked {
            let target = wdl_target_for(&game.result, stm);
            self.out.push(WdlSample {
                target,
                own,
                opp,
                ply,
            });
        }
    }
}

/// The side-to-move-relative outcome as a win-probability: the side to move won
/// -> 1.0, drew -> 0.5, lost -> 0.0.
fn wdl_target_for(result: &str, stm: Color) -> f32 {
    use engine_core::Color::*;
    match (result, stm) {
        ("1-0", White) | ("0-1", Black) => 1.0,
        ("1/2-1/2", _) => 0.5,
        _ => 0.0, // the side to move lost
    }
}

/// Deterministically subsamples `items` down to at most `n` evenly spaced
/// elements (returns them all when there are already `<= n`).
fn evenly_spaced<T: Clone>(items: &[T], n: usize) -> Vec<T> {
    if n == 0 {
        return Vec::new();
    }
    if items.len() <= n {
        return items.to_vec();
    }
    (0..n).map(|k| items[k * items.len() / n].clone()).collect()
}

/// Parses a PGN (whole games) and returns the labelled middlegame WDL samples.
/// `&[u8]` implements `io::Read`, so the `&str` bytes feed the reader directly.
pub fn gen_wdl_data_samples(pgn: &str, config: WdlSampleConfig) -> Vec<WdlSample> {
    gen_wdl_data_samples_from_reader(pgn.as_bytes(), config)
        .expect("reading an in-memory PGN cannot fail")
}

/// Streams a PGN from any `io::Read` (a file, or `zstdcat | -` on stdin) and
/// returns the labelled middlegame WDL samples. This is the entry the CLI uses
/// so the multi-GB Lichess export is never materialized as one `String`; the
/// returned `Vec` is bounded by `per_game * games`. Propagates any read error.
pub fn gen_wdl_data_samples_from_reader<R: std::io::Read>(
    reader: R,
    config: WdlSampleConfig,
) -> Result<Vec<WdlSample>, String> {
    let mut builder = WdlBuilder {
        config,
        stream_index: 0,
        out: Vec::new(),
    };
    Reader::new(reader)
        .visit_all_games(&mut builder)
        .map_err(|error| format!("failed to read PGN: {error}"))?;
    Ok(builder.out)
}

/// One sampled position for the Stockfish-eval teacher: the same middlegame
/// position `gen-wdl-data` would emit, but carrying the board `fen` (so a UCI
/// engine can re-evaluate it) instead of the game-outcome target. The
/// side-to-move-relative feature indices (`own`/`opp`) and `ply` correspond 1:1
/// to the matching `WdlSample` for the same config.
#[derive(Clone)]
pub struct EvalPositionSample {
    pub fen: String,
    pub own: Vec<usize>,
    pub opp: Vec<usize>,
    pub ply: u32,
}

struct EvalPositionGame {
    chess: Chess,
    board: Board,
    /// `(features_own, features_opp, fen, ply)` for every eligible ply.
    positions: Vec<(Vec<usize>, Vec<usize>, String, u32)>,
    ply: u32,
    valid: bool,
    result: String,
}

struct EvalPositionBuilder {
    config: WdlSampleConfig,
    stream_index: usize,
    out: Vec<EvalPositionSample>,
}

impl Visitor for EvalPositionBuilder {
    type Tags = WdlTags;
    type Movetext = EvalPositionGame;
    type Output = ();

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        ControlFlow::Continue(WdlTags::default())
    }

    fn tag(
        &mut self,
        tags: &mut WdlTags,
        name: &[u8],
        value: RawTag<'_>,
    ) -> ControlFlow<Self::Output> {
        let value = std::str::from_utf8(value.as_bytes()).unwrap_or_default();
        match name {
            b"Event" => tags.event = value.to_string(),
            b"Variant" => tags.variant = value.to_string(),
            b"WhiteElo" => tags.white_elo = value.parse().unwrap_or(0),
            b"BlackElo" => tags.black_elo = value.parse().unwrap_or(0),
            b"Result" => tags.result = value.to_string(),
            _ => {}
        }
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, tags: WdlTags) -> ControlFlow<Self::Output, Self::Movetext> {
        let (i, n) = self.config.shard;
        // Count every game in stream order and gate on the index *before* the
        // rating filter, so `0/n, 1/n, ...` partition the games disjointly —
        // exactly as `WdlBuilder` does, so the sampled game set matches.
        let in_shard = n == 0 || self.stream_index % n == i;
        self.stream_index += 1;
        let valid = in_shard && WdlBuilder::accepts_tags(&tags);
        ControlFlow::Continue(EvalPositionGame {
            chess: Chess::default(),
            board: Board::startpos(),
            positions: Vec::new(),
            ply: 0,
            valid,
            result: tags.result,
        })
    }

    fn san(&mut self, game: &mut EvalPositionGame, san: SanPlus) -> ControlFlow<Self::Output> {
        if !game.valid {
            return ControlFlow::Continue(());
        }
        let Ok(mv) = san.san.to_move(&game.chess) else {
            game.valid = false;
            return ControlFlow::Continue(());
        };
        let stm = game.board.side_to_move;
        // Capture the pre-move features *and the pre-move FEN* for the eligible
        // plies; only commit them once the move has applied cleanly to `board`.
        let eligible = game.ply >= self.config.min_ply;
        let captured = eligible.then(|| {
            (
                active_features(&game.board, stm),
                active_features(&game.board, opposite(stm)),
                game.board.to_fen(),
            )
        });
        let uci = UciMove::from_standard(mv.clone()).to_string();
        match game
            .board
            .parse_uci_move(&uci)
            .and_then(|parsed| game.board.make_move(parsed))
        {
            Ok(_) => {
                if let Some((own, opp, fen)) = captured {
                    game.positions.push((own, opp, fen, game.ply));
                }
                game.chess.play_unchecked(mv);
                game.ply += 1;
            }
            Err(_) => game.valid = false,
        }
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, game: EvalPositionGame) -> Self::Output {
        if !game.valid || !matches!(game.result.as_str(), "1-0" | "0-1" | "1/2-1/2") {
            return;
        }
        // Trim the last `end_trim` plies (mechanical), then evenly subsample to
        // `per_game` positions (deterministic — no RNG). Identical to
        // `WdlBuilder::end_game` so the sampled positions match 1:1.
        let last_ply = game.ply;
        let eligible: Vec<_> = game
            .positions
            .into_iter()
            .filter(|(_, _, _, ply)| *ply + self.config.end_trim < last_ply)
            .collect();
        let picked = evenly_spaced(&eligible, self.config.per_game);
        for (own, opp, fen, ply) in picked {
            self.out.push(EvalPositionSample {
                fen,
                own,
                opp,
                ply,
            });
        }
    }
}

/// Parses a PGN (whole games) and returns the FEN-labelled middlegame positions
/// — the same positions `gen_wdl_data_samples` samples, carrying the board FEN
/// instead of the game-outcome target so a UCI engine can re-evaluate them.
pub fn gen_eval_positions(pgn: &str, config: WdlSampleConfig) -> Vec<EvalPositionSample> {
    gen_eval_positions_from_reader(pgn.as_bytes(), config)
        .expect("reading an in-memory PGN cannot fail")
}

/// Streams a PGN from any `io::Read` (a file, or `zstdcat | -` on stdin) and
/// returns the FEN-labelled middlegame positions. Mirrors
/// `gen_wdl_data_samples_from_reader`; the returned `Vec` is bounded by
/// `per_game * games`. Propagates any read error.
pub fn gen_eval_positions_from_reader<R: std::io::Read>(
    reader: R,
    config: WdlSampleConfig,
) -> Result<Vec<EvalPositionSample>, String> {
    let mut builder = EvalPositionBuilder {
        config,
        stream_index: 0,
        out: Vec::new(),
    };
    Reader::new(reader)
        .visit_all_games(&mut builder)
        .map_err(|error| format!("failed to read PGN: {error}"))?;
    Ok(builder.out)
}

/// Drives one persistent Stockfish process to relabel `fen<TAB>own<TAB>opp`
/// lines with a fixed-node cp eval, printing `cp<TAB>own<TAB>opp` (own/opp
/// passed through verbatim) — the `target<TAB>own<TAB>opp` format the trainer
/// reads. Streams line by line so a multi-GB shard never buffers.
///
/// Robustness matters here: a shard is tens of thousands of positions, so one
/// malformed line or one Stockfish scoring error must not abort it. Bad lines
/// are counted and skipped, and the skip total is reported on stderr at the
/// end. Starts the engine with `start`, which spawns the process and completes
/// the `uci`/`isready` handshake.
pub fn run_label_sf<R: std::io::Read>(
    reader: R,
    nodes: u64,
    engine_path: &str,
) -> Result<(), String> {
    let mut engine = UciProcess::start(engine_path, Duration::from_secs(60))?;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let mut skipped: u64 = 0;
    for line in BufReader::new(reader).lines() {
        let line = line.map_err(|error| format!("failed to read positions: {error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let mut fields = line.split('\t');
        let (Some(fen), Some(own), Some(opp), None) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            skipped += 1;
            continue;
        };
        match engine.score_position(fen, nodes) {
            Ok(cp) => writeln!(out, "{cp}\t{own}\t{opp}")
                .map_err(|error| format!("failed to write labelled position: {error}"))?,
            Err(_) => {
                // A per-position hiccup (timeout, or an aborted search left in an
                // unknown state) must not poison or silently skip the rest of the
                // shard. Drop and respawn the engine so the next position always
                // starts from a known-clean state, rather than reusing a possibly
                // hung process and timing out every subsequent line. If the
                // respawn itself fails the engine is dead, so abort the shard
                // loudly instead of looping forever on a corpse.
                skipped += 1;
                engine = UciProcess::start(engine_path, Duration::from_secs(60))?;
            }
        }
    }
    out.flush()
        .map_err(|error| format!("failed to flush labelled positions: {error}"))?;
    if skipped > 0 {
        eprintln!("label-sf: skipped {skipped} positions (malformed or unscored)");
    }
    Ok(())
}

/// Comma-joins feature indices — the `own`/`opp` CSV the trainer reads.
fn join_indices(values: &[usize]) -> String {
    values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Parses a public dataset's eval field into centipawns *in the file's own sign
/// convention* (the caller applies any side-to-move flip). Accepts:
/// - plain integer centipawns (`"24"`, `"-135"`),
/// - mate scores (`"#3"`, `"#-2"`, `"mate 4"`) -> `+/- MATE_CP` by sign, matching
///   how `label-sf`'s UCI parser maps mates, and
/// - decimal pawn units (`"0.24"`, `"-1.35"`) -> centipawns.
fn parse_public_eval_cp(field: &str) -> Option<i32> {
    let field = field.trim();
    let mate = field
        .strip_prefix('#')
        .or_else(|| field.strip_prefix("mate "))
        .map(str::trim);
    if let Some(distance) = mate {
        let n: i32 = distance.parse().ok()?;
        return Some(if n >= 0 { MATE_CP } else { -MATE_CP });
    }
    if field.contains('.') {
        let pawns: f64 = field.parse().ok()?;
        return Some((pawns * 100.0).round() as i32);
    }
    field.parse::<i32>().ok()
}

/// Splits a `FEN<sep>eval` row: the eval is the final tab- or comma-delimited
/// field, and the FEN (which contains spaces but no tab/comma) is everything
/// before it. Tab wins over comma so a stray comma inside a header can't confuse
/// a tab-separated row.
fn split_fen_eval(line: &str) -> Option<(&str, &str)> {
    let (fen, eval) = line
        .rsplit_once('\t')
        .or_else(|| line.rsplit_once(','))?;
    Some((fen.trim(), eval.trim()))
}

/// Converts one public `FEN<sep>eval` row into a `(cp, own, opp)` training row —
/// the exact tuple `label-sf` produces, but with the centipawn label taken from
/// the file instead of a Stockfish pass. Returns `None` for a malformed FEN or
/// eval (header rows, blanks) so the streaming caller can skip and count them.
///
/// Feature extraction reuses [`active_features`], so the emitted own/opp indices
/// are byte-identical to what the network sees at inference — the invariant a
/// Python reimplementation of the 768-feature layout would silently risk.
///
/// `white_relative` handles sign convention: most public dumps report eval from
/// White's point of view, but the trainer's labels are side-to-move-relative
/// (`own` == side to move), so the sign is flipped for black-to-move positions.
/// Pass `white_relative = false` when the source is already side-to-move POV.
pub fn label_fen_row(line: &str, white_relative: bool) -> Option<(i32, Vec<usize>, Vec<usize>)> {
    let (fen, eval) = split_fen_eval(line.trim())?;
    let board = Board::from_fen(fen).ok()?;
    let raw = parse_public_eval_cp(eval)?;
    let stm = board.side_to_move;
    let cp = match (white_relative, stm) {
        (true, Color::Black) => -raw,
        _ => raw,
    };
    Some((cp, active_features(&board, stm), active_features(&board, opposite(stm))))
}

/// Streams public `FEN<sep>eval` rows (tab- or comma-separated) and prints
/// `cp<TAB>own<TAB>opp` — the same format `label-sf` emits — taking the label
/// from the file rather than evaluating with Stockfish. This lets a pre-scored
/// public dataset (Kaggle/HF FEN+eval dumps) feed the trainer directly, skipping
/// the whole `gen-eval-positions | label-sf` self-play + fixed-node fan-out.
/// Malformed rows are counted and skipped, reported on stderr at the end.
pub fn run_label_fens<R: std::io::Read>(reader: R, white_relative: bool) -> Result<(), String> {
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let mut skipped: u64 = 0;
    for line in BufReader::new(reader).lines() {
        let line = line.map_err(|error| format!("failed to read positions: {error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        match label_fen_row(&line, white_relative) {
            Some((cp, own, opp)) => writeln!(out, "{cp}\t{}\t{}", join_indices(&own), join_indices(&opp))
                .map_err(|error| format!("failed to write labelled position: {error}"))?,
            None => skipped += 1,
        }
    }
    out.flush()
        .map_err(|error| format!("failed to flush labelled positions: {error}"))?;
    if skipped > 0 {
        eprintln!("label-fens: skipped {skipped} lines (malformed FEN or eval)");
    }
    Ok(())
}

/// Per-position record cap for `gen-search-telemetry`. A depth-N search over a
/// midgame position emits at most a few hundred thousand move decisions; this
/// bounds a pathological position without truncating realistic ones. Records are
/// drained per position, so the cap applies per search, not per shard.
pub const SEARCH_TELEMETRY_CAP: usize = 4_000_000;

/// Reads FENs one per line and, for each, runs a fixed-depth search with
/// per-move-decision telemetry enabled, printing every collected `MoveDecision`
/// as a TSV row prefixed with a per-position `pos_id` (0-based, in input order).
/// The header row is printed once. Malformed FENs are counted and skipped, then
/// reported on stderr at the end, the same shard-friendly ergonomics as
/// [`run_label_sf`]. `pos_id` increments once per searched position (skipped
/// FENs produce no rows and do not consume an id); it groups the rows emitted by
/// one search.
pub fn run_gen_search_telemetry<R: std::io::Read>(reader: R, depth: u8) -> Result<(), String> {
    if depth == 0 {
        return Err("invalid depth 0: need depth >= 1".to_string());
    }
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    writeln!(out, "{TELEMETRY_TSV_HEADER}")
        .map_err(|error| format!("failed to write telemetry header: {error}"))?;
    let mut searcher = Searcher::default();
    searcher.enable_telemetry(SEARCH_TELEMETRY_CAP);
    // Observe CLASSICAL LMR (learned LMR is now the default): the telemetry is the
    // training substrate for the reduction corrector, so its `reduction`/`raised_alpha`
    // must reflect the baseline the model corrects, not the model's own decisions.
    searcher.set_lmr_model(None);
    let mut skipped: u64 = 0;
    let mut pos_id: u64 = 0;
    for line in BufReader::new(reader).lines() {
        let line = line.map_err(|error| format!("failed to read positions: {error}"))?;
        let fen = line.trim();
        if fen.is_empty() {
            continue;
        }
        let board = match Board::from_fen(fen) {
            Ok(board) => board,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };
        searcher.search(
            &board,
            SearchLimits {
                depth: Some(depth),
                ..SearchLimits::default()
            },
        );
        for record in searcher.take_telemetry() {
            writeln!(out, "{}", record.to_tsv_row(pos_id))
                .map_err(|error| format!("failed to write telemetry row: {error}"))?;
        }
        pos_id += 1;
    }
    out.flush()
        .map_err(|error| format!("failed to flush telemetry: {error}"))?;
    if skipped > 0 {
        eprintln!("gen-search-telemetry: skipped {skipped} malformed FENs");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_TACTICAL_SUITE, EVAL_DIMENSIONS, EVAL_SPSA_SPECS, EvalSpsaConfig,
        ExternalMatchConfig, MatchScore,
        SPSA_DIMENSIONS, SPSA_SPECS, SpsaConfig, SpsaRng, classify_bestmove_token,
        eval_params_from_tsv, eval_params_to_tsv, eval_params_to_vector, external_match_game_count,
        external_tsv_report, measure_throughput, run_eval_spsa_campaign,
        run_spsa_campaign, run_tactical_suite, search_params_to_vector, spsa_tsv_report,
        spsa_update, sprt, tactical_solve_rate, tactical_tsv_report, throughput_tsv_report,
        vector_to_eval_params, vector_to_search_params, MatchConfig, SprtConfig, SprtDecision,
        random_opening_fens, run_eval_gate_fens, run_mobility_gate, run_nnue_gauntlet,
        run_nnue_gauntlet_with_move_time, sprt_tsv_report, summarize, BaselineMode,
        baseline_searcher,
        gen_wdl_data_samples, gen_wdl_data_samples_from_reader, WdlSample, WdlSampleConfig,
        gen_eval_positions, gen_eval_positions_from_reader, EvalPositionSample,
        parse_uci_score_cp, MATE_CP,
    };
    use engine_search::{EvalParams, Nnue, SearchParams, TaperedScore};
    use std::{sync::Arc, time::Duration};

    #[test]
    fn classify_bestmove_token_treats_no_move_reports_as_game_over() {
        // A real move parses through.
        assert_eq!(classify_bestmove_token("e2e4"), Some("e2e4".to_string()));
        // Stockfish emits `(none)` in a terminal position; the UCI spec uses
        // `0000`. Both, and an empty token, mean the game is over, not a move to
        // parse. Regression: `(none)` previously crashed the campaign.
        assert_eq!(classify_bestmove_token("(none)"), None);
        assert_eq!(classify_bestmove_token("0000"), None);
        assert_eq!(classify_bestmove_token(""), None);
    }

    #[test]
    fn parse_uci_score_reads_last_cp_and_clamps_mate() {
        // cp from the last info line before bestmove
        assert_eq!(parse_uci_score_cp("info depth 20 score cp 37 nodes 1 pv e2e4"), Some(37));
        assert_eq!(parse_uci_score_cp("info depth 20 score cp -145 pv"), Some(-145));
        // mate -> clamped +/- MATE_CP
        assert_eq!(parse_uci_score_cp("info depth 30 score mate 3 pv"), Some(MATE_CP));
        assert_eq!(parse_uci_score_cp("info depth 30 score mate -2 pv"), Some(-MATE_CP));
        // mate 0 means the side to move is already mated -> a loss, not a win
        assert_eq!(parse_uci_score_cp("info depth 30 score mate 0 pv"), Some(-MATE_CP));
        // bound qualifier follows the value -> value kept, qualifier ignored
        assert_eq!(parse_uci_score_cp("info depth 20 score cp 37 lowerbound nodes 1"), Some(37));
        // non-score info line -> None (caller keeps the previous score)
        assert_eq!(parse_uci_score_cp("info depth 1 nodes 20"), None);
    }

    #[test]
    fn random_openings_are_legal_and_varied() {
        let fens = random_opening_fens(8, 6, 42);
        assert_eq!(fens.len(), 8);
        for fen in &fens {
            assert!(engine_core::Board::from_fen(fen).is_ok(), "opening FEN parses: {fen}");
        }
        // The walks diverge, so not every opening is identical.
        let unique: std::collections::HashSet<&String> = fens.iter().collect();
        assert!(unique.len() > 1, "openings should vary");
    }

    #[test]
    fn mobility_gate_plays_games_and_reports() {
        let records = run_mobility_gate(2, 0xC0FFEE, Duration::from_millis(5), 10).expect("gate runs");
        assert_eq!(records.len(), 4); // 2 openings x 2 colors
        let report = sprt_tsv_report(summarize(&records), SprtConfig::default());
        assert!(report.contains("decision"));
    }

    #[test]
    fn eval_vector_round_trips_default_params() {
        let params = EvalParams::default();
        let restored = vector_to_eval_params(&eval_params_to_vector(&params));
        assert_eq!(restored, params);
        // The vector width is pinned to the spec table so they cannot drift.
        assert_eq!(eval_params_to_vector(&params).len(), EVAL_DIMENSIONS);
        assert_eq!(EVAL_SPSA_SPECS.len(), EVAL_DIMENSIONS);
    }

    #[test]
    fn eval_vector_projection_uses_the_documented_order() {
        // Distinct values in every slot pin the fixed order (queen mg/eg is
        // slots 6/7, bishop_pair is 16, passed_pawn_base is 17).
        let params = EvalParams {
            knight: TaperedScore::new(300, 310),
            bishop: TaperedScore::new(320, 330),
            rook: TaperedScore::new(490, 500),
            queen: TaperedScore::new(880, 900),
            knight_mobility: TaperedScore::new(3, 4),
            bishop_mobility: TaperedScore::new(2, 3),
            rook_mobility: TaperedScore::new(1, 5),
            queen_mobility: TaperedScore::new(0, 2),
            bishop_pair: 40,
            passed_pawn_base: 24,
        };
        let vector = eval_params_to_vector(&params);
        let expected = [
            300.0, 310.0, 320.0, 330.0, 490.0, 500.0, 880.0, 900.0, 3.0, 4.0, 2.0, 3.0, 1.0, 5.0,
            0.0, 2.0, 40.0, 24.0,
        ];
        assert_eq!(vector, expected);
        assert_eq!(vector_to_eval_params(&vector), params);
    }

    #[test]
    fn vector_to_eval_params_clamps_out_of_range() {
        // Wildly out-of-range in both directions; every slot must land on its
        // spec bound.
        let mut low = [0.0_f64; EVAL_DIMENSIONS];
        let mut high = [0.0_f64; EVAL_DIMENSIONS];
        for index in 0..EVAL_DIMENSIONS {
            low[index] = EVAL_SPSA_SPECS[index].min - 1000.0;
            high[index] = EVAL_SPSA_SPECS[index].max + 1000.0;
        }
        let low_params = eval_params_to_vector(&vector_to_eval_params(&low));
        let high_params = eval_params_to_vector(&vector_to_eval_params(&high));
        for index in 0..EVAL_DIMENSIONS {
            assert_eq!(low_params[index], EVAL_SPSA_SPECS[index].min);
            assert_eq!(high_params[index], EVAL_SPSA_SPECS[index].max);
        }
    }

    #[test]
    fn each_eval_spec_governs_its_own_vector_slot() {
        // Perturb exactly ONE slot far past its own spec bound, starting from the
        // default (which sits inside every window). The perturbed slot must clamp
        // to THAT spec's min/max, and — critically — every OTHER slot must stay at
        // its default. That isolation proves spec index i governs vector slot i and
        // that the field at slot i is the one clamped: a specs/vector misordering
        // (e.g. queen bounds landing on a mobility slot, or `vector_to_eval_params`
        // reading slot i into the wrong field) would either bleed into a neighbour
        // slot or miss the bound here. The default round-trip test cannot catch
        // this because the defaults never touch a bound.
        let default_vector = eval_params_to_vector(&EvalParams::default());
        for slot in 0..EVAL_DIMENSIONS {
            let spec = EVAL_SPSA_SPECS[slot];
            for (perturbed, expected) in
                [(spec.max + 1000.0, spec.max), (spec.min - 1000.0, spec.min)]
            {
                let mut vector = default_vector;
                vector[slot] = perturbed;
                let result = eval_params_to_vector(&vector_to_eval_params(&vector));
                assert_eq!(
                    result[slot], expected,
                    "slot {slot} ({}) must clamp to its own spec bound",
                    spec.name
                );
                for other in 0..EVAL_DIMENSIONS {
                    if other != slot {
                        assert_eq!(
                            result[other], default_vector[other],
                            "perturbing slot {slot} ({}) must not disturb slot {other} ({})",
                            spec.name, EVAL_SPSA_SPECS[other].name
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn eval_tsv_round_trips_and_rejects_wrong_arity() {
        let params = EvalParams::default();
        let tsv = eval_params_to_tsv(&params);
        assert_eq!(tsv.split('\t').count(), EVAL_DIMENSIONS);
        assert_eq!(eval_params_from_tsv(&tsv).unwrap(), params);
        assert!(eval_params_from_tsv("1\t2\t3").is_err());
    }

    #[test]
    fn eval_gate_default_vs_default_is_well_formed() {
        let fens = random_opening_fens(2, 8, 0xE7A1);
        let records = run_eval_gate_fens(
            &fens,
            EvalParams::default(),
            EvalParams::default(),
            Duration::from_millis(5),
            10,
        )
        .expect("eval gate runs");
        assert_eq!(records.len(), 4); // 2 openings x 2 colors
        let report = sprt_tsv_report(summarize(&records), SprtConfig::default());
        assert!(report.contains("decision"));
    }

    #[test]
    fn eval_gate_rejects_a_materially_broken_candidate() {
        // A candidate that values its pieces far below a pawn throws them away
        // and collapses to a near-bare king — a lopsidedly bad eval. The gate
        // must not reward it. The decisive signal this self-play harness
        // produces is *wins*: `outcome_from_status` maps a search that returns
        // no move (`Ongoing`, no best move) to a draw, so a side that foresees
        // the forced mate against it "resigns into a draw" rather than being
        // recorded as a loss. Getting mated on the board is therefore rare, but
        // a materially broken candidate can never *beat* the sound default. So
        // we assert it takes zero wins and does not exceed an even score — a
        // sound tuned candidate, by contrast, would win games and exceed 0.5.
        let crippled = EvalParams {
            knight: TaperedScore::equal(20),
            bishop: TaperedScore::equal(20),
            rook: TaperedScore::equal(20),
            queen: TaperedScore::equal(20),
            ..EvalParams::default()
        };
        let fens = random_opening_fens(8, 8, 0x10517);
        let records = run_eval_gate_fens(
            &fens,
            crippled,
            EvalParams::default(),
            Duration::from_millis(30),
            160,
        )
        .expect("eval gate runs");
        assert_eq!(records.len(), 16); // 8 openings x 2 colors
        let score = summarize(&records);
        assert_eq!(
            score.wins, 0,
            "a materially broken candidate must not beat the default eval: {score:?}"
        );
        assert!(
            score.score_fraction().is_some_and(|fraction| fraction <= 0.5),
            "a materially broken candidate must not exceed an even score: {score:?}"
        );
    }

    #[test]
    fn nnue_gauntlet_plays_both_colours_for_each_position() {
        let net = Arc::new(Nnue::from_seed(1, 8));
        let config = MatchConfig {
            candidate_depth: 2,
            baseline_depth: 2,
            max_plies: 12,
        };
        let records = run_nnue_gauntlet(
            &["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"],
            net,
            config,
        )
        .expect("gauntlet runs");
        assert_eq!(records.len(), 2);
        assert_eq!(summarize(&records).games(), 2);
    }

    #[test]
    fn timed_nnue_gauntlet_stops_each_search_at_its_move_budget() {
        let records = run_nnue_gauntlet_with_move_time(
            &["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"],
            Arc::new(Nnue::from_seed(1, 8)),
            MatchConfig {
                candidate_depth: 4,
                baseline_depth: 4,
                max_plies: 160,
            },
            Duration::ZERO,
        )
        .expect("timed gauntlet runs");

        assert_eq!(records.len(), 2);
        assert!(records.iter().all(|record| record.plies == 0));
    }

    #[test]
    fn baseline_mode_selects_champion_net_or_handcrafted() {
        assert!(baseline_searcher(BaselineMode::Champion).has_nnue(), "champion baseline keeps the bundled net");
        assert!(!baseline_searcher(BaselineMode::Handcrafted).has_nnue(), "handcrafted baseline disables NNUE");
    }

    #[test]
    fn spsa_rng_is_reproducible_and_seed_sensitive() {
        let mut a = SpsaRng::new(42);
        let mut b = SpsaRng::new(42);
        let mut c = SpsaRng::new(43);
        let first = a.direction(SPSA_DIMENSIONS);
        assert_eq!(first, b.direction(SPSA_DIMENSIONS));
        // Every drawn component is a Rademacher +/-1.
        assert!(first.iter().all(|value| *value == 1.0 || *value == -1.0));
        // A different seed almost surely differs over the whole run.
        let differs = (0..8).any(|_| a.direction(SPSA_DIMENSIONS) != c.direction(SPSA_DIMENSIONS));
        assert!(differs);
    }

    #[test]
    fn spsa_vector_round_trips_default_params() {
        let params = SearchParams::default();
        let restored = vector_to_search_params(&search_params_to_vector(&params));
        // `mobility_scale` is intentionally excluded from the SPSA vector, so it
        // is not preserved by the round-trip: `vector_to_search_params` always
        // resets it to 0. Every other (tunable) field must round-trip exactly.
        assert_eq!(restored, SearchParams { mobility_scale: 0, ..params });
    }

    #[test]
    fn spsa_update_moves_toward_the_winning_side() {
        let theta = search_params_to_vector(&SearchParams::default());
        let direction = [1.0; SPSA_DIMENSIONS];
        let up = spsa_update(&theta, &direction, 1.0, 1.0, &SPSA_SPECS);
        let down = spsa_update(&theta, &direction, 0.0, 1.0, &SPSA_SPECS);
        for index in 0..SPSA_DIMENSIONS {
            // Defaults sit strictly inside the bounds, so a decisive result moves
            // every parameter in the direction of the stronger side.
            assert!(up[index] > theta[index], "dimension {index} should increase");
            assert!(down[index] < theta[index], "dimension {index} should decrease");
        }
        // A drawn result leaves the parameters unchanged.
        assert_eq!(spsa_update(&theta, &direction, 0.5, 1.0, &SPSA_SPECS), theta);
    }

    #[test]
    fn spsa_update_clamps_to_bounds() {
        let at_max: Vec<f64> = SPSA_SPECS.iter().map(|spec| spec.max).collect();
        let theta: [f64; SPSA_DIMENSIONS] = at_max.try_into().unwrap();
        let direction = [1.0; SPSA_DIMENSIONS];
        let stepped = spsa_update(&theta, &direction, 1.0, 1.0, &SPSA_SPECS);
        for index in 0..SPSA_DIMENSIONS {
            assert!(stepped[index] <= SPSA_SPECS[index].max);
            assert_eq!(stepped[index], SPSA_SPECS[index].max);
        }
    }

    #[test]
    fn spsa_smoke_campaign_returns_in_bounds_parameters() {
        let config = SpsaConfig {
            iterations: 2,
            learning_rate: 1.0,
            seed: 7,
            match_config: MatchConfig {
                candidate_depth: 2,
                baseline_depth: 2,
                max_plies: 16,
            },
        };
        let report = run_spsa_campaign(
            &["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"],
            SearchParams::default(),
            config,
        )
        .expect("smoke campaign runs");
        assert_eq!(report.iterations.len(), 2);
        let tuned = search_params_to_vector(&report.tuned);
        for index in 0..SPSA_DIMENSIONS {
            assert!(tuned[index] >= SPSA_SPECS[index].min);
            assert!(tuned[index] <= SPSA_SPECS[index].max);
        }
        let tsv = spsa_tsv_report(&report);
        assert!(tsv.starts_with("engine_version\titeration"));
        assert!(tsv.contains("aspiration_window"));
    }

    #[test]
    fn eval_spsa_smoke_campaign_returns_in_bounds_parameters() {
        // Two iterations over one position at a tiny movetime/ply budget: enough
        // to exercise perturb -> per-side-eval match -> spsa_update and prove the
        // tuned EvalParams land inside every EVAL_SPSA_SPECS window.
        let config = EvalSpsaConfig {
            iterations: 2,
            learning_rate: 1.0,
            seed: 7,
            move_time: Duration::from_millis(5),
            max_plies: 16,
        };
        let report = run_eval_spsa_campaign(
            &["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"],
            EvalParams::default(),
            config,
        )
        .expect("eval smoke campaign runs");
        assert_eq!(report.iterations.len(), 2);
        let tuned = eval_params_to_vector(&report.tuned);
        for index in 0..EVAL_DIMENSIONS {
            assert!(tuned[index] >= EVAL_SPSA_SPECS[index].min, "slot {index} below its min");
            assert!(tuned[index] <= EVAL_SPSA_SPECS[index].max, "slot {index} above its max");
        }
    }

    #[test]
    fn search_param_spsa_matches_the_frozen_tuned_params() {
        let cfg = SpsaConfig {
            iterations: 3,
            match_config: MatchConfig {
                candidate_depth: 2,
                baseline_depth: 2,
                max_plies: 16,
            },
            ..SpsaConfig::default()
        };
        let pos = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"];
        let report = run_spsa_campaign(&pos, SearchParams::default(), cfg).unwrap();
        // Pre-refactor snapshot captured from CI. The generalization of the SPSA
        // primitives (slices + specs argument) must reproduce these exactly.
        assert_eq!(
            report.tuned,
            SearchParams {
                aspiration_window: 50,
                razor_margin_base: 120,
                razor_margin_scale: 80,
                reverse_futility_base: 100,
                reverse_futility_scale: 90,
                late_move_pruning_base: 3,
                late_move_pruning_scale: 2,
                null_move_reduction: 3,
                mobility_scale: 0,
                ..SearchParams::default()
            }
        );
    }

    #[test]
    fn balanced_score_has_zero_elo_difference() {
        assert_eq!(MatchScore::default().elo_difference(), Some(0.0));
        assert_eq!(
            MatchScore {
                wins: 5,
                draws: 0,
                losses: 5,
            }
            .elo_difference(),
            Some(0.0)
        );
    }

    #[test]
    fn winning_score_has_positive_elo_difference() {
        assert!(
            MatchScore {
                wins: 7,
                draws: 2,
                losses: 1,
            }
            .elo_difference()
            .is_some_and(|elo| elo > 0.0)
        );
    }

    #[test]
    fn sprt_keeps_balanced_results_inconclusive_and_accepts_a_large_win_margin() {
        let config = SprtConfig::default();
        assert!(sprt(MatchScore::default(), config).is_none());
        assert_eq!(
            sprt(
                MatchScore {
                    wins: 10,
                    draws: 0,
                    losses: 10,
                },
                config,
            )
            .unwrap()
            .decision,
            SprtDecision::Continue
        );
        assert_eq!(
            sprt(
                MatchScore {
                    wins: 400,
                    draws: 0,
                    losses: 0,
                },
                config,
            )
            .unwrap()
            .decision,
            SprtDecision::AcceptH1
        );
    }

    #[test]
    fn report_records_the_match_depths() {
        let report = super::tsv_report(
            &[super::GameRecord {
                fen: "test".to_string(),
                candidate_color: engine_core::Color::White,
                outcome: super::GameOutcome::Draw,
                plies: 10,
            }],
            super::MatchConfig {
                candidate_depth: 5,
                baseline_depth: 3,
                max_plies: 80,
            },
        );
        assert!(report.contains("candidate_depth\tbaseline_depth"));
        assert!(report.contains("\t5\t3\t"));
    }

    #[test]
    fn external_match_requires_a_uci_path_and_schedules_color_pairs() {
        let missing_path = ExternalMatchConfig::default();
        assert!(missing_path.validate().is_err());

        let configured = ExternalMatchConfig {
            uci_path: Some("/tmp/stockfish".to_string()),
            ..ExternalMatchConfig::default()
        };
        assert!(configured.validate().is_ok());
        assert_eq!(external_match_game_count(16), 32);
    }

    #[test]
    fn external_report_records_the_pinned_opponent_settings() {
        let config = ExternalMatchConfig {
            uci_path: Some("/opt/stockfish".to_string()),
            ..ExternalMatchConfig::default()
        };
        let report = external_tsv_report(
            &[super::GameRecord {
                fen: "test".to_string(),
                candidate_color: engine_core::Color::Black,
                outcome: super::GameOutcome::Loss,
                plies: 42,
            }],
            &config,
        );
        // Time parity: the candidate is reported by its movetime, not a depth.
        assert!(report.contains("opponent_uci\tcandidate_movetime_ms\topponent_movetime_ms"));
        assert!(!report.contains("candidate_depth"));
        assert!(report.contains("/opt/stockfish"));
    }

    #[test]
    fn external_match_defaults_to_equal_time_for_both_engines() {
        let config = ExternalMatchConfig::default();
        assert_eq!(config.candidate_movetime, Duration::from_millis(100));
        assert_eq!(config.opponent_movetime, Duration::from_millis(100));
        assert_eq!(config.candidate_move_overhead, Duration::from_millis(10));
    }

    #[test]
    fn throughput_sample_reports_nodes_per_second() {
        let sample = measure_throughput(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            3,
        )
        .expect("start position is valid");
        assert_eq!(sample.depth, 3);
        assert!(sample.nodes > 0);
        assert!(sample.nodes_per_second > 0);
    }

    #[test]
    fn throughput_report_contains_measurement_fields() {
        let sample = measure_throughput(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            2,
        )
        .expect("start position is valid");
        let report = throughput_tsv_report(&[sample]);
        assert!(report.contains("nodes_per_second"));
        assert!(report.contains("\t2\t"));
    }

    #[test]
    fn tactical_suite_reports_a_versioned_solve_rate() {
        let results = run_tactical_suite(DEFAULT_TACTICAL_SUITE).unwrap();
        assert_eq!(results.len(), DEFAULT_TACTICAL_SUITE.len());
        assert_eq!(tactical_solve_rate(&results), Some(1.0));

        let report = tactical_tsv_report(&results);
        assert!(report.starts_with("engine_version\tcase\tdepth"));
        assert!(report.contains("mate_in_one"));
        assert!(report.contains("\t1.000000\n"));
    }

    const WDL_FIXTURE: &str = concat!(
        "[Event \"Rated Blitz\"]\n[WhiteElo \"2400\"]\n[BlackElo \"2400\"]\n[Result \"1-0\"]\n\n",
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0\n",
    );

    /// Four distinct rated games (a White win, a draw, a White win, a Black win),
    /// each 20 plies of standard theory, for the shard-partition test. Distinct
    /// openings guarantee the sampled middlegame positions never collide across
    /// games, so the shards are set-disjoint.
    const WDL_MULTI_GAME_FIXTURE: &str = concat!(
        // Game 0: Ruy Lopez, White win.
        "[Event \"Rated Blitz\"]\n[WhiteElo \"2400\"]\n[BlackElo \"2400\"]\n[Result \"1-0\"]\n\n",
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0\n\n",
        // Game 1: Italian, draw.
        "[Event \"Rated Blitz\"]\n[WhiteElo \"2300\"]\n[BlackElo \"2350\"]\n[Result \"1/2-1/2\"]\n\n",
        "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d3 d6 6. O-O O-O 7. Re1 a6 8. Bb3 Ba7 9. h3 h6 10. Nbd2 Re8 1/2-1/2\n\n",
        // Game 2: Queen's Gambit Declined, White win.
        "[Event \"Rated Rapid\"]\n[WhiteElo \"2500\"]\n[BlackElo \"2450\"]\n[Result \"1-0\"]\n\n",
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 h6 7. Bh4 b6 8. cxd5 exd5 9. Bd3 Bb7 10. O-O Nbd7 1-0\n\n",
        // Game 3: Sicilian, Black win.
        "[Event \"Rated Blitz\"]\n[WhiteElo \"2600\"]\n[BlackElo \"2620\"]\n[Result \"0-1\"]\n\n",
        "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be2 e5 7. Nb3 Be7 8. O-O O-O 9. Be3 Be6 10. Nd5 Nbd7 0-1\n\n",
    );

    fn wdl_line(sample: &WdlSample) -> String {
        let join = |indices: &[usize]| {
            indices
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        format!("{}\t{}\t{}", sample.target, join(&sample.own), join(&sample.opp))
    }

    #[test]
    fn gen_wdl_data_labels_positions_by_side_to_move_outcome() {
        // A 1-0 game: White-to-move sampled positions score 1.0, Black-to-move 0.0.
        let samples = gen_wdl_data_samples(
            WDL_FIXTURE,
            WdlSampleConfig {
                min_ply: 8,
                end_trim: 5,
                per_game: 6,
                shard: (0, 1),
            },
        );
        assert!(!samples.is_empty());
        for s in &samples {
            assert!(s.target == 1.0 || s.target == 0.0, "1-0 game targets are 1.0/0.0");
            // white-to-move (even ply index) -> won -> 1.0; black-to-move -> 0.0
            assert_eq!(s.target, if s.ply % 2 == 0 { 1.0 } else { 0.0 });
            assert!(!s.own.is_empty() && !s.opp.is_empty());
            assert!(s.own.iter().all(|&i| i < 768) && s.opp.iter().all(|&i| i < 768));
        }
    }

    #[test]
    fn gen_wdl_data_reader_matches_str_path() {
        // The CLI streams the PGN through `gen_wdl_data_samples_from_reader`
        // (over stdin / a file) to avoid buffering the multi-GB export as one
        // `String`. That reader path must yield exactly the same samples as the
        // in-memory `&str` path the tests use.
        let config = WdlSampleConfig {
            min_ply: 8,
            end_trim: 5,
            per_game: 6,
            shard: (0, 1),
        };
        let via_str: Vec<String> = gen_wdl_data_samples(WDL_MULTI_GAME_FIXTURE, config)
            .iter()
            .map(wdl_line)
            .collect();
        let via_reader: Vec<String> =
            gen_wdl_data_samples_from_reader(WDL_MULTI_GAME_FIXTURE.as_bytes(), config)
                .expect("reader path succeeds")
                .iter()
                .map(wdl_line)
                .collect();
        assert!(!via_str.is_empty());
        assert_eq!(via_str, via_reader);
    }

    #[test]
    fn gen_wdl_data_is_deterministic() {
        let config = WdlSampleConfig {
            min_ply: 8,
            end_trim: 5,
            per_game: 6,
            shard: (0, 1),
        };
        let first: Vec<String> = gen_wdl_data_samples(WDL_MULTI_GAME_FIXTURE, config)
            .iter()
            .map(wdl_line)
            .collect();
        let second: Vec<String> = gen_wdl_data_samples(WDL_MULTI_GAME_FIXTURE, config)
            .iter()
            .map(wdl_line)
            .collect();
        assert!(!first.is_empty());
        assert_eq!(first, second);
    }

    #[test]
    fn gen_wdl_data_skips_the_opening_and_the_endgame() {
        // The fixture applies exactly 20 plies, so with min_ply 8 / end_trim 5
        // every sample must land in [8, 20 - 5).
        let min_ply = 8;
        let end_trim = 5;
        let last_ply = 20;
        let samples = gen_wdl_data_samples(
            WDL_FIXTURE,
            WdlSampleConfig {
                min_ply,
                end_trim,
                per_game: 100,
                shard: (0, 1),
            },
        );
        assert!(!samples.is_empty());
        for s in &samples {
            assert!(s.ply >= min_ply, "no opening plies: {}", s.ply);
            assert!(s.ply + end_trim < last_ply, "no endgame plies: {}", s.ply);
        }
    }

    #[test]
    fn gen_wdl_data_caps_positions_per_game() {
        let per_game = 4;
        let samples = gen_wdl_data_samples(
            WDL_FIXTURE,
            WdlSampleConfig {
                min_ply: 8,
                end_trim: 5,
                per_game,
                shard: (0, 1),
            },
        );
        // The single-game fixture yields at most `per_game` samples.
        assert!(!samples.is_empty());
        assert!(samples.len() <= per_game);
    }

    #[test]
    fn gen_wdl_data_shards_partition_disjointly() {
        let config = |shard| WdlSampleConfig {
            min_ply: 8,
            end_trim: 5,
            per_game: 6,
            shard,
        };
        let lines = |shard| {
            gen_wdl_data_samples(WDL_MULTI_GAME_FIXTURE, config(shard))
                .iter()
                .map(wdl_line)
                .collect::<Vec<_>>()
        };
        let all = lines((0, 1));
        let shard0 = lines((0, 3));
        let shard1 = lines((1, 3));
        let shard2 = lines((2, 3));

        assert!(!all.is_empty());
        // Every shard is non-empty (the four distinct games spread across 0,1,2).
        assert!(!shard0.is_empty() && !shard1.is_empty() && !shard2.is_empty());

        use std::collections::HashSet;
        let set0: HashSet<&String> = shard0.iter().collect();
        let set1: HashSet<&String> = shard1.iter().collect();
        let set2: HashSet<&String> = shard2.iter().collect();
        // No TSV line appears in more than one shard.
        assert!(set0.is_disjoint(&set1));
        assert!(set0.is_disjoint(&set2));
        assert!(set1.is_disjoint(&set2));

        // The union of the three shards equals the full (0/1) set exactly.
        let union: HashSet<&String> = set0.union(&set1).cloned().collect::<HashSet<_>>()
            .union(&set2)
            .cloned()
            .collect();
        let all_set: HashSet<&String> = all.iter().collect();
        assert_eq!(union, all_set);
        // And the shards partition the samples with no duplication.
        assert_eq!(shard0.len() + shard1.len() + shard2.len(), all.len());
    }

    fn eval_line(sample: &EvalPositionSample) -> String {
        let join = |indices: &[usize]| {
            indices
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        format!("{}\t{}\t{}", sample.fen, join(&sample.own), join(&sample.opp))
    }

    #[test]
    fn gen_eval_positions_emits_fen_and_valid_features() {
        let samples = gen_eval_positions(
            WDL_FIXTURE,
            WdlSampleConfig {
                min_ply: 8,
                end_trim: 5,
                per_game: 6,
                shard: (0, 1),
            },
        );
        assert!(!samples.is_empty());
        for s in &samples {
            // FEN parses back to a legal position.
            assert!(engine_core::Board::from_fen(&s.fen).is_ok(), "fen parses: {}", s.fen);
            assert!(!s.own.is_empty() && !s.opp.is_empty());
            assert!(s.own.iter().all(|&i| i < 768) && s.opp.iter().all(|&i| i < 768));
        }
        // Same sampled positions as the WDL sampler for the same config (same games,
        // same plies) — only the payload differs.
        let wdl = gen_wdl_data_samples(
            WDL_FIXTURE,
            WdlSampleConfig {
                min_ply: 8,
                end_trim: 5,
                per_game: 6,
                shard: (0, 1),
            },
        );
        assert_eq!(samples.len(), wdl.len());
        assert!(samples
            .iter()
            .zip(&wdl)
            .all(|(e, w)| e.own == w.own && e.opp == w.opp && e.ply == w.ply));
    }

    #[test]
    fn gen_eval_positions_matches_wdl_sampling_across_configs() {
        // The eval sampler must stay 1:1 with the WDL sampler at *every* config,
        // not just the single (8/5/6) point the test above checks. Vary min_ply,
        // end_trim, per_game, and shard so any future divergence between the two
        // samplers (eligibility, trim, subsample, sharding) trips CI structurally.
        let configs = [
            WdlSampleConfig { min_ply: 4, end_trim: 2, per_game: 20, shard: (0, 1) },
            WdlSampleConfig { min_ply: 12, end_trim: 8, per_game: 3, shard: (0, 1) },
            WdlSampleConfig { min_ply: 8, end_trim: 5, per_game: 6, shard: (0, 2) },
            WdlSampleConfig { min_ply: 8, end_trim: 5, per_game: 6, shard: (1, 2) },
        ];
        for cfg in configs {
            let eval = gen_eval_positions(WDL_FIXTURE, cfg);
            let wdl = gen_wdl_data_samples(WDL_FIXTURE, cfg);
            assert_eq!(eval.len(), wdl.len(), "count mismatch at {cfg:?}");
            assert!(
                eval.iter().zip(&wdl).all(|(e, w)| e.own == w.own && e.opp == w.opp && e.ply == w.ply),
                "sampled positions diverged at {cfg:?}",
            );
        }
    }

    #[test]
    fn gen_eval_positions_reader_matches_str_path() {
        let config = WdlSampleConfig {
            min_ply: 8,
            end_trim: 5,
            per_game: 6,
            shard: (0, 1),
        };
        let via_str: Vec<String> = gen_eval_positions(WDL_MULTI_GAME_FIXTURE, config)
            .iter()
            .map(eval_line)
            .collect();
        let via_reader: Vec<String> =
            gen_eval_positions_from_reader(WDL_MULTI_GAME_FIXTURE.as_bytes(), config)
                .expect("reader path succeeds")
                .iter()
                .map(eval_line)
                .collect();
        assert!(!via_str.is_empty());
        assert_eq!(via_str, via_reader);
    }

    #[test]
    fn gen_eval_positions_is_deterministic() {
        let config = WdlSampleConfig {
            min_ply: 8,
            end_trim: 5,
            per_game: 6,
            shard: (0, 1),
        };
        let first: Vec<String> = gen_eval_positions(WDL_MULTI_GAME_FIXTURE, config)
            .iter()
            .map(eval_line)
            .collect();
        let second: Vec<String> = gen_eval_positions(WDL_MULTI_GAME_FIXTURE, config)
            .iter()
            .map(eval_line)
            .collect();
        assert!(!first.is_empty());
        assert_eq!(first, second);
    }

    #[test]
    fn gen_eval_positions_caps_positions_per_game() {
        let per_game = 4;
        let samples = gen_eval_positions(
            WDL_FIXTURE,
            WdlSampleConfig {
                min_ply: 8,
                end_trim: 5,
                per_game,
                shard: (0, 1),
            },
        );
        assert!(!samples.is_empty());
        assert!(samples.len() <= per_game);
    }

    #[test]
    fn gen_eval_positions_shards_partition_disjointly() {
        let config = |shard| WdlSampleConfig {
            min_ply: 8,
            end_trim: 5,
            per_game: 6,
            shard,
        };
        let lines = |shard| {
            gen_eval_positions(WDL_MULTI_GAME_FIXTURE, config(shard))
                .iter()
                .map(eval_line)
                .collect::<Vec<_>>()
        };
        let all = lines((0, 1));
        let shard0 = lines((0, 3));
        let shard1 = lines((1, 3));
        let shard2 = lines((2, 3));

        assert!(!all.is_empty());
        assert!(!shard0.is_empty() && !shard1.is_empty() && !shard2.is_empty());

        use std::collections::HashSet;
        let set0: HashSet<&String> = shard0.iter().collect();
        let set1: HashSet<&String> = shard1.iter().collect();
        let set2: HashSet<&String> = shard2.iter().collect();
        assert!(set0.is_disjoint(&set1));
        assert!(set0.is_disjoint(&set2));
        assert!(set1.is_disjoint(&set2));

        let union: HashSet<&String> = set0.union(&set1).cloned().collect::<HashSet<_>>()
            .union(&set2)
            .cloned()
            .collect();
        let all_set: HashSet<&String> = all.iter().collect();
        assert_eq!(union, all_set);
        assert_eq!(shard0.len() + shard1.len() + shard2.len(), all.len());
    }

    #[test]
    fn label_fen_row_features_match_active_features_and_pass_through_stm_eval() {
        // White to move: a white-relative eval passes through unchanged, and the
        // features must equal active_features() for the parsed board — the exact
        // indices label-sf/gen-eval-positions would emit for the same position.
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let board = engine_core::Board::from_fen(fen).unwrap();
        let (cp, own, opp) = super::label_fen_row(&format!("{fen},20"), true).unwrap();
        assert_eq!(cp, 20);
        assert_eq!(own, engine_search::active_features(&board, engine_core::Color::White));
        assert_eq!(opp, engine_search::active_features(&board, engine_core::Color::Black));
    }

    #[test]
    fn label_fen_row_flips_sign_for_black_to_move_when_white_relative() {
        // Black to move with a +50 White-relative eval means -50 for the side to
        // move; with --stm-relative (white_relative = false) it passes through.
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        let (cp_wr, _, _) = super::label_fen_row(&format!("{fen}\t50"), true).unwrap();
        assert_eq!(cp_wr, -50, "white-relative eval must flip for black to move");
        let (cp_stm, _, _) = super::label_fen_row(&format!("{fen}\t50"), false).unwrap();
        assert_eq!(cp_stm, 50, "stm-relative eval must pass through");
    }

    #[test]
    fn parse_public_eval_cp_handles_int_mate_and_pawns() {
        assert_eq!(super::parse_public_eval_cp("135"), Some(135));
        assert_eq!(super::parse_public_eval_cp("-42"), Some(-42));
        assert_eq!(super::parse_public_eval_cp("#3"), Some(MATE_CP));
        assert_eq!(super::parse_public_eval_cp("#-2"), Some(-MATE_CP));
        assert_eq!(super::parse_public_eval_cp("mate 5"), Some(MATE_CP));
        assert_eq!(super::parse_public_eval_cp("0.24"), Some(24));
        assert_eq!(super::parse_public_eval_cp("-1.35"), Some(-135));
        assert_eq!(super::parse_public_eval_cp("garbage"), None);
    }

    #[test]
    fn label_fen_row_skips_malformed() {
        assert!(super::label_fen_row("not a position line", true).is_none());
        assert!(super::label_fen_row("fen,eval", true).is_none()); // header row
    }
}
