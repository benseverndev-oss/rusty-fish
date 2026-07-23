//! Rigorous A/B compare harness for two engine configurations.
//!
//! The research direction (`docs/RESEARCH-DIRECTION.md`) mandates that every
//! experiment be judged under **equal nodes, equal wall time, multiple time
//! controls, SPRT, a tactical suite, and throughput — not equal-depth alone**.
//! Equal-depth is misleading the moment a change alters node counts (which every
//! learned-search change does), so this module adds the missing axes on top of
//! the existing match / SPRT / tactical / throughput machinery in `lib.rs`
//! rather than re-implementing any of it.
//!
//! The unit of comparison is an [`EngineConfig`] — a `SearchParams` + `EvalParams`
//! (+ optional NNUE net). Two configs play a match over a set of openings, both
//! colours, under a [`BudgetMode`] (equal nodes, equal movetime, or equal depth).
//! A future learned-search toggle is meant to slot in as just another
//! [`EngineConfig`] field, so the whole harness keeps working unchanged.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use engine_core::{Board, Color};
use engine_search::{
    EvalParams, Nnue, SearchLimits, SearchParams, SearchResult, Searcher,
};

use super::{
    GameOutcome, GameRecord, MatchScore, SprtConfig, SprtResult, TacticalCase, TacticalResult,
    ThroughputSample, measure_throughput, outcome_from_status, random_opening_fens,
    run_tactical_suite, sprt, summarize, tactical_solve_rate,
};

/// How much each side is allowed to spend per move. This is the axis the harness
/// exists for: the same match can be replayed under a fixed **node** budget
/// (isolates decision-quality-per-node — the fair way to compare a search change
/// that alters node counts), a fixed **movetime** budget (equal wall clock), or a
/// fixed **depth** (the legacy, misleading-when-node-counts-move axis, kept for
/// completeness).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetMode {
    /// `go nodes N` semantics: search until ~N nodes have been visited, then move.
    Nodes(u64),
    /// Equal wall time: think for this long per move.
    Movetime(Duration),
    /// Fixed search depth per move.
    Depth(u8),
}

impl BudgetMode {
    /// A short, stable label for reports and TSV rows, e.g. `nodes=50000`,
    /// `movetime=100ms`, `depth=6`.
    pub fn label(self) -> String {
        match self {
            BudgetMode::Nodes(n) => format!("nodes={n}"),
            BudgetMode::Movetime(t) => format!("movetime={}ms", t.as_millis()),
            BudgetMode::Depth(d) => format!("depth={d}"),
        }
    }
}

/// One engine configuration under test. Everything that distinguishes the two
/// players in a compare lives here, so a match is fully described by two
/// `EngineConfig`s plus a [`BudgetMode`]. A future learned-search experiment adds
/// its toggle as a field here and every entry point below keeps working.
#[derive(Clone)]
pub struct EngineConfig {
    pub name: String,
    pub search: SearchParams,
    pub eval: EvalParams,
    /// When `Some`, the searcher keeps this NNUE net; when `None` it plays the
    /// hand-crafted evaluation (mirrors the gate convention in `lib.rs`).
    pub nnue: Option<Arc<Nnue>>,
}

impl EngineConfig {
    /// A hand-crafted-eval config with the engine's default search/eval params.
    /// The natural baseline and the natural "identical opponent" for the
    /// self-consistency sanity check.
    pub fn handcrafted(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            search: SearchParams::default(),
            eval: EvalParams::default(),
            nnue: None,
        }
    }

    /// Same as [`handcrafted`](Self::handcrafted) but overriding the search
    /// params — the common "differing `SearchParams`" A/B case.
    pub fn with_search(name: impl Into<String>, search: SearchParams) -> Self {
        Self {
            name: name.into(),
            search,
            eval: EvalParams::default(),
            nnue: None,
        }
    }
}

/// Builds a searcher for `config`. The move overhead is trimmed to 3 ms because
/// this is an automated harness with no GUI latency to reserve for — the default
/// 25 ms would under-spend a small movetime budget and break parity (same
/// reasoning as `gate_searcher` in `lib.rs`).
fn config_searcher(config: &EngineConfig) -> Searcher {
    let mut searcher = Searcher::default();
    searcher.set_nnue(config.nnue.clone());
    searcher.set_search_params(config.search);
    searcher.set_eval_params(config.eval);
    let mut options = searcher.options().clone();
    options.move_overhead = Duration::from_millis(3);
    searcher.set_options(options);
    searcher
}

/// Searches `board` until ~`budget` nodes have been visited, then returns the
/// best move found so far — the harness-level implementation of `go nodes N`.
///
/// The search core is deliberately **not** touched. Instead this drives the
/// public iterative-deepening entry point with an infinite depth limit and a
/// stop signal: after each completed depth the callback observes the cumulative
/// node count, and once it reaches the budget it flips the shared atomic. The
/// search checks that atomic at the top of every node (`should_stop`), so the
/// next deepening iteration aborts immediately and the move from the last
/// completed depth is returned. Consequently the search stops on the first depth
/// whose cumulative nodes reach the budget: final node count is `>= budget` and
/// overshoots by at most the growth of that one depth — the same "check the limit
/// between work units, not mid-node" discipline a real `go nodes` uses.
pub fn search_to_node_budget(searcher: &mut Searcher, board: &Board, budget: u64) -> SearchResult {
    let stop = Arc::new(AtomicBool::new(false));
    let callback_stop = Arc::clone(&stop);
    searcher.search_with_callback_and_stop_signal(
        board,
        SearchLimits {
            infinite: true,
            ..SearchLimits::default()
        },
        Some(Arc::clone(&stop)),
        move |info| {
            if info.nodes >= budget {
                callback_stop.store(true, Ordering::Relaxed);
            }
        },
    )
}

/// Runs one per-move search under `mode`, returning the raw [`SearchResult`] so
/// the caller can both play the move and accumulate node/time totals for nps.
fn search_with_budget(searcher: &mut Searcher, board: &Board, mode: BudgetMode) -> SearchResult {
    match mode {
        BudgetMode::Depth(depth) => searcher.search(
            board,
            SearchLimits {
                depth: Some(depth),
                ..SearchLimits::default()
            },
        ),
        BudgetMode::Movetime(move_time) => searcher.search(
            board,
            SearchLimits {
                movetime: Some(move_time),
                ..SearchLimits::default()
            },
        ),
        BudgetMode::Nodes(budget) => search_to_node_budget(searcher, board, budget),
    }
}

/// Accumulated search work for one side of a match, for a nps estimate.
#[derive(Clone, Copy, Debug, Default)]
struct SideWork {
    nodes: u64,
    elapsed: Duration,
}

impl SideWork {
    fn record(&mut self, result: &SearchResult) {
        self.nodes = self.nodes.saturating_add(result.nodes);
        self.elapsed = self.elapsed.saturating_add(result.elapsed);
    }

    fn merge(&mut self, other: SideWork) {
        self.nodes = self.nodes.saturating_add(other.nodes);
        self.elapsed = self.elapsed.saturating_add(other.elapsed);
    }

    fn nps(self) -> u64 {
        let nanos = self.elapsed.as_nanos().max(1);
        (u128::from(self.nodes) * 1_000_000_000 / nanos) as u64
    }
}

/// Plays one game between `candidate` (playing `candidate_color`) and `baseline`,
/// both sides bounded by `mode`, capped at `max_plies`. Returns the game record
/// plus each side's accumulated search work for the nps estimate.
fn play_budget_game(
    fen: &str,
    candidate_color: Color,
    candidate: &EngineConfig,
    baseline: &EngineConfig,
    mode: BudgetMode,
    max_plies: u32,
) -> Result<(GameRecord, SideWork, SideWork), String> {
    let mut board = Board::from_fen(fen)?;
    let mut candidate_searcher = config_searcher(candidate);
    let mut baseline_searcher = config_searcher(baseline);
    let mut candidate_work = SideWork::default();
    let mut baseline_work = SideWork::default();
    for ply in 0..max_plies {
        let candidate_to_move = board.side_to_move == candidate_color;
        let result = if candidate_to_move {
            let result = search_with_budget(&mut candidate_searcher, &board, mode);
            candidate_work.record(&result);
            result
        } else {
            let result = search_with_budget(&mut baseline_searcher, &board, mode);
            baseline_work.record(&result);
            result
        };
        let Some(mv) = result.best_move else {
            return Ok((
                GameRecord {
                    fen: fen.to_string(),
                    candidate_color,
                    outcome: outcome_from_status(board.game_status(), candidate_color),
                    plies: ply,
                },
                candidate_work,
                baseline_work,
            ));
        };
        board.make_move(mv)?;
    }
    Ok((
        GameRecord {
            fen: fen.to_string(),
            candidate_color,
            outcome: GameOutcome::Draw,
            plies: max_plies,
        },
        candidate_work,
        baseline_work,
    ))
}

/// Knobs for a single [`run_bench_compare`] match.
#[derive(Clone, Copy, Debug)]
pub struct BenchCompareConfig {
    pub mode: BudgetMode,
    pub max_plies: u32,
    pub sprt: SprtConfig,
}

impl Default for BenchCompareConfig {
    fn default() -> Self {
        Self {
            mode: BudgetMode::Nodes(50_000),
            max_plies: 120,
            sprt: SprtConfig::default(),
        }
    }
}

/// The verdict of one compare match on one budget axis.
#[derive(Clone, Debug)]
pub struct BenchCompareReport {
    pub candidate_name: String,
    pub baseline_name: String,
    pub mode: BudgetMode,
    pub records: Vec<GameRecord>,
    pub score: MatchScore,
    pub sprt: Option<SprtResult>,
    pub elo: Option<f64>,
    pub candidate_nps: u64,
    pub baseline_nps: u64,
}

/// Plays `candidate` vs `baseline` over `fens`, both colours, bounded by
/// `config.mode`, and reports W/D/L, the SPRT verdict (reusing the existing
/// [`sprt`]), the estimated Elo, and each side's nps. This is the core A/B axis
/// every other entry point below composes.
pub fn run_bench_compare<S: AsRef<str>>(
    fens: &[S],
    candidate: &EngineConfig,
    baseline: &EngineConfig,
    config: BenchCompareConfig,
) -> Result<BenchCompareReport, String> {
    let mut records = Vec::with_capacity(fens.len() * 2);
    let mut candidate_work = SideWork::default();
    let mut baseline_work = SideWork::default();
    for fen in fens {
        for candidate_color in [Color::White, Color::Black] {
            let (record, candidate_side, baseline_side) = play_budget_game(
                fen.as_ref(),
                candidate_color,
                candidate,
                baseline,
                config.mode,
                config.max_plies,
            )?;
            records.push(record);
            candidate_work.merge(candidate_side);
            baseline_work.merge(baseline_side);
        }
    }
    let score = summarize(&records);
    Ok(BenchCompareReport {
        candidate_name: candidate.name.clone(),
        baseline_name: baseline.name.clone(),
        mode: config.mode,
        score,
        sprt: sprt(score, config.sprt),
        elo: score.elo_difference(),
        candidate_nps: candidate_work.nps(),
        baseline_nps: baseline_work.nps(),
        records,
    })
}

/// One TSV row (with header) for a compare report.
pub fn bench_compare_tsv_report(report: &BenchCompareReport) -> String {
    let mut out = String::from(
        "engine_version\tcandidate\tbaseline\tbudget\twins\tdraws\tlosses\telo\tllr\tdecision\tcandidate_nps\tbaseline_nps\n",
    );
    out.push_str(&bench_compare_tsv_row(report));
    out
}

/// The value row for a compare report, header-free, so a sweep can stack many
/// rows under one header.
fn bench_compare_tsv_row(report: &BenchCompareReport) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
        env!("CARGO_PKG_VERSION"),
        report.candidate_name,
        report.baseline_name,
        report.mode.label(),
        report.score.wins,
        report.score.draws,
        report.score.losses,
        report
            .elo
            .map_or_else(|| "".to_string(), |elo| format!("{elo:.2}")),
        report
            .sprt
            .map_or_else(|| "".to_string(), |result| format!("{:.4}", result.log_likelihood_ratio)),
        report
            .sprt
            .map_or_else(|| "".to_string(), |result| format!("{:?}", result.decision)),
        report.candidate_nps,
        report.baseline_nps,
    )
}

/// Runs [`run_bench_compare`] across several budgets so a change can be checked
/// for time-control dependence rather than trusted at a single setting. Returns
/// one report per mode, in the order given.
pub fn run_bench_sweep<S: AsRef<str>>(
    fens: &[S],
    candidate: &EngineConfig,
    baseline: &EngineConfig,
    modes: &[BudgetMode],
    max_plies: u32,
    sprt_config: SprtConfig,
) -> Result<Vec<BenchCompareReport>, String> {
    modes
        .iter()
        .map(|&mode| {
            run_bench_compare(
                fens,
                candidate,
                baseline,
                BenchCompareConfig {
                    mode,
                    max_plies,
                    sprt: sprt_config,
                },
            )
        })
        .collect()
}

/// A stacked TSV table, one row per budget in the sweep.
pub fn bench_sweep_tsv_report(reports: &[BenchCompareReport]) -> String {
    let mut out = String::from(
        "engine_version\tcandidate\tbaseline\tbudget\twins\tdraws\tlosses\telo\tllr\tdecision\tcandidate_nps\tbaseline_nps\n",
    );
    for report in reports {
        out.push_str(&bench_compare_tsv_row(report));
    }
    out
}

/// Knobs for the one-command [`run_bench_report`] full evaluation.
#[derive(Clone, Debug)]
pub struct BenchReportConfig {
    /// Node budget for the equal-nodes match.
    pub nodes: u64,
    /// Per-move budget for the equal-movetime match.
    pub movetime: Duration,
    /// The budgets swept for the time-control-dependence table.
    pub sweep_modes: Vec<BudgetMode>,
    pub max_plies: u32,
    pub sprt: SprtConfig,
    /// Tactical cases (reused from `lib.rs`; pass [`super::DEFAULT_TACTICAL_SUITE`]).
    pub tactical: &'static [TacticalCase],
    /// `(fen, depth)` positions for the throughput/nps measurement.
    pub throughput: Vec<(String, u8)>,
}

impl Default for BenchReportConfig {
    fn default() -> Self {
        Self {
            nodes: 50_000,
            movetime: Duration::from_millis(50),
            sweep_modes: vec![
                BudgetMode::Nodes(10_000),
                BudgetMode::Nodes(50_000),
                BudgetMode::Movetime(Duration::from_millis(20)),
                BudgetMode::Movetime(Duration::from_millis(50)),
            ],
            max_plies: 120,
            sprt: SprtConfig::default(),
            tactical: super::DEFAULT_TACTICAL_SUITE,
            throughput: vec![(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
                6,
            )],
        }
    }
}

/// The consolidated result of evaluating two configs under every axis the
/// research direction mandates.
#[derive(Clone, Debug)]
pub struct BenchFullReport {
    pub candidate_name: String,
    pub baseline_name: String,
    pub equal_nodes: BenchCompareReport,
    pub equal_movetime: BenchCompareReport,
    pub sweep: Vec<BenchCompareReport>,
    pub tactical: Vec<TacticalResult>,
    pub throughput: Vec<ThroughputSample>,
}

/// The "evaluate every experiment under all axes" entry point: given two configs
/// and openings, runs an equal-nodes match, an equal-movetime match, the
/// multi-time-control sweep, the tactical suite, and a throughput measurement —
/// reusing the existing tactical and throughput code — and returns everything so
/// the caller can render one consolidated report.
pub fn run_bench_report<S: AsRef<str>>(
    fens: &[S],
    candidate: &EngineConfig,
    baseline: &EngineConfig,
    config: BenchReportConfig,
) -> Result<BenchFullReport, String> {
    let equal_nodes = run_bench_compare(
        fens,
        candidate,
        baseline,
        BenchCompareConfig {
            mode: BudgetMode::Nodes(config.nodes),
            max_plies: config.max_plies,
            sprt: config.sprt,
        },
    )?;
    let equal_movetime = run_bench_compare(
        fens,
        candidate,
        baseline,
        BenchCompareConfig {
            mode: BudgetMode::Movetime(config.movetime),
            max_plies: config.max_plies,
            sprt: config.sprt,
        },
    )?;
    let sweep = run_bench_sweep(
        fens,
        candidate,
        baseline,
        &config.sweep_modes,
        config.max_plies,
        config.sprt,
    )?;
    let tactical = run_tactical_suite(config.tactical)?;
    let throughput = config
        .throughput
        .iter()
        .map(|(fen, depth)| measure_throughput(fen, *depth))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(BenchFullReport {
        candidate_name: candidate.name.clone(),
        baseline_name: baseline.name.clone(),
        equal_nodes,
        equal_movetime,
        sweep,
        tactical,
        throughput,
    })
}

/// Renders the full report as a single consolidated, human-readable block with a
/// section per axis and the SPRT verdict on each match axis.
pub fn bench_full_report_text(report: &BenchFullReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# bench-report: {} (candidate) vs {} (baseline)\n\n",
        report.candidate_name, report.baseline_name
    ));

    out.push_str("## equal-nodes match\n");
    out.push_str(&compare_summary_line(&report.equal_nodes));
    out.push_str("\n## equal-movetime match\n");
    out.push_str(&compare_summary_line(&report.equal_movetime));

    out.push_str("\n## multi-time-control sweep\n");
    out.push_str("budget\twins\tdraws\tlosses\telo\tdecision\tcandidate_nps\tbaseline_nps\n");
    for compare in &report.sweep {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            compare.mode.label(),
            compare.score.wins,
            compare.score.draws,
            compare.score.losses,
            compare
                .elo
                .map_or_else(|| "n/a".to_string(), |elo| format!("{elo:.2}")),
            compare
                .sprt
                .map_or_else(|| "n/a".to_string(), |result| format!("{:?}", result.decision)),
            compare.candidate_nps,
            compare.baseline_nps,
        ));
    }

    out.push_str("\n## tactical suite\n");
    let solve_rate = tactical_solve_rate(&report.tactical).unwrap_or(0.0);
    out.push_str(&format!("solve_rate\t{solve_rate:.4}\n"));
    for case in &report.tactical {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\n",
            case.name,
            case.expected_move,
            case.actual_move.as_deref().unwrap_or(""),
            case.solved,
        ));
    }

    out.push_str("\n## throughput\n");
    out.push_str("depth\tnodes\telapsed_ms\tnodes_per_second\n");
    for sample in &report.throughput {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\n",
            sample.depth,
            sample.nodes,
            sample.elapsed.as_millis(),
            sample.nodes_per_second,
        ));
    }
    out
}

/// A one-line human summary of a single compare match.
fn compare_summary_line(report: &BenchCompareReport) -> String {
    format!(
        "{}: {}W {}D {}L | elo {} | SPRT {} | nps {} vs {}\n",
        report.mode.label(),
        report.score.wins,
        report.score.draws,
        report.score.losses,
        report
            .elo
            .map_or_else(|| "n/a".to_string(), |elo| format!("{elo:.2}")),
        report
            .sprt
            .map_or_else(|| "n/a".to_string(), |result| format!("{:?}", result.decision)),
        report.candidate_nps,
        report.baseline_nps,
    )
}

/// Convenience: deterministic opening set for the CLI entry points, so the same
/// seed/count reproduces the same match. Walks `plies` random legal moves from
/// the start position for each of `count` openings (reuses `random_opening_fens`).
pub fn compare_openings(count: usize, seed: u64) -> Vec<String> {
    random_opening_fens(count, 8, seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::SprtDecision;
    use engine_search::SearchParams;

    const SANITY_FENS: &[&str] = &[
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3",
        "r1bq1rk1/pp1nbppp/2p1pn2/3p4/3P4/2NBPN2/PPQ1BPPP/R3K2R w KQ - 4 8",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ];

    #[test]
    fn node_budget_is_honored_within_tolerance() {
        // A node budget should stop at ~that node count: the search completes the
        // first depth whose cumulative nodes reach the budget, so the final count
        // is >= budget and overshoots by at most that one depth's growth.
        let board = Board::from_fen(SANITY_FENS[3]).expect("fen");
        for &budget in &[2_000_u64, 10_000, 40_000] {
            let mut searcher = config_searcher(&EngineConfig::handcrafted("t"));
            let result = search_to_node_budget(&mut searcher, &board, budget);
            assert!(result.best_move.is_some(), "budget search returns a move");
            assert!(
                result.nodes >= budget,
                "budget {budget}: consumed {} nodes, want >= budget",
                result.nodes
            );
            // Overshoot bounded generously: never more than 8x the budget (in
            // practice one extra depth). This is the "~that node count" tolerance.
            assert!(
                result.nodes <= budget.saturating_mul(8),
                "budget {budget}: consumed {} nodes, overshoot too large",
                result.nodes
            );
        }
    }

    #[test]
    fn larger_node_budget_searches_more() {
        // Monotonicity: a bigger budget visits at least as many nodes and reaches
        // at least as deep, confirming the budget actually governs search effort.
        let board = Board::from_fen(SANITY_FENS[3]).expect("fen");
        let mut small = config_searcher(&EngineConfig::handcrafted("s"));
        let mut large = config_searcher(&EngineConfig::handcrafted("l"));
        let small_result = search_to_node_budget(&mut small, &board, 2_000);
        let large_result = search_to_node_budget(&mut large, &board, 60_000);
        assert!(large_result.nodes > small_result.nodes);
        assert!(large_result.depth >= small_result.depth);
    }

    #[test]
    fn node_budget_search_is_deterministic() {
        // Same board + same budget + same config -> identical node count, depth,
        // and move. The equal-nodes axis must be reproducible to be trustworthy.
        let board = Board::from_fen(SANITY_FENS[3]).expect("fen");
        let mut first = config_searcher(&EngineConfig::handcrafted("a"));
        let mut second = config_searcher(&EngineConfig::handcrafted("b"));
        let a = search_to_node_budget(&mut first, &board, 20_000);
        let b = search_to_node_budget(&mut second, &board, 20_000);
        assert_eq!(a.nodes, b.nodes);
        assert_eq!(a.depth, b.depth);
        assert_eq!(a.best_move, b.best_move);
    }

    #[test]
    fn identical_configs_are_a_balanced_non_decisive_match() {
        // Self-consistency: config-A vs an identical config-A over openings, both
        // colours, must be a balanced result the SPRT does NOT falsely Accept H1.
        // Deterministic engines + color-swapped openings => each opening's two
        // games mirror, so wins == losses and the estimated Elo is ~0. This proves
        // the harness isn't biased toward a colour/side.
        let config = EngineConfig::handcrafted("self");
        let report = run_bench_compare(
            SANITY_FENS,
            &config,
            &config,
            BenchCompareConfig {
                mode: BudgetMode::Nodes(6_000),
                max_plies: 40,
                sprt: SprtConfig::default(),
            },
        )
        .expect("compare runs");
        assert_eq!(report.records.len(), SANITY_FENS.len() * 2);
        assert_eq!(
            report.score.wins, report.score.losses,
            "color-swapped identical configs must be symmetric: {:?}",
            report.score
        );
        assert_eq!(report.elo, Some(0.0), "identical configs => ~0 Elo");
        // The SPRT must not falsely accept the "candidate is better" hypothesis.
        if let Some(result) = report.sprt {
            assert_ne!(
                result.decision,
                SprtDecision::AcceptH1,
                "balanced self-play must not Accept H1"
            );
        }
    }

    #[test]
    fn compare_is_deterministic_across_runs() {
        // Same seed/openings/config => same match result (W/D/L identical).
        let candidate = EngineConfig::handcrafted("cand");
        let baseline = EngineConfig::with_search(
            "base",
            SearchParams {
                mobility_scale: 0,
                ..SearchParams::default()
            },
        );
        let fens = compare_openings(4, 0xB0BA_CAFE);
        let run = || {
            run_bench_compare(
                &fens,
                &candidate,
                &baseline,
                BenchCompareConfig {
                    mode: BudgetMode::Nodes(4_000),
                    max_plies: 30,
                    sprt: SprtConfig::default(),
                },
            )
            .expect("compare runs")
            .score
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn sweep_reports_one_row_per_budget() {
        // The multi-TC sweep yields one verdict per budget so a change can be
        // checked for TC-dependence.
        let config = EngineConfig::handcrafted("x");
        let modes = [
            BudgetMode::Nodes(3_000),
            BudgetMode::Depth(3),
            BudgetMode::Movetime(Duration::from_millis(5)),
        ];
        let reports = run_bench_sweep(SANITY_FENS, &config, &config, &modes, 24, SprtConfig::default())
            .expect("sweep runs");
        assert_eq!(reports.len(), modes.len());
        for (report, mode) in reports.iter().zip(modes) {
            assert_eq!(report.mode, mode);
            // Self-play stays symmetric on every axis.
            assert_eq!(report.score.wins, report.score.losses);
        }
        let table = bench_sweep_tsv_report(&reports);
        assert_eq!(table.lines().count(), modes.len() + 1); // header + one row per mode
    }

    #[test]
    fn full_report_runs_every_axis() {
        // The one-command report must exercise all axes: both matches, the sweep,
        // the tactical suite, and a throughput sample, and render a consolidated
        // block naming each section.
        let config = EngineConfig::handcrafted("full");
        let report = run_bench_report(
            SANITY_FENS,
            &config,
            &config,
            BenchReportConfig {
                nodes: 3_000,
                movetime: Duration::from_millis(5),
                sweep_modes: vec![BudgetMode::Nodes(3_000), BudgetMode::Depth(3)],
                max_plies: 24,
                sprt: SprtConfig::default(),
                tactical: super::super::DEFAULT_TACTICAL_SUITE,
                throughput: vec![(
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string(),
                    3,
                )],
            },
        )
        .expect("report runs");
        assert_eq!(report.sweep.len(), 2);
        assert!(!report.tactical.is_empty());
        assert!(!report.throughput.is_empty());
        // Self-play symmetry holds on both primary axes.
        assert_eq!(report.equal_nodes.score.wins, report.equal_nodes.score.losses);
        assert_eq!(
            report.equal_movetime.score.wins,
            report.equal_movetime.score.losses
        );
        let text = bench_full_report_text(&report);
        assert!(text.contains("equal-nodes match"));
        assert!(text.contains("equal-movetime match"));
        assert!(text.contains("multi-time-control sweep"));
        assert!(text.contains("tactical suite"));
        assert!(text.contains("throughput"));
    }
}
