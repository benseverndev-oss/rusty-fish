use std::time::Duration;

use engine_core::{Board, Color, GameStatus};
use engine_search::{SearchLimits, Searcher};

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

fn play_game(fen: &str, candidate_color: Color, config: MatchConfig) -> Result<GameRecord, String> {
    let mut board = Board::from_fen(fen)?;
    let mut candidate = Searcher::default();
    let mut baseline = Searcher::default();
    for ply in 0..config.max_plies {
        let depth = if board.side_to_move == candidate_color {
            config.candidate_depth
        } else {
            config.baseline_depth
        };
        let searcher = if board.side_to_move == candidate_color {
            &mut candidate
        } else {
            &mut baseline
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

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_TACTICAL_SUITE, MatchScore, measure_throughput, run_tactical_suite,
        sprt, tactical_solve_rate, tactical_tsv_report, throughput_tsv_report, SprtConfig,
        SprtDecision,
    };

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
}
