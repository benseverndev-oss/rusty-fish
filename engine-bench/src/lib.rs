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
    use super::{MatchScore, measure_throughput, throughput_tsv_report};

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
}
