#[cfg(test)]
mod tests {
    use super::{
        MATE_LABEL_CP, StockfishLabel, calibration_candidates, choose_budget, parse_info_score,
    };
    use std::collections::VecDeque;

    trait UciTransport {
        fn send(&mut self, command: &str) -> Result<(), String>;
        fn next_line(&mut self) -> Result<String, String>;
    }

    struct FakeUci {
        commands: Vec<String>,
        replies: VecDeque<Result<String, String>>,
    }

    impl UciTransport for FakeUci {
        fn send(&mut self, command: &str) -> Result<(), String> {
            self.commands.push(command.to_string());
            Ok(())
        }
        fn next_line(&mut self) -> Result<String, String> {
            self.replies
                .pop_front()
                .unwrap_or_else(|| Err("timeout".to_string()))
        }
    }

    fn evaluate_one_with_transport<T: UciTransport>(
        transport: &mut T,
        fen: &str,
        nodes: u64,
    ) -> Result<super::StockfishLabel, String> {
        transport.send("ucinewgame")?;
        transport.send("isready")?;
        if transport.next_line()? != "readyok" {
            return Err("missing readyok".into());
        }
        transport.send(&format!("position fen {fen}"))?;
        transport.send(&format!("go nodes {nodes}"))?;
        let mut score = None;
        let mut reported_nodes = None;
        loop {
            let line = transport.next_line()?;
            if line.starts_with("child-error") {
                return Err(line);
            }
            if let Some(value) = super::parse_info_score_checked(&line)? {
                score = Some(value);
            }
            if let Some(value) = super::parse_info_nodes(&line) {
                reported_nodes = Some(value);
            }
            if let Some(bestmove) = line.strip_prefix("bestmove ") {
                return super::finish_stockfish_label(fen, score, reported_nodes, bestmove);
            }
        }
    }

    #[test]
    fn parses_cp_and_converts_mate_to_documented_clamp() {
        assert_eq!(
            parse_info_score("info depth 12 score cp -37 nodes 25000"),
            Some(-37)
        );
        assert_eq!(
            parse_info_score("info depth 18 score mate 3 nodes 25000"),
            Some(MATE_LABEL_CP)
        );
    }

    #[test]
    fn calibration_chooses_lowest_budget_within_twenty_cp_p95() {
        assert_eq!(choose_budget(&[(25_000, 18), (100_000, 9)]), Some(25_000));
        assert_eq!(choose_budget(&[(25_000, 24), (100_000, 14)]), Some(100_000));
    }

    #[test]
    fn calibration_uses_400k_when_no_lower_budget_meets_the_p95_limit() {
        assert_eq!(choose_budget(&[(25_000, 21), (100_000, 22)]), Some(400_000));
    }

    #[test]
    fn calibration_compares_each_lower_budget_to_the_400k_reference() {
        let labels = |score_cp| {
            vec![StockfishLabel {
                fen: "fen".to_string(),
                score_cp,
                nodes: 1,
            }]
        };
        assert_eq!(
            calibration_candidates(&labels(0), &labels(100), &labels(10)),
            [(25_000, 10), (100_000, 90)]
        );
    }

    #[test]
    fn fake_transport_checks_order_and_parses_a_label() {
        let mut transport = FakeUci {
            commands: Vec::new(),
            replies: VecDeque::from([
                Ok("readyok".to_string()),
                Ok("info depth 9 score mate -2 nodes 25000".to_string()),
                Ok("bestmove e2e4".to_string()),
            ]),
        };
        let label = evaluate_one_with_transport(&mut transport, "fen", 25_000).unwrap();
        assert_eq!(label.score_cp, -MATE_LABEL_CP);
        assert_eq!(
            transport.commands,
            [
                "ucinewgame",
                "isready",
                "position fen fen",
                "go nodes 25000"
            ]
        );
    }

    #[test]
    fn fake_transport_accepts_valid_early_completion_and_records_reported_nodes() {
        let mut transport = FakeUci {
            commands: Vec::new(),
            replies: VecDeque::from([
                Ok("readyok".to_string()),
                Ok("info depth 7 score cp 31 nodes 6624".to_string()),
                Ok("bestmove e2e4".to_string()),
            ]),
        };

        let label = evaluate_one_with_transport(&mut transport, "fen", 25_000).unwrap();
        assert_eq!(label.score_cp, 31);
        assert_eq!(label.nodes, 6_624);
        assert_eq!(
            transport.commands.last(),
            Some(&"go nodes 25000".to_string())
        );
    }

    #[test]
    fn fake_transport_rejects_malformed_scores_timeouts_and_child_errors() {
        let cases = [
            VecDeque::from([Err("timeout".to_string())]),
            VecDeque::from([
                Ok("readyok".to_string()),
                Ok("info depth 9 score cp nope nodes 25000".to_string()),
                Ok("bestmove e2e4".to_string()),
            ]),
            VecDeque::from([
                Ok("readyok".to_string()),
                Ok("child-error status 1".to_string()),
            ]),
        ];
        for replies in cases {
            let mut transport = FakeUci {
                commands: Vec::new(),
                replies,
            };
            assert!(evaluate_one_with_transport(&mut transport, "fen", 25_000).is_err());
        }
    }

    #[test]
    fn fake_transport_rejects_malformed_score_even_when_a_valid_score_follows() {
        let mut transport = FakeUci {
            commands: Vec::new(),
            replies: VecDeque::from([
                Ok("readyok".to_string()),
                Ok("info depth 9 score cp nope nodes 25000".to_string()),
                Ok("info depth 10 score cp 17 nodes 25000".to_string()),
                Ok("bestmove e2e4".to_string()),
            ]),
        };
        assert!(evaluate_one_with_transport(&mut transport, "fen", 25_000).is_err());
    }
}
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

use crate::dataset::sha256_hex;

/// Centipawn target used for every forced mate, regardless of distance.
pub const MATE_LABEL_CP: i32 = 10_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StockfishConfig {
    pub binary: PathBuf,
    pub binary_sha256: String,
    pub hash_mb: u32,
    pub node_budget: u64,
    pub response_timeout: Duration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StockfishLabel {
    pub fen: String,
    pub score_cp: i32,
    pub nodes: u64,
}

pub fn parse_info_score(line: &str) -> Option<i32> {
    parse_info_score_checked(line).ok().flatten()
}

fn parse_info_score_checked(line: &str) -> Result<Option<i32>, String> {
    let fields: Vec<_> = line.split_whitespace().collect();
    let Some(index) = fields.iter().position(|field| *field == "score") else {
        return Ok(None);
    };
    let score = fields
        .get(index..index + 3)
        .ok_or_else(|| format!("malformed Stockfish score output: {line}"))?;
    match score {
        ["score", "cp", value] => value
            .parse()
            .map(Some)
            .map_err(|_| format!("malformed Stockfish centipawn score: {line}")),
        ["score", "mate", value] => value
            .parse::<i32>()
            .map(|mate| {
                Some(if mate.is_negative() {
                    -MATE_LABEL_CP
                } else {
                    MATE_LABEL_CP
                })
            })
            .map_err(|_| format!("malformed Stockfish mate score: {line}")),
        _ => Err(format!("malformed Stockfish score output: {line}")),
    }
}

fn parse_info_nodes(line: &str) -> Option<u64> {
    let fields: Vec<_> = line.split_whitespace().collect();
    fields
        .windows(2)
        .find(|window| window[0] == "nodes")
        .and_then(|window| window[1].parse().ok())
}

fn finish_stockfish_label(
    fen: &str,
    score: Option<i32>,
    nodes: Option<u64>,
    bestmove: &str,
) -> Result<StockfishLabel, String> {
    if bestmove.split_whitespace().next().unwrap_or_default() == "0000" {
        return Err("Stockfish returned no legal move".into());
    }
    Ok(StockfishLabel {
        fen: fen.to_string(),
        score_cp: score
            .ok_or_else(|| "Stockfish returned bestmove without a parseable score".to_string())?,
        nodes: nodes
            .ok_or_else(|| "Stockfish returned bestmove without reported nodes".to_string())?,
    })
}

fn choose_budget(samples: &[(u64, i32)]) -> Option<u64> {
    samples
        .iter()
        .find(|(_, p95_error)| *p95_error <= 20)
        .map(|(budget, _)| *budget)
        .or(Some(400_000))
}

pub fn label_positions(
    config: &StockfishConfig,
    fens: &[String],
) -> Result<Vec<StockfishLabel>, String> {
    verify_binary(config)?;
    let mut process = UciProcess::start(config)?;
    fens.iter()
        .map(|fen| evaluate_one(&mut process, fen, config.node_budget))
        .collect()
}

/// Evaluates 25k, 100k, and 400k exactly, choosing the lowest budget whose
/// P95 deviation from the mandatory 400k reference is at most 20 centipawns.
pub fn calibrate_node_budget(config: &StockfishConfig, fens: &[String]) -> Result<u64, String> {
    if fens.is_empty() {
        return Err("cannot calibrate Stockfish with no positions".into());
    }
    let labels_25k = label_at_budget(config, fens, 25_000)?;
    let labels_100k = label_at_budget(config, fens, 100_000)?;
    let labels_400k = label_at_budget(config, fens, 400_000)?;
    let candidates = calibration_candidates(&labels_25k, &labels_100k, &labels_400k);
    Ok(choose_budget(&candidates).expect("400k fallback is always present"))
}

fn calibration_candidates(
    labels_25k: &[StockfishLabel],
    labels_100k: &[StockfishLabel],
    labels_400k: &[StockfishLabel],
) -> [(u64, i32); 2] {
    [
        (25_000, p95_score_delta(labels_25k, labels_400k)),
        (100_000, p95_score_delta(labels_100k, labels_400k)),
    ]
}

fn p95_score_delta(actual: &[StockfishLabel], reference: &[StockfishLabel]) -> i32 {
    let mut errors: Vec<i32> = actual
        .iter()
        .zip(reference)
        .map(|(actual, reference)| (actual.score_cp - reference.score_cp).abs())
        .collect();
    errors.sort_unstable();
    errors[((errors.len() * 95).saturating_add(99) / 100).saturating_sub(1)]
}

fn label_at_budget(
    config: &StockfishConfig,
    fens: &[String],
    node_budget: u64,
) -> Result<Vec<StockfishLabel>, String> {
    let mut config = config.clone();
    config.node_budget = node_budget;
    label_positions(&config, fens)
}

fn verify_binary(config: &StockfishConfig) -> Result<(), String> {
    if config.node_budget == 0 {
        return Err("Stockfish node budget must be positive".into());
    }
    let actual = sha256_hex(&std::fs::read(&config.binary).map_err(|error| {
        format!(
            "failed to read Stockfish binary {}: {error}",
            config.binary.display()
        )
    })?);
    if !actual.eq_ignore_ascii_case(&config.binary_sha256) {
        return Err(format!(
            "Stockfish binary digest mismatch for {}",
            config.binary.display()
        ));
    }
    Ok(())
}

fn evaluate_one(process: &mut UciProcess, fen: &str, nodes: u64) -> Result<StockfishLabel, String> {
    process.send("ucinewgame")?;
    process.send("isready")?;
    process.wait_for("readyok")?;
    process.send(&format!("position fen {fen}"))?;
    process.send(&format!("go nodes {nodes}"))?;
    process.wait_for_score_and_bestmove(fen, nodes)
}

struct UciProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: Receiver<Result<String, String>>,
    response_timeout: Duration,
}

impl UciProcess {
    fn start(config: &StockfishConfig) -> Result<Self, String> {
        let mut child = Command::new(&config.binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|error| {
                format!(
                    "failed to start Stockfish {}: {error}",
                    config.binary.display()
                )
            })?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| "Stockfish has no stdin".to_string())?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "Stockfish has no stdout".to_string())?;
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                if sender
                    .send(line.map_err(|error| error.to_string()))
                    .is_err()
                {
                    break;
                }
            }
        });
        let mut process = Self {
            child,
            stdin,
            stdout: receiver,
            response_timeout: config.response_timeout,
        };
        process.send("uci")?;
        process.wait_for("uciok")?;
        process.send("setoption name Threads value 1")?;
        process.send(&format!("setoption name Hash value {}", config.hash_mb))?;
        process.send("isready")?;
        process.wait_for("readyok")?;
        Ok(process)
    }

    fn send(&mut self, command: &str) -> Result<(), String> {
        writeln!(self.stdin, "{command}")
            .map_err(|error| format!("failed to send UCI command `{command}`: {error}"))?;
        self.stdin
            .flush()
            .map_err(|error| format!("failed to flush UCI command `{command}`: {error}"))
    }

    fn wait_for(&mut self, expected: &str) -> Result<(), String> {
        loop {
            if self.next_line()? == expected {
                return Ok(());
            }
        }
    }

    fn wait_for_score_and_bestmove(
        &mut self,
        fen: &str,
        _requested_nodes: u64,
    ) -> Result<StockfishLabel, String> {
        let mut score = None;
        let mut reported_nodes = None;
        loop {
            let line = self.next_line()?;
            if line.starts_with("info ") {
                if let Some(value) = parse_info_score_checked(&line)? {
                    score = Some(value);
                }
                if let Some(value) = parse_info_nodes(&line) {
                    reported_nodes = Some(value);
                }
            }
            if let Some(bestmove) = line.strip_prefix("bestmove ") {
                let label = finish_stockfish_label(fen, score, reported_nodes, bestmove)?;
                if let Some(status) = self
                    .child
                    .try_wait()
                    .map_err(|error| format!("failed to inspect Stockfish exit status: {error}"))?
                {
                    if !status.success() {
                        return Err(format!("Stockfish exited unsuccessfully: {status}"));
                    }
                }
                return Ok(label);
            }
        }
    }

    fn next_line(&self) -> Result<String, String> {
        self.stdout
            .recv_timeout(self.response_timeout)
            .map_err(|error| format!("timed out waiting for Stockfish response: {error}"))?
    }
}

impl Drop for UciProcess {
    fn drop(&mut self) {
        let _ = self.send("quit");
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
