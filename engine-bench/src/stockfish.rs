#[cfg(test)]
mod tests {
    use super::{choose_budget, parse_info_score, MATE_LABEL_CP};
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
        loop {
            let line = transport.next_line()?;
            if line.starts_with("child-error") {
                return Err(line);
            }
            if let Some(value) = super::parse_info_score(&line) {
                score = Some(value);
            }
            if line.starts_with("bestmove ") {
                return score
                    .map(|score_cp| super::StockfishLabel {
                        fen: fen.to_string(),
                        score_cp,
                        nodes,
                    })
                    .ok_or_else(|| "missing score".to_string());
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
        assert_eq!(choose_budget(&[(25_000, 18), (100_000, 9)]), 25_000);
        assert_eq!(choose_budget(&[(25_000, 24), (100_000, 14)]), 100_000);
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
    let fields: Vec<_> = line.split_whitespace().collect();
    let score = fields.windows(3).find(|window| window[0] == "score")?;
    match score {
        ["score", "cp", value] => value.parse().ok(),
        ["score", "mate", value] => value.parse::<i32>().ok().map(|mate| {
            if mate.is_negative() {
                -MATE_LABEL_CP
            } else {
                MATE_LABEL_CP
            }
        }),
        _ => None,
    }
}

fn choose_budget(samples: &[(u64, i32)]) -> u64 {
    samples
        .iter()
        .find(|(_, p95_error)| *p95_error <= 20)
        .map(|(budget, _)| *budget)
        .or_else(|| samples.last().map(|(budget, _)| *budget))
        .unwrap_or(0)
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

/// Determines the least standard node budget whose 95th-percentile deviation
/// from a 400k-node reference is at most 20 centipawns.
pub fn calibrate_node_budget(config: &StockfishConfig, fens: &[String]) -> Result<u64, String> {
    if fens.is_empty() {
        return Err("cannot calibrate Stockfish with no positions".into());
    }
    let reference = label_at_budget(config, fens, 400_000)?;
    let mut candidates = Vec::new();
    for budget in [25_000, 100_000, 250_000] {
        let labels = label_at_budget(config, fens, budget)?;
        let mut errors: Vec<i32> = labels
            .iter()
            .zip(&reference)
            .map(|(actual, expected)| (actual.score_cp - expected.score_cp).abs())
            .collect();
        errors.sort_unstable();
        let p95 = errors[((errors.len() * 95).saturating_add(99) / 100).saturating_sub(1)];
        candidates.push((budget, p95));
    }
    Ok(choose_budget(&candidates))
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
        requested_nodes: u64,
    ) -> Result<StockfishLabel, String> {
        let mut score = None;
        let mut reported_nodes = None;
        loop {
            let line = self.next_line()?;
            if line.starts_with("info ") {
                if let Some(value) = parse_info_score(&line) {
                    score = Some(value);
                }
                let fields: Vec<_> = line.split_whitespace().collect();
                if let Some(value) = fields
                    .windows(2)
                    .find(|window| window[0] == "nodes")
                    .and_then(|window| window[1].parse().ok())
                {
                    reported_nodes = Some(value);
                }
            }
            if let Some(bestmove) = line.strip_prefix("bestmove ") {
                if bestmove.split_whitespace().next().unwrap_or_default() == "0000" {
                    return Err("Stockfish returned no legal move".into());
                }
                let score = score.ok_or_else(|| {
                    "Stockfish returned bestmove without a parseable score".to_string()
                })?;
                let nodes = reported_nodes.ok_or_else(|| {
                    "Stockfish returned bestmove without reported nodes".to_string()
                })?;
                if nodes < requested_nodes {
                    return Err(format!(
                        "Stockfish reported {nodes} nodes, below requested {requested_nodes}"
                    ));
                }
                if let Some(status) = self
                    .child
                    .try_wait()
                    .map_err(|error| format!("failed to inspect Stockfish exit status: {error}"))?
                {
                    if !status.success() {
                        return Err(format!("Stockfish exited unsuccessfully: {status}"));
                    }
                }
                return Ok(StockfishLabel {
                    fen: fen.to_string(),
                    score_cp: score,
                    nodes,
                });
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
