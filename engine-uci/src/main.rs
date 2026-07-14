use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::{Arc, mpsc};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use engine_core::{Board, CLASSIC_STARTPOS_FEN};
use engine_search::{
    ClockControl, Nnue, SearchLimits, SearchOptions, SearchResult, Searcher, SyzygyTablebases,
};

fn main() -> io::Result<()> {
    let mut stdout = io::stdout();
    let mut state = EngineState::default();
    let (command_tx, command_rx) = mpsc::channel();
    thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(line) => {
                    if command_tx.send(line).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
    let mut active_search: Option<ActiveSearch> = None;

    loop {
        if let Some(search) = active_search.as_ref()
            && let Ok(result) = search.result_rx.try_recv()
        {
            active_search
                .take()
                .expect("active search must be present")
                .worker
                .join()
                .expect("search worker panicked");
            write_best_move(&mut stdout, result)?;
            stdout.flush()?;
        }

        let line = match command_rx.recv_timeout(Duration::from_millis(10)) {
            Ok(line) => line,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed == "uci" {
            write_uci_header(&mut stdout, &state.options)?;
        } else if trimmed == "isready" {
            writeln!(stdout, "readyok")?;
        } else if trimmed == "ucinewgame" {
            stop_and_join_active_search(&mut active_search);
            state.board = Board::startpos();
        } else if trimmed.starts_with("position ") {
            stop_and_join_active_search(&mut active_search);
            if let Err(err) = apply_position(&mut state.board, trimmed) {
                writeln!(stdout, "info string position error: {err}")?;
            }
        } else if trimmed.starts_with("setoption ") {
            stop_and_join_active_search(&mut active_search);
            if let Err(err) = apply_option(&mut state, trimmed) {
                writeln!(stdout, "info string setoption error: {err}")?;
            }
        } else if trimmed.starts_with("go") {
            stop_and_join_active_search(&mut active_search);
            active_search = Some(start_search(
                state.board.clone(),
                state.options.clone(),
                state.syzygy_path.clone(),
                state.nnue.clone(),
                parse_go(trimmed),
            ));
        } else if trimmed == "stop" {
            stop_active_search(&active_search);
        } else if trimmed == "d" {
            writeln!(stdout, "info string fen {}", state.board.to_fen())?;
        } else if trimmed == "quit" {
            stop_active_search(&active_search);
            break;
        }
        stdout.flush()?;
    }

    Ok(())
}

struct ActiveSearch {
    stop_signal: Arc<AtomicBool>,
    result_rx: mpsc::Receiver<SearchResult>,
    worker: JoinHandle<()>,
}

fn start_search(
    board: Board,
    options: SearchOptions,
    syzygy_path: Option<String>,
    nnue: Option<Arc<Nnue>>,
    limits: SearchLimits,
) -> ActiveSearch {
    let stop_signal = Arc::new(AtomicBool::new(false));
    let worker_signal = Arc::clone(&stop_signal);
    let (result_tx, result_rx) = mpsc::channel();
    let worker = thread::spawn(move || {
        let mut searcher = Searcher::default();
        let syzygy_probe_limit = options.syzygy_probe_limit;
        searcher.set_options(options);
        searcher.set_syzygy_tablebases(
            syzygy_path.and_then(|path| SyzygyTablebases::load(&path, syzygy_probe_limit).ok()),
        );
        searcher.set_nnue(nnue);
        let result = searcher.search_with_stop_signal(&board, limits, worker_signal);
        let _ = result_tx.send(result);
    });
    ActiveSearch {
        stop_signal,
        result_rx,
        worker,
    }
}

fn stop_active_search(active_search: &Option<ActiveSearch>) {
    if let Some(search) = active_search {
        search.stop_signal.store(true, Ordering::Relaxed);
    }
}

fn stop_and_join_active_search(active_search: &mut Option<ActiveSearch>) {
    stop_active_search(active_search);
    if let Some(search) = active_search.take() {
        search.worker.join().expect("search worker panicked");
    }
}

fn write_best_move(mut stdout: impl Write, result: SearchResult) -> io::Result<()> {
    if let Some(best_move) = result.best_move {
        writeln!(stdout, "bestmove {best_move}")
    } else {
        writeln!(stdout, "bestmove 0000")
    }
}

#[derive(Default)]
struct EngineState {
    board: Board,
    searcher: Searcher,
    options: SearchOptions,
    syzygy_path: Option<String>,
    nnue: Option<Arc<Nnue>>,
}

fn write_uci_header(mut stdout: impl Write, options: &SearchOptions) -> io::Result<()> {
    writeln!(stdout, "id name Rusty Fish")?;
    writeln!(stdout, "id author Ben Severn + Codex")?;
    writeln!(stdout, "option name SyzygyPath type string default")?;
    writeln!(stdout, "option name SyzygyProbeDepth type spin default {} min 1 max 64", options.syzygy_probe_depth)?;
    writeln!(stdout, "option name SyzygyProbeLimit type spin default {} min 3 max 7", options.syzygy_probe_limit)?;
    writeln!(
        stdout,
        "option name Hash type spin default {} min 1 max 1024",
        options.hash_mb
    )?;
    writeln!(
        stdout,
        "option name Move Overhead type spin default {} min 0 max 5000",
        options.move_overhead.as_millis()
    )?;
    writeln!(
        stdout,
        "option name Max Depth type spin default {} min 1 max 64",
        options.max_depth
    )?;
    writeln!(
        stdout,
        "option name Threads type spin default {} min 1 max 256",
        options.threads
    )?;
    writeln!(stdout, "option name EvalFile type string default")?;
    writeln!(stdout, "uciok")
}

fn apply_position(board: &mut Board, command: &str) -> Result<(), String> {
    let rest = command
        .strip_prefix("position ")
        .ok_or_else(|| "missing position command".to_string())?;

    let mut parts = rest.split(" moves ");
    let base = parts.next().unwrap_or_default();
    let moves = parts.next();

    if base == "startpos" {
        *board = Board::from_fen(CLASSIC_STARTPOS_FEN)?;
    } else if let Some(fen) = base.strip_prefix("fen ") {
        *board = Board::from_fen(fen)?;
    } else {
        return Err(format!("unsupported position command: {command}"));
    }

    if let Some(moves) = moves {
        for mv_text in moves.split_whitespace() {
            let mv = board.parse_uci_move(mv_text)?;
            board.make_move(mv)?;
        }
    }

    Ok(())
}

fn apply_option(state: &mut EngineState, command: &str) -> Result<(), String> {
    let tail = command
        .strip_prefix("setoption ")
        .ok_or_else(|| "missing setoption prefix".to_string())?;
    let tokens: Vec<&str> = tail.split_whitespace().collect();
    if tokens.len() < 4 || tokens[0] != "name" {
        return Err("expected `setoption name <...> value <...>`".to_string());
    }
    let value_idx = tokens
        .iter()
        .position(|token| *token == "value")
        .ok_or_else(|| "missing `value` in setoption".to_string())?;
    let name = tokens[1..value_idx].join(" ");
    let value = tokens
        .get(value_idx + 1..)
        .map(|parts| parts.join(" "))
        .filter(|s| !s.is_empty());

    match name.as_str() {
        "SyzygyPath" => {
            let path = value.unwrap_or_default();
            if !path.is_empty() && path.split(';').any(|entry| !Path::new(entry).is_dir()) {
                return Err(format!("Syzygy tablebase directory does not exist: {path}"));
            }
            state.syzygy_path = (!path.is_empty()).then_some(path);
        }
        "SyzygyProbeDepth" => {
            let value = value.ok_or_else(|| "missing option value".to_string())?;
            state.options.syzygy_probe_depth = value.parse::<u8>().map_err(|_| format!("invalid SyzygyProbeDepth value: {value}"))?.clamp(1, 64);
        }
        "SyzygyProbeLimit" => {
            let value = value.ok_or_else(|| "missing option value".to_string())?;
            state.options.syzygy_probe_limit = value.parse::<u8>().map_err(|_| format!("invalid SyzygyProbeLimit value: {value}"))?.clamp(3, 7);
        }
        "Hash" => {
            let value = value.ok_or_else(|| "missing option value".to_string())?;
            let hash_mb = value
                .parse::<usize>()
                .map_err(|_| format!("invalid Hash value: {value}"))?;
            state.options.hash_mb = hash_mb.clamp(1, 1024);
        }
        "Move Overhead" => {
            let value = value.ok_or_else(|| "missing option value".to_string())?;
            let ms = value
                .parse::<u64>()
                .map_err(|_| format!("invalid Move Overhead value: {value}"))?;
            state.options.move_overhead = Duration::from_millis(ms.min(5_000));
        }
        "Max Depth" => {
            let value = value.ok_or_else(|| "missing option value".to_string())?;
            let depth = value
                .parse::<u8>()
                .map_err(|_| format!("invalid Max Depth value: {value}"))?;
            state.options.max_depth = depth.clamp(1, 64);
        }
        "Threads" => {
            let value = value.ok_or_else(|| "missing option value".to_string())?;
            let threads = value
                .parse::<usize>()
                .map_err(|_| format!("invalid Threads value: {value}"))?;
            state.options.threads = threads.clamp(1, 256);
        }
        "EvalFile" => {
            match value {
                None => state.nnue = None,
                Some(path) => {
                    let nnue = Nnue::from_file(&path)?;
                    state.nnue = Some(Arc::new(nnue));
                }
            }
        }
        _ => return Err(format!("unsupported option: {name}")),
    }

    state.searcher.set_options(state.options.clone());
    Ok(())
}

fn parse_go(command: &str) -> SearchLimits {
    let tokens: Vec<&str> = command.split_whitespace().collect();
    let mut limits = SearchLimits::default();
    let mut white_time = None;
    let mut black_time = None;
    let mut white_increment = Duration::ZERO;
    let mut black_increment = Duration::ZERO;
    let mut moves_to_go = None;

    let mut idx = 1;
    while idx < tokens.len() {
        match tokens[idx] {
            "depth" if idx + 1 < tokens.len() => {
                limits.depth = tokens[idx + 1].parse::<u8>().ok();
                idx += 2;
            }
            "movetime" if idx + 1 < tokens.len() => {
                limits.movetime = parse_millis(tokens[idx + 1]).map(Duration::from_millis);
                idx += 2;
            }
            "wtime" if idx + 1 < tokens.len() => {
                white_time = parse_millis(tokens[idx + 1]).map(Duration::from_millis);
                idx += 2;
            }
            "btime" if idx + 1 < tokens.len() => {
                black_time = parse_millis(tokens[idx + 1]).map(Duration::from_millis);
                idx += 2;
            }
            "winc" if idx + 1 < tokens.len() => {
                white_increment = parse_millis(tokens[idx + 1])
                    .map(Duration::from_millis)
                    .unwrap_or(Duration::ZERO);
                idx += 2;
            }
            "binc" if idx + 1 < tokens.len() => {
                black_increment = parse_millis(tokens[idx + 1])
                    .map(Duration::from_millis)
                    .unwrap_or(Duration::ZERO);
                idx += 2;
            }
            "movestogo" if idx + 1 < tokens.len() => {
                moves_to_go = tokens[idx + 1].parse::<u32>().ok();
                idx += 2;
            }
            "infinite" => {
                limits.infinite = true;
                idx += 1;
            }
            _ => idx += 1,
        }
    }

    if let (Some(white_time), Some(black_time)) = (white_time, black_time) {
        limits.clock = Some(ClockControl {
            white_time,
            black_time,
            white_increment,
            black_increment,
            moves_to_go,
        });
    }

    limits
}

fn parse_millis(raw: &str) -> Option<u64> {
    raw.parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, mpsc};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::Duration;

    use engine_search::ClockControl;

    use super::{
        ActiveSearch, EngineState, apply_option, parse_go, stop_active_search, write_uci_header,
    };

    #[test]
    fn parse_go_supports_clock_controls() {
        let limits = parse_go("go wtime 60000 btime 45000 winc 1000 binc 500 movestogo 20");
        let clock = limits.clock.expect("clock limits");
        assert_eq!(clock.white_time, Duration::from_millis(60_000));
        assert_eq!(clock.black_time, Duration::from_millis(45_000));
        assert_eq!(clock.white_increment, Duration::from_millis(1_000));
        assert_eq!(clock.black_increment, Duration::from_millis(500));
        assert_eq!(clock.moves_to_go, Some(20));
    }

    #[test]
    fn parse_go_supports_movetime_and_depth() {
        let limits = parse_go("go depth 6 movetime 1500");
        assert_eq!(limits.depth, Some(6));
        assert_eq!(limits.movetime, Some(Duration::from_millis(1_500)));
        assert!(limits.clock.is_none());
    }

    #[test]
    fn setoption_updates_searcher_options() {
        let mut state = EngineState::default();
        apply_option(&mut state, "setoption name Hash value 64").unwrap();
        apply_option(&mut state, "setoption name Move Overhead value 100").unwrap();
        apply_option(&mut state, "setoption name Max Depth value 20").unwrap();
        apply_option(&mut state, "setoption name Threads value 8").unwrap();
        assert_eq!(state.options.hash_mb, 64);
        assert_eq!(state.options.move_overhead, Duration::from_millis(100));
        assert_eq!(state.options.max_depth, 20);
        assert_eq!(state.options.threads, 8);
        assert_eq!(state.searcher.options().hash_mb, 64);
    }

    #[test]
    fn threads_option_is_advertised_and_clamped() {
        let mut state = EngineState::default();
        apply_option(&mut state, "setoption name Threads value 0").unwrap();
        assert_eq!(state.options.threads, 1);
        apply_option(&mut state, "setoption name Threads value 9999").unwrap();
        assert_eq!(state.options.threads, 256);

        let mut header = Vec::new();
        write_uci_header(&mut header, &state.options).unwrap();
        let header = String::from_utf8(header).unwrap();
        assert!(header.contains("option name Threads type spin default 256 min 1 max 256"));
    }

    #[test]
    fn eval_file_loads_a_network_and_keeps_it_on_error() {
        let mut state = EngineState::default();
        let path = std::env::temp_dir()
            .join(format!("rusty-fish-net-{}.rfnn", std::process::id()));
        std::fs::write(&path, engine_search::Nnue::from_seed(5, 8).to_bytes()).unwrap();

        let command = format!("setoption name EvalFile value {}", path.display());
        apply_option(&mut state, &command).unwrap();
        assert!(state.nnue.is_some());

        // A missing file errors and keeps the previously loaded network.
        assert!(
            apply_option(&mut state, "setoption name EvalFile value /nonexistent/rusty-fish.rfnn")
                .is_err()
        );
        assert!(state.nnue.is_some());

        let mut header = Vec::new();
        write_uci_header(&mut header, &state.options).unwrap();
        assert!(String::from_utf8(header).unwrap().contains("option name EvalFile type string"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn syzygy_path_keeps_the_previous_path_on_error() {
        let mut state = EngineState::default();
        apply_option(&mut state, "setoption name SyzygyPath value .").unwrap();
        assert_eq!(state.syzygy_path.as_deref(), Some("."));
        assert!(apply_option(&mut state, "setoption name SyzygyPath value missing-tables").is_err());
        assert_eq!(state.syzygy_path.as_deref(), Some("."));
    }

    #[test]
    fn syzygy_probe_options_are_advertised_and_clamped() {
        let mut state = EngineState::default();
        apply_option(&mut state, "setoption name SyzygyProbeDepth value 0").unwrap();
        apply_option(&mut state, "setoption name SyzygyProbeLimit value 99").unwrap();
        assert_eq!(state.options.syzygy_probe_depth, 1);
        assert_eq!(state.options.syzygy_probe_limit, 7);

        let mut header = Vec::new();
        write_uci_header(&mut header, &state.options).unwrap();
        let header = String::from_utf8(header).unwrap();
        assert!(header.contains("option name SyzygyProbeDepth type spin default 1 min 1 max 64"));
        assert!(header.contains("option name SyzygyProbeLimit type spin default 7 min 3 max 7"));
    }

    #[test]
    fn clock_control_type_is_constructible() {
        let clock = ClockControl {
            white_time: Duration::from_secs(60),
            black_time: Duration::from_secs(60),
            white_increment: Duration::from_secs(1),
            black_increment: Duration::from_secs(1),
            moves_to_go: Some(30),
        };
        assert_eq!(clock.moves_to_go, Some(30));
    }

    #[test]
    fn stop_command_signal_is_shared_with_active_search() {
        let stop_signal = Arc::new(AtomicBool::new(false));
        let (_result_tx, result_rx) = mpsc::channel();
        let active = Some(ActiveSearch {
            stop_signal: Arc::clone(&stop_signal),
            result_rx,
            worker: thread::spawn(|| {}),
        });

        stop_active_search(&active);

        assert!(stop_signal.load(Ordering::Relaxed));
    }
}
