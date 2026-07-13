use std::io::{self, BufRead, Write};
use std::time::Duration;

use engine_core::{Board, CLASSIC_STARTPOS_FEN};
use engine_search::{ClockControl, SearchInfo, SearchLimits, SearchOptions, Searcher};

fn main() -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut state = EngineState::default();

    for line in stdin.lock().lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed == "uci" {
            write_uci_header(&mut stdout, &state.options)?;
        } else if trimmed == "isready" {
            writeln!(stdout, "readyok")?;
        } else if trimmed == "ucinewgame" {
            state.board = Board::startpos();
        } else if trimmed.starts_with("position ") {
            if let Err(err) = apply_position(&mut state.board, trimmed) {
                writeln!(stdout, "info string position error: {err}")?;
            }
        } else if trimmed.starts_with("setoption ") {
            if let Err(err) = apply_option(&mut state, trimmed) {
                writeln!(stdout, "info string setoption error: {err}")?;
            }
        } else if trimmed.starts_with("go") {
            let limits = parse_go(trimmed);
            let result = state.searcher.search_with_callback(&state.board, limits, |info| {
                print_info(info);
            });
            if let Some(best_move) = result.best_move {
                writeln!(stdout, "bestmove {best_move}")?;
            } else {
                writeln!(stdout, "bestmove 0000")?;
            }
        } else if trimmed == "d" {
            writeln!(stdout, "info string fen {}", state.board.to_fen())?;
        } else if trimmed == "quit" {
            break;
        }
        stdout.flush()?;
    }

    Ok(())
}

#[derive(Default)]
struct EngineState {
    board: Board,
    searcher: Searcher,
    options: SearchOptions,
}

fn write_uci_header(mut stdout: impl Write, options: &SearchOptions) -> io::Result<()> {
    writeln!(stdout, "id name Rusty Fish")?;
    writeln!(stdout, "id author Ben Severn + Codex")?;
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
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "missing option value".to_string())?;

    match name.as_str() {
        "Hash" => {
            let hash_mb = value
                .parse::<usize>()
                .map_err(|_| format!("invalid Hash value: {value}"))?;
            state.options.hash_mb = hash_mb.clamp(1, 1024);
        }
        "Move Overhead" => {
            let ms = value
                .parse::<u64>()
                .map_err(|_| format!("invalid Move Overhead value: {value}"))?;
            state.options.move_overhead = Duration::from_millis(ms.min(5_000));
        }
        "Max Depth" => {
            let depth = value
                .parse::<u8>()
                .map_err(|_| format!("invalid Max Depth value: {value}"))?;
            state.options.max_depth = depth.clamp(1, 64);
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

fn print_info(info: &SearchInfo) {
    let pv = info
        .pv
        .iter()
        .map(|mv| mv.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "info depth {} score cp {} nodes {} time {} pv {}",
        info.depth,
        info.score_cp,
        info.nodes,
        info.elapsed.as_millis(),
        pv
    );
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use engine_search::ClockControl;

    use super::{apply_option, parse_go, EngineState};

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
        assert_eq!(state.options.hash_mb, 64);
        assert_eq!(state.options.move_overhead, Duration::from_millis(100));
        assert_eq!(state.options.max_depth, 20);
        assert_eq!(state.searcher.options().hash_mb, 64);
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
}
