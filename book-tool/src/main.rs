use std::collections::{BTreeMap, HashSet};
use std::ops::ControlFlow;
use std::path::Path;

use engine_core::{Board, Color};
use pgn_reader::shakmaty::{Chess, Position, uci::UciMove};
use pgn_reader::{RawTag, Reader, SanPlus, Visitor};

/// The generator records the position before each of the first this-many plies,
/// so `hitrate` checks at most this many positions per line. Both uses must stay
/// equal or the metric misreports.
const BOOK_MAX_PLIES: u32 = 16;

#[derive(Clone, Copy)]
struct BookFilter {
    min_rating: u32,
    max_plies: u32,
}

struct BookReport {
    source_games: u32,
    accepted_games: u32,
    positions: usize,
    entries: usize,
    alternatives: usize,
    book: String,
}

impl BookReport {
    fn metrics_tsv(&self) -> String {
        format!(
            "metric\tvalue\nsource_games\t{}\naccepted_games\t{}\npositions\t{}\nentries\t{}\nalternatives\t{}\n",
            self.source_games, self.accepted_games, self.positions, self.entries, self.alternatives
        )
    }
}

#[derive(Default)]
struct Tags {
    event: String,
    variant: String,
    white_elo: u32,
    black_elo: u32,
    result: String,
}

struct Game {
    tags: Tags,
    chess: Chess,
    board: Board,
    moves: Vec<(String, String, Color)>,
    valid: bool,
}

#[derive(Clone, Copy, Default)]
struct MoveStats {
    weight: u32,
    observations: u32,
}

struct Builder {
    filter: BookFilter,
    source_games: u32,
    accepted_games: u32,
    counts: BTreeMap<String, BTreeMap<String, MoveStats>>,
}

impl Builder {
    fn accepts_tags(&self, tags: &Tags) -> bool {
        tags.white_elo >= self.filter.min_rating
            && tags.black_elo >= self.filter.min_rating
            && tags.event.to_ascii_lowercase().contains("rated")
            && (tags.variant.is_empty() || tags.variant.eq_ignore_ascii_case("standard"))
    }
}

impl Visitor for Builder {
    type Tags = Tags;
    type Movetext = Game;
    type Output = ();

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        self.source_games += 1;
        ControlFlow::Continue(Tags::default())
    }

    fn tag(
        &mut self,
        tags: &mut Tags,
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

    fn begin_movetext(&mut self, tags: Tags) -> ControlFlow<Self::Output, Self::Movetext> {
        let valid = self.accepts_tags(&tags);
        ControlFlow::Continue(Game {
            tags,
            chess: Chess::default(),
            board: Board::startpos(),
            moves: Vec::new(),
            valid,
        })
    }

    fn san(&mut self, game: &mut Game, san: SanPlus) -> ControlFlow<Self::Output> {
        let Ok(mv) = san.san.to_move(&game.chess) else {
            game.valid = false;
            return ControlFlow::Continue(());
        };
        if game.valid && (game.moves.len() as u32) < self.filter.max_plies {
            let uci = UciMove::from_standard(mv.clone()).to_string();
            let signature = game.board.position_signature();
            let side = game.board.side_to_move;
            match game
                .board
                .parse_uci_move(&uci)
                .and_then(|parsed| game.board.make_move(parsed))
            {
                Ok(_) => game.moves.push((signature, uci, side)),
                Err(_) => game.valid = false,
            }
        }
        game.chess.play_unchecked(mv);
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, game: Game) -> Self::Output {
        let result = game.tags.result.as_str();
        if !game.valid || !matches!(result, "1-0" | "0-1" | "1/2-1/2") {
            return;
        }
        self.accepted_games += 1;
        for (fen, mv, side) in game.moves {
            // Twice the specified 0.5 + score_fraction weight, kept integral.
            let points = match (result, side) {
                ("1-0", Color::White) | ("0-1", Color::Black) => 3,
                ("1/2-1/2", _) => 2,
                _ => 1,
            };
            let stats = self.counts.entry(fen).or_default().entry(mv).or_default();
            stats.weight += points;
            stats.observations += 1;
        }
    }
}

fn load_book_signatures(book: &str) -> Result<HashSet<String>, String> {
    let mut lines = book.lines();
    match lines.next() {
        Some("rusty-fish-book v2") => {}
        _ => return Err("book is missing the rusty-fish-book v2 header".to_string()),
    }
    let mut signatures = HashSet::new();
    for line in lines {
        let Some((signature, _)) = line.split_once('\t') else {
            return Err(format!("malformed book line: {line}"));
        };
        signatures.insert(signature.to_string());
    }
    Ok(signatures)
}

struct HitRate {
    signatures: HashSet<String>,
    lines: u32,
    plies_checked: u64,
    plies_in_book: u64,
    depth_sum: u64,
    fully_covered_lines: u32,
    error: Option<String>,
}

struct HitLine {
    chess: Chess,
    board: Board,
    checked: u32,
    in_book: u32,
    depth: u32,
    consecutive: bool,
    valid: bool,
}

impl Visitor for HitRate {
    type Tags = ();
    type Movetext = HitLine;
    type Output = ();

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, _tags: ()) -> ControlFlow<Self::Output, Self::Movetext> {
        ControlFlow::Continue(HitLine {
            chess: Chess::default(),
            board: Board::startpos(),
            checked: 0,
            in_book: 0,
            depth: 0,
            consecutive: true,
            valid: true,
        })
    }

    fn san(&mut self, line: &mut HitLine, san: SanPlus) -> ControlFlow<Self::Output> {
        if !line.valid || line.checked >= BOOK_MAX_PLIES {
            return ControlFlow::Continue(());
        }
        let Ok(mv) = san.san.to_move(&line.chess) else {
            self.error = Some(format!("illegal or unparsable suite move: {}", san.san));
            line.valid = false;
            return ControlFlow::Continue(());
        };
        let signature = line.board.position_signature();
        let hit = self.signatures.contains(&signature);
        line.checked += 1;
        if hit {
            line.in_book += 1;
            if line.consecutive {
                line.depth += 1;
            }
        } else {
            line.consecutive = false;
        }
        let uci = UciMove::from_standard(mv.clone()).to_string();
        match line
            .board
            .parse_uci_move(&uci)
            .and_then(|parsed| line.board.make_move(parsed))
        {
            Ok(_) => line.chess.play_unchecked(mv),
            Err(_) => {
                self.error = Some(format!("illegal suite move: {uci}"));
                line.valid = false;
            }
        }
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, line: HitLine) -> Self::Output {
        if !line.valid {
            return;
        }
        self.lines += 1;
        self.plies_checked += u64::from(line.checked);
        self.plies_in_book += u64::from(line.in_book);
        self.depth_sum += u64::from(line.depth);
        if line.checked > 0 && line.in_book == line.checked {
            self.fully_covered_lines += 1;
        }
    }
}

fn hitrate(book_path: &Path, suite_path: &Path) -> Result<(), String> {
    let book = std::fs::read_to_string(book_path)
        .map_err(|error| format!("could not read {}: {error}", book_path.display()))?;
    let signatures = load_book_signatures(&book)?;

    let suite = std::fs::read_to_string(suite_path)
        .map_err(|error| format!("could not read {}: {error}", suite_path.display()))?;
    let mut visitor = HitRate {
        signatures,
        lines: 0,
        plies_checked: 0,
        plies_in_book: 0,
        depth_sum: 0,
        fully_covered_lines: 0,
        error: None,
    };
    let mut reader = Reader::new(suite.as_bytes());
    reader
        .visit_all_games(&mut visitor)
        .map_err(|error| error.to_string())?;
    if let Some(error) = visitor.error {
        return Err(error);
    }
    if visitor.plies_checked == 0 {
        return Err("suite checked no positions".to_string());
    }

    let hit_rate = visitor.plies_in_book as f64 / visitor.plies_checked as f64;
    let mean_book_depth = visitor.depth_sum as f64 / f64::from(visitor.lines);
    print!(
        "metric\tvalue\nlines\t{}\nplies_checked\t{}\nplies_in_book\t{}\nhit_rate\t{hit_rate:.6}\nmean_book_depth\t{mean_book_depth:.6}\nfully_covered_lines\t{}\n",
        visitor.lines, visitor.plies_checked, visitor.plies_in_book, visitor.fully_covered_lines
    );
    Ok(())
}

fn build_book<R: std::io::Read>(
    pgn: R,
    filter: BookFilter,
    max_positions: Option<usize>,
) -> Result<BookReport, String> {
    let mut reader = Reader::new(pgn);
    let mut builder = Builder {
        filter,
        source_games: 0,
        accepted_games: 0,
        counts: BTreeMap::new(),
    };
    reader
        .visit_all_games(&mut builder)
        .map_err(|error| error.to_string())?;

    let positions = builder.counts.len();
    let mut retained: Vec<(String, Vec<(String, MoveStats)>, u32)> = Vec::new();
    for (fen, moves) in builder.counts {
        let mut moves: Vec<_> = moves
            .into_iter()
            .filter(|(_, stats)| stats.observations >= 3)
            .collect();
        if moves.is_empty() {
            continue;
        }
        moves.sort_unstable_by(|(left_move, left), (right_move, right)| {
            right
                .weight
                .cmp(&left.weight)
                .then_with(|| left_move.cmp(right_move))
        });
        let observations = moves.iter().map(|(_, stats)| stats.observations).sum();
        retained.push((fen, moves, observations));
    }

    if let Some(max_positions) = max_positions
        && retained.len() > max_positions
    {
        // Rank by observations descending, breaking ties on ascending FEN so the
        // retained set is stable across runs, then restore FEN order for output.
        retained.sort_unstable_by(|(left_fen, _, left), (right_fen, _, right)| {
            right.cmp(left).then_with(|| left_fen.cmp(right_fen))
        });
        retained.truncate(max_positions);
        retained.sort_unstable_by(|(left_fen, _, _), (right_fen, _, _)| left_fen.cmp(right_fen));
    }

    let mut entries = 0;
    let mut alternatives = 0;
    let mut book = String::from("rusty-fish-book v2\n");
    for (fen, moves, _) in retained {
        entries += 1;
        alternatives += moves.len();
        let alternatives = moves
            .into_iter()
            .map(|(mv, stats)| format!("{mv}:{}", stats.weight))
            .collect::<Vec<_>>()
            .join(" ");
        book.push_str(&format!("{fen}\t{alternatives}\n"));
    }
    Ok(BookReport {
        source_games: builder.source_games,
        accepted_games: builder.accepted_games,
        positions,
        entries,
        alternatives,
        book,
    })
}

fn generate(
    input: &Path,
    book: &Path,
    metrics: &Path,
    max_positions: Option<usize>,
) -> Result<(), String> {
    let filter = BookFilter {
        min_rating: 2200,
        max_plies: BOOK_MAX_PLIES,
    };
    let report = if input == Path::new("-") {
        build_book(std::io::stdin().lock(), filter, max_positions)?
    } else {
        let file = std::fs::File::open(input)
            .map_err(|error| format!("could not read {}: {error}", input.display()))?;
        build_book(std::io::BufReader::new(file), filter, max_positions)?
    };
    let metrics_tsv = report.metrics_tsv();
    std::fs::write(book, report.book)
        .map_err(|error| format!("could not write {}: {error}", book.display()))?;
    std::fs::write(metrics, metrics_tsv)
        .map_err(|error| format!("could not write {}: {error}", metrics.display()))?;
    Ok(())
}

fn run(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let program = args.next().unwrap_or_else(|| "book-tool".to_string());
    let usage = format!(
        "usage: {program} generate <input.pgn> <book.txt> <metrics.tsv> [--max-positions N]\n       {program} hitrate <book.txt> <suite.pgn>"
    );
    let Some(command) = args.next() else {
        return Err(usage);
    };
    match command.as_str() {
        "generate" => {
            let mut positional = Vec::new();
            let mut max_positions = None;
            while let Some(arg) = args.next() {
                if arg == "--max-positions" {
                    let value = args.next().ok_or_else(|| usage.clone())?;
                    let parsed = value
                        .parse::<usize>()
                        .map_err(|_| format!("invalid --max-positions value: {value}"))?;
                    max_positions = Some(parsed);
                } else {
                    positional.push(arg);
                }
            }
            let [input, book, metrics] = positional.as_slice() else {
                return Err(usage);
            };
            generate(
                Path::new(input),
                Path::new(book),
                Path::new(metrics),
                max_positions,
            )
        }
        "hitrate" => {
            let positional: Vec<String> = args.collect();
            let [book, suite] = positional.as_slice() else {
                return Err(usage);
            };
            hitrate(Path::new(book), Path::new(suite))
        }
        other => Err(format!("unknown command: {other}")),
    }
}

fn main() -> Result<(), String> {
    run(std::env::args())
}
