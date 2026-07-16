use std::collections::BTreeMap;
use std::io::Cursor;
use std::ops::ControlFlow;
use std::path::Path;

use engine_core::{Board, Color};
use pgn_reader::shakmaty::{Chess, Position, uci::UciMove};
use pgn_reader::{RawTag, Reader, SanPlus, Visitor};

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

struct Builder {
    filter: BookFilter,
    source_games: u32,
    accepted_games: u32,
    counts: BTreeMap<String, BTreeMap<String, u32>>,
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
            *self.counts.entry(fen).or_default().entry(mv).or_default() += points;
        }
    }
}

fn build_book(pgn: &str, filter: BookFilter) -> Result<BookReport, String> {
    let mut reader = Reader::new(Cursor::new(pgn.as_bytes()));
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
    let mut entries = 0;
    let mut alternatives = 0;
    let mut book = String::from("rusty-fish-book v2\n");
    for (fen, moves) in builder.counts {
        let mut moves: Vec<_> = moves
            .into_iter()
            .filter(|(_, weight)| *weight >= 3)
            .collect();
        moves.sort_unstable_by(|(left_move, left_weight), (right_move, right_weight)| {
            right_weight
                .cmp(left_weight)
                .then_with(|| left_move.cmp(right_move))
        });
        if moves.is_empty() {
            continue;
        }
        entries += 1;
        alternatives += moves.len();
        let alternatives = moves
            .into_iter()
            .map(|(mv, weight)| format!("{mv}:{weight}"))
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

fn generate(input: &Path, book: &Path, metrics: &Path) -> Result<(), String> {
    let pgn = std::fs::read_to_string(input)
        .map_err(|error| format!("could not read {}: {error}", input.display()))?;
    let report = build_book(
        &pgn,
        BookFilter {
            min_rating: 2200,
            max_plies: 16,
        },
    )?;
    let metrics_tsv = report.metrics_tsv();
    std::fs::write(book, report.book)
        .map_err(|error| format!("could not write {}: {error}", book.display()))?;
    std::fs::write(metrics, metrics_tsv)
        .map_err(|error| format!("could not write {}: {error}", metrics.display()))?;
    Ok(())
}

fn run(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let program = args.next().unwrap_or_else(|| "book-tool".to_string());
    let Some(command) = args.next() else {
        return Err(format!(
            "usage: {program} generate <input.pgn> <book.txt> <metrics.tsv>"
        ));
    };
    if command != "generate" {
        return Err(format!("unknown command: {command}"));
    }
    let (Some(input), Some(book), Some(metrics)) = (args.next(), args.next(), args.next()) else {
        return Err(format!(
            "usage: {program} generate <input.pgn> <book.txt> <metrics.tsv>"
        ));
    };
    if args.next().is_some() {
        return Err(format!(
            "usage: {program} generate <input.pgn> <book.txt> <metrics.tsv>"
        ));
    }
    generate(Path::new(&input), Path::new(&book), Path::new(&metrics))
}

fn main() -> Result<(), String> {
    run(std::env::args())
}
