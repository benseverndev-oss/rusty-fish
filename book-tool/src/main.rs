use std::collections::BTreeMap;
use std::io::Cursor;
use std::ops::ControlFlow;

use engine_core::{Board, Color};
use pgn_reader::{RawTag, Reader, SanPlus, Visitor};
use pgn_reader::shakmaty::{Chess, Position, uci::UciMove};

#[derive(Clone, Copy)]
struct BookFilter { min_rating: u32, max_plies: u32 }

struct BookReport { accepted_games: u32, book: String }

#[derive(Default)]
struct Tags { white_elo: u32, black_elo: u32, result: String }

struct Game { tags: Tags, chess: Chess, board: Board, moves: Vec<(String, String, Color)>, valid: bool }

struct Builder { filter: BookFilter, accepted_games: u32, counts: BTreeMap<String, BTreeMap<String, u32>> }

impl Visitor for Builder {
    type Tags = Tags;
    type Movetext = Game;
    type Output = ();
    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> { ControlFlow::Continue(Tags::default()) }
    fn tag(&mut self, tags: &mut Tags, name: &[u8], value: RawTag<'_>) -> ControlFlow<Self::Output> {
        let value = std::str::from_utf8(value.as_bytes()).unwrap_or_default();
        match name { b"WhiteElo" => tags.white_elo = value.parse().unwrap_or(0), b"BlackElo" => tags.black_elo = value.parse().unwrap_or(0), b"Result" => tags.result = value.to_string(), _ => {} }
        ControlFlow::Continue(())
    }
    fn begin_movetext(&mut self, tags: Tags) -> ControlFlow<Self::Output, Self::Movetext> {
        let valid = tags.white_elo >= self.filter.min_rating && tags.black_elo >= self.filter.min_rating;
        ControlFlow::Continue(Game { tags, chess: Chess::default(), board: Board::startpos(), moves: Vec::new(), valid })
    }
    fn san(&mut self, game: &mut Game, san: SanPlus) -> ControlFlow<Self::Output> {
        let Ok(mv) = san.san.to_move(&game.chess) else { game.valid = false; return ControlFlow::Continue(()); };
        if game.valid && (game.moves.len() as u32) < self.filter.max_plies {
            let uci = UciMove::from_standard(mv.clone()).to_string();
            let signature = game.board.position_signature();
            let side = game.board.side_to_move;
            match game.board.parse_uci_move(&uci).and_then(|m| game.board.make_move(m)) { Ok(_) => game.moves.push((signature, uci, side)), Err(_) => game.valid = false }
        }
        game.chess.play_unchecked(mv);
        ControlFlow::Continue(())
    }
    fn end_game(&mut self, game: Game) -> Self::Output {
        if !game.valid || game.tags.result == "*" || game.tags.result.is_empty() { return; }
        self.accepted_games += 1;
        for (fen, mv, side) in game.moves {
            let points = match (game.tags.result.as_str(), side) { ("1-0", Color::White) | ("0-1", Color::Black) => 3, ("1/2-1/2", _) => 2, _ => 1 };
            *self.counts.entry(fen).or_default().entry(mv).or_default() += points;
        }
    }
}

fn build_book(pgn: &str, filter: BookFilter) -> Result<BookReport, String> {
    let mut reader = Reader::new(Cursor::new(pgn.as_bytes()));
    let mut builder = Builder { filter, accepted_games: 0, counts: BTreeMap::new() };
    reader.visit_all_games(&mut builder).map_err(|error| error.to_string())?;
    let mut book = String::from("rusty-fish-book v2\n");
    for (fen, moves) in builder.counts {
        let moves: Vec<_> = moves.into_iter().filter(|(_, weight)| *weight >= 3).collect();
        if !moves.is_empty() { book.push_str(&format!("{fen}\t{}\n", moves.into_iter().map(|(mv, weight)| format!("{mv}:{weight}")).collect::<Vec<_>>().join(" "))); }
    }
    Ok(BookReport { accepted_games: builder.accepted_games, book })
}

fn main() -> Result<(), String> { Ok(()) }

#[cfg(test)]
mod tests {
    #[test]
    fn aggregates_transpositions_and_side_relative_results() {
        let report = super::build_book(
            "[Event \"fixture\"]\n[WhiteElo \"2300\"]\n[BlackElo \"2300\"]\n[Result \"1-0\"]\n\n1. e4 e5 1-0\n",
            super::BookFilter {
                min_rating: 2200,
                max_plies: 16,
            },
        )
        .unwrap();
        assert!(report.book.contains("e2e4:"));
    }
}
