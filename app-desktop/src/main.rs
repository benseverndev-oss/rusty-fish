use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Duration;

use eframe::egui::{self, Color32, RichText, ScrollArea, Vec2};
use engine_core::{piece_unicode, Board, ChessMove, GameStatus, PieceKind, Square};
use engine_search::{SearchLimits, SearchOptions, SearchResult, Searcher};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Rusty Fish",
        options,
        Box::new(|_cc| Ok(Box::new(RustyFishApp::default()))),
    )
}

struct RustyFishApp {
    initial_board: Board,
    board: Board,
    moves: Vec<ChessMove>,
    move_labels: Vec<String>,
    current_ply: usize,
    selected_square: Option<Square>,
    highlighted_targets: Vec<Square>,
    fen_input: String,
    pgn_text: String,
    status_text: String,
    analysis_depth: u8,
    hash_mb: usize,
    move_overhead_ms: u64,
    max_depth: u8,
    engine_service: EngineService,
    latest_result: Option<SearchResult>,
    apply_engine_move_on_result: bool,
}

impl Default for RustyFishApp {
    fn default() -> Self {
        let board = Board::startpos();
        Self {
            initial_board: board.clone(),
            board: board.clone(),
            moves: Vec::new(),
            move_labels: Vec::new(),
            current_ply: 0,
            selected_square: None,
            highlighted_targets: Vec::new(),
            fen_input: board.to_fen(),
            pgn_text: String::new(),
            status_text: "Ready".to_string(),
            analysis_depth: 4,
            hash_mb: 16,
            move_overhead_ms: 25,
            max_depth: 16,
            engine_service: EngineService::new(),
            latest_result: None,
            apply_engine_move_on_result: false,
        }
    }
}

impl eframe::App for RustyFishApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_search();

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("New Game").clicked() {
                    self.reset_to_board(Board::startpos(), "New game");
                }
                if ui.button("Analyze Position").clicked() && !self.engine_service.is_searching {
                    self.sync_engine_options();
                    self.start_search(false);
                }
                if ui.button("Play Engine Move").clicked() && !self.engine_service.is_searching {
                    self.sync_engine_options();
                    self.start_search(true);
                }
                ui.label("Depth");
                ui.add(egui::Slider::new(&mut self.analysis_depth, 1..=self.max_depth.min(32)));
                if self.engine_service.is_searching {
                    ui.label("Engine: searching");
                }
            });
            ui.horizontal(|ui| {
                ui.label("FEN");
                ui.text_edit_singleline(&mut self.fen_input);
                if ui.button("Load FEN").clicked() {
                    match Board::from_fen(&self.fen_input) {
                        Ok(board) => self.reset_to_board(board, "Loaded FEN"),
                        Err(err) => self.status_text = format!("Invalid FEN: {err}"),
                    }
                }
                if ui.button("Sync Current FEN").clicked() {
                    self.fen_input = self.board.to_fen();
                    self.status_text = "Current FEN copied into input".to_string();
                }
            });
            ui.horizontal(|ui| {
                ui.label("Hash MB");
                ui.add(egui::Slider::new(&mut self.hash_mb, 1..=256));
                ui.label("Overhead ms");
                ui.add(egui::Slider::new(&mut self.move_overhead_ms, 0..=1000));
                ui.label("Max Depth");
                ui.add(egui::Slider::new(&mut self.max_depth, 1..=32));
                if ui.button("Apply Engine Settings").clicked() {
                    self.sync_engine_options();
                    self.status_text = "Engine settings applied".to_string();
                }
            });
        });

        egui::SidePanel::right("analysis").min_width(300.0).show(ctx, |ui| {
            ui.heading("Analysis");
            ui.label(&self.status_text);
            ui.label(format!("Viewing ply {}/{}", self.current_ply, self.moves.len()));
            ui.separator();

            ui.horizontal(|ui| {
                if ui.button("|<").clicked() {
                    self.go_to_ply(0);
                }
                if ui.button("<").clicked() {
                    self.go_to_ply(self.current_ply.saturating_sub(1));
                }
                if ui.button(">").clicked() {
                    self.go_to_ply((self.current_ply + 1).min(self.moves.len()));
                }
                if ui.button(">|").clicked() {
                    self.go_to_ply(self.moves.len());
                }
            });

            ui.separator();
            if let Some(result) = &self.latest_result {
                ui.label(format!("Depth: {}", result.depth));
                ui.label(format!("Score: {} cp", result.score_cp));
                ui.label(format!("Nodes: {}", result.nodes));
                ui.label(format!("Time: {} ms", result.elapsed.as_millis()));
                ui.label(format!(
                    "PV: {}",
                    result
                        .pv
                        .iter()
                        .map(|mv| mv.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                ));
                if let Some(best) = result.best_move {
                    ui.label(format!("Best move: {best}"));
                }
            } else {
                ui.label("No search yet");
            }

            ui.separator();
            ui.heading("Move List");
            let mut requested_ply = None;
            ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
                for (index, label) in self.move_labels.iter().enumerate() {
                    let move_number = index / 2 + 1;
                    let prefix = if index % 2 == 0 {
                        format!("{move_number}. ")
                    } else {
                        String::new()
                    };
                    let selected = self.current_ply == index + 1;
                    if ui
                        .selectable_label(selected, format!("{prefix}{label}"))
                        .clicked()
                    {
                        requested_ply = Some(index + 1);
                    }
                }
            });
            if let Some(ply) = requested_ply {
                self.go_to_ply(ply);
            }

            ui.separator();
            ui.heading("PGN");
            ui.horizontal(|ui| {
                if ui.button("Export PGN").clicked() {
                    self.pgn_text = export_pgn(&self.initial_board, &self.move_labels);
                    self.status_text = "Exported PGN to text area".to_string();
                }
                if ui.button("Import PGN").clicked() {
                    match import_pgn(&self.initial_board, &self.pgn_text) {
                        Ok((moves, labels)) => {
                            self.moves = moves;
                            self.move_labels = labels;
                            self.go_to_ply(self.moves.len());
                            self.status_text = "Imported PGN".to_string();
                        }
                        Err(err) => self.status_text = format!("PGN import failed: {err}"),
                    }
                }
            });
            ui.add(
                egui::TextEdit::multiline(&mut self.pgn_text)
                    .desired_rows(10)
                    .hint_text("Paste PGN or use Export PGN"),
            );

            ui.separator();
            let mut status_board = self.board.clone();
            ui.label(match status_board.game_status() {
                GameStatus::Ongoing => "Game status: ongoing".to_string(),
                GameStatus::Checkmate(color) => format!("Checkmate: {:?} to move is mated", color),
                GameStatus::Stalemate => "Game status: stalemate".to_string(),
                GameStatus::DrawByRepetition => "Game status: draw by repetition".to_string(),
                GameStatus::DrawByFiftyMoveRule => "Game status: draw by fifty-move rule".to_string(),
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("Rusty Fish Board");
            });
            ui.add_space(8.0);
            egui::Grid::new("board_grid")
                .spacing(Vec2::splat(2.0))
                .show(ui, |ui| {
                    for rank in (0..8).rev() {
                        for file in 0..8 {
                            let square = Square::from_file_rank(file, rank).expect("board square");
                            let piece_text = self
                                .board
                                .piece_at(square)
                                .map(piece_unicode)
                                .map(|ch| ch.to_string())
                                .unwrap_or_else(|| " ".to_string());
                            let is_light = (file + rank) % 2 == 0;
                            let mut fill = if is_light {
                                Color32::from_rgb(238, 238, 210)
                            } else {
                                Color32::from_rgb(118, 150, 86)
                            };
                            if self.highlighted_targets.contains(&square) {
                                fill = Color32::from_rgb(220, 196, 84);
                            }
                            let mut button = egui::Button::new(
                                RichText::new(piece_text).size(28.0).color(Color32::BLACK),
                            )
                            .min_size(Vec2::splat(48.0))
                            .fill(fill);
                            if self.selected_square == Some(square) {
                                button = button.stroke(egui::Stroke::new(2.0, Color32::YELLOW));
                            }
                            if ui.add(button).clicked() {
                                self.on_square_clicked(square);
                            }
                        }
                        ui.end_row();
                    }
                });
        });
    }
}

impl RustyFishApp {
    fn reset_to_board(&mut self, board: Board, status: &str) {
        self.initial_board = board.clone();
        self.board = board.clone();
        self.moves.clear();
        self.move_labels.clear();
        self.current_ply = 0;
        self.selected_square = None;
        self.highlighted_targets.clear();
        self.fen_input = board.to_fen();
        self.latest_result = None;
        self.pgn_text.clear();
        self.status_text = status.to_string();
        self.engine_service.clear_pending();
    }

    fn go_to_ply(&mut self, ply: usize) {
        self.current_ply = ply.min(self.moves.len());
        self.board = replay_board(&self.initial_board, &self.moves[..self.current_ply]);
        self.selected_square = None;
        self.highlighted_targets.clear();
        self.fen_input = self.board.to_fen();
        self.latest_result = None;
    }

    fn on_square_clicked(&mut self, square: Square) {
        if let Some(selected) = self.selected_square {
            if selected == square {
                self.selected_square = None;
                self.highlighted_targets.clear();
                return;
            }

            let candidates = self
                .board
                .generate_legal_moves()
                .into_iter()
                .filter(|mv| mv.from == selected && mv.to == square)
                .collect::<Vec<_>>();
            if candidates.is_empty() {
                if self
                    .board
                    .piece_at(square)
                    .is_some_and(|piece| piece.color == self.board.side_to_move)
                {
                    self.select_square(square);
                } else {
                    self.selected_square = None;
                    self.highlighted_targets.clear();
                }
                return;
            }

            let chosen = candidates
                .iter()
                .find(|mv| mv.promotion == Some(PieceKind::Queen))
                .copied()
                .unwrap_or(candidates[0]);
            self.apply_move(chosen);
            self.selected_square = None;
            self.highlighted_targets.clear();
        } else if self
            .board
            .piece_at(square)
            .is_some_and(|piece| piece.color == self.board.side_to_move)
        {
            self.select_square(square);
        }
    }

    fn select_square(&mut self, square: Square) {
        self.selected_square = Some(square);
        self.highlighted_targets = self
            .board
            .generate_legal_moves()
            .into_iter()
            .filter(|mv| mv.from == square)
            .map(|mv| mv.to)
            .collect();
    }

    fn apply_move(&mut self, mv: ChessMove) {
        if self.current_ply < self.moves.len() {
            self.moves.truncate(self.current_ply);
            self.move_labels.truncate(self.current_ply);
        }

        let san = move_to_san(&mut self.board.clone(), mv).unwrap_or_else(|_| mv.to_string());
        match self.board.make_move(mv) {
            Ok(_) => {
                self.moves.push(mv);
                self.move_labels.push(san);
                self.current_ply = self.moves.len();
                self.fen_input = self.board.to_fen();
                self.latest_result = None;
                self.status_text = format!("Played {mv}");
            }
            Err(err) => {
                self.status_text = format!("Move failed: {err}");
            }
        }
    }

    fn start_search(&mut self, apply_engine_move: bool) {
        self.apply_engine_move_on_result = apply_engine_move;
        self.status_text = if apply_engine_move {
            "Searching for engine move...".to_string()
        } else {
            "Analyzing position...".to_string()
        };
        let _ = self.engine_service.request(EngineRequest::Analyze {
            board: self.board.clone(),
            limits: SearchLimits {
                depth: Some(self.analysis_depth),
                movetime: Some(Duration::from_millis(1_500)),
                ..SearchLimits::default()
            },
        });
        self.engine_service.is_searching = true;
    }

    fn poll_search(&mut self) {
        loop {
            match self.engine_service.try_recv() {
                Ok(EngineEvent::AnalysisComplete(result)) => {
                    self.engine_service.is_searching = false;
                    if self.apply_engine_move_on_result {
                        if let Some(best_move) = result.best_move {
                            self.apply_move(best_move);
                            self.status_text = format!("Engine played {best_move}");
                        }
                    } else {
                        self.status_text = "Analysis complete".to_string();
                    }
                    self.latest_result = Some(result);
                    self.apply_engine_move_on_result = false;
                }
                Ok(EngineEvent::OptionsApplied(options)) => {
                    self.hash_mb = options.hash_mb;
                    self.move_overhead_ms = options.move_overhead.as_millis() as u64;
                    self.max_depth = options.max_depth;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.engine_service.is_searching = false;
                    self.status_text = "Engine service disconnected".to_string();
                    break;
                }
            }
        }
    }

    fn sync_engine_options(&mut self) {
        let options = SearchOptions {
            max_depth: self.max_depth,
            hash_mb: self.hash_mb,
            move_overhead: Duration::from_millis(self.move_overhead_ms),
            ..SearchOptions::default()
        };
        let _ = self
            .engine_service
            .request(EngineRequest::SetOptions(options));
    }
}

struct EngineService {
    command_tx: Sender<EngineRequest>,
    event_rx: Receiver<EngineEvent>,
    is_searching: bool,
}

impl EngineService {
    fn new() -> Self {
        let (command_tx, command_rx) = mpsc::channel::<EngineRequest>();
        let (event_tx, event_rx) = mpsc::channel::<EngineEvent>();
        thread::spawn(move || {
            let mut searcher = Searcher::default();
            while let Ok(command) = command_rx.recv() {
                match command {
                    EngineRequest::SetOptions(options) => {
                        searcher.set_options(options.clone());
                        let _ = event_tx.send(EngineEvent::OptionsApplied(options));
                    }
                    EngineRequest::Analyze { board, limits } => {
                        let result = searcher.search(&board, limits);
                        let _ = event_tx.send(EngineEvent::AnalysisComplete(result));
                    }
                }
            }
        });
        Self {
            command_tx,
            event_rx,
            is_searching: false,
        }
    }

    fn request(&self, request: EngineRequest) -> Result<(), mpsc::SendError<EngineRequest>> {
        self.command_tx.send(request)
    }

    fn try_recv(&self) -> Result<EngineEvent, TryRecvError> {
        self.event_rx.try_recv()
    }

    fn clear_pending(&mut self) {
        while self.event_rx.try_recv().is_ok() {}
        self.is_searching = false;
    }
}

enum EngineRequest {
    SetOptions(SearchOptions),
    Analyze { board: Board, limits: SearchLimits },
}

enum EngineEvent {
    OptionsApplied(SearchOptions),
    AnalysisComplete(SearchResult),
}

fn replay_board(initial_board: &Board, moves: &[ChessMove]) -> Board {
    let mut board = initial_board.clone();
    for mv in moves {
        board.make_move(*mv).expect("stored move should remain legal");
    }
    board
}

fn export_pgn(initial_board: &Board, move_labels: &[String]) -> String {
    let mut body = String::new();
    for (index, san) in move_labels.iter().enumerate() {
        if index % 2 == 0 {
            if !body.is_empty() {
                body.push(' ');
            }
            body.push_str(&(index / 2 + 1).to_string());
            body.push('.');
        }
        body.push(' ');
        body.push_str(san);
    }

    format!(
        "[Event \"Rusty Fish Analysis\"]\n[Site \"Local\"]\n[FEN \"{}\"]\n[Result \"*\"]\n\n{} *\n",
        initial_board.to_fen(),
        body.trim()
    )
}

fn import_pgn(initial_board: &Board, text: &str) -> Result<(Vec<ChessMove>, Vec<String>), String> {
    let mut board = initial_board.clone();
    let mut moves = Vec::new();
    let mut labels = Vec::new();

    for token in tokenize_pgn(text) {
        let legal_moves = board.generate_legal_moves();
        let found = legal_moves.into_iter().find(|mv| {
            normalize_move_token(&mv.to_string()) == normalize_move_token(&token)
                || move_to_san(&mut board.clone(), *mv)
                    .map(|san| normalize_move_token(&san) == normalize_move_token(&token))
                    .unwrap_or(false)
        });

        let Some(mv) = found else {
            return Err(format!("could not match PGN token `{token}`"));
        };
        let san = move_to_san(&mut board.clone(), mv)?;
        board.make_move(mv)?;
        moves.push(mv);
        labels.push(san);
    }

    Ok((moves, labels))
}

fn tokenize_pgn(text: &str) -> Vec<String> {
    let mut cleaned = String::new();
    let mut in_comment = false;
    for ch in text.chars() {
        match ch {
            '{' => in_comment = true,
            '}' => in_comment = false,
            _ if !in_comment => cleaned.push(ch),
            _ => {}
        }
    }

    cleaned
        .lines()
        .filter(|line| !line.trim_start().starts_with('['))
        .flat_map(|line| line.split_whitespace())
        .filter_map(|token| {
            let token = token.trim();
            if token.is_empty()
                || token == "*"
                || token == "1-0"
                || token == "0-1"
                || token == "1/2-1/2"
                || token.chars().all(|ch| ch.is_ascii_digit() || ch == '.')
            {
                return None;
            }
            Some(
                token
                    .trim_matches(|ch: char| ch == '!' || ch == '?')
                    .to_string(),
            )
        })
        .collect()
}

fn normalize_move_token(token: &str) -> String {
    token
        .trim()
        .trim_end_matches(['+', '#', '!', '?'])
        .replace('0', "O")
        .to_ascii_lowercase()
}

fn move_to_san(board: &mut Board, mv: ChessMove) -> Result<String, String> {
    let piece = board
        .piece_at(mv.from)
        .ok_or_else(|| format!("no piece on {}", mv.from))?;

    if piece.kind == PieceKind::King {
        match (mv.from.to_string().as_str(), mv.to.to_string().as_str()) {
            ("e1", "g1") | ("e8", "g8") => return annotate_san(board, mv, "O-O".to_string()),
            ("e1", "c1") | ("e8", "c8") => return annotate_san(board, mv, "O-O-O".to_string()),
            _ => {}
        }
    }

    let legal_moves = board.generate_legal_moves();
    let is_capture = board.piece_at(mv.to).is_some()
        || (piece.kind == PieceKind::Pawn && board.en_passant() == Some(mv.to));
    let mut san = String::new();

    if piece.kind != PieceKind::Pawn {
        san.push(piece_letter(piece.kind));

        let siblings = legal_moves
            .iter()
            .filter(|candidate| {
                **candidate != mv
                    && candidate.to == mv.to
                    && board.piece_at(candidate.from).is_some_and(|other| other.kind == piece.kind)
            })
            .copied()
            .collect::<Vec<_>>();

        if !siblings.is_empty() {
            let same_file = siblings.iter().any(|other| other.from.file() == mv.from.file());
            let same_rank = siblings.iter().any(|other| other.from.rank() == mv.from.rank());
            if !same_file {
                san.push((b'a' + mv.from.file()) as char);
            } else if !same_rank {
                san.push((b'1' + mv.from.rank()) as char);
            } else {
                san.push((b'a' + mv.from.file()) as char);
                san.push((b'1' + mv.from.rank()) as char);
            }
        }
    } else if is_capture {
        san.push((b'a' + mv.from.file()) as char);
    }

    if is_capture {
        san.push('x');
    }
    san.push_str(&mv.to.to_string());

    if let Some(promotion) = mv.promotion {
        san.push('=');
        san.push(piece_letter(promotion));
    }

    annotate_san(board, mv, san)
}

fn annotate_san(board: &mut Board, mv: ChessMove, mut san: String) -> Result<String, String> {
    let undo = board.make_move(mv)?;
    let mut clone = board.clone();
    match clone.game_status() {
        GameStatus::Checkmate(_) => san.push('#'),
        _ if board.in_check(board.side_to_move) => san.push('+'),
        _ => {}
    }
    board.unmake_move(mv, undo);
    Ok(san)
}

fn piece_letter(kind: PieceKind) -> char {
    match kind {
        PieceKind::Pawn => ' ',
        PieceKind::Knight => 'N',
        PieceKind::Bishop => 'B',
        PieceKind::Rook => 'R',
        PieceKind::Queen => 'Q',
        PieceKind::King => 'K',
    }
}

#[cfg(test)]
mod tests {
    use engine_core::Board;

    use super::{export_pgn, import_pgn, move_to_san, tokenize_pgn};

    #[test]
    fn san_for_basic_opening_move() {
        let mut board = Board::startpos();
        let mv = board.parse_uci_move("e2e4").unwrap();
        assert_eq!(move_to_san(&mut board, mv).unwrap(), "e4");
    }

    #[test]
    fn export_and_import_round_trip() {
        let initial = Board::startpos();
        let text = export_pgn(&initial, &["e4".to_string(), "e5".to_string(), "Nf3".to_string()]);
        let (moves, labels) = import_pgn(&initial, &text).unwrap();
        assert_eq!(moves.len(), 3);
        assert_eq!(labels[0], "e4");
        assert_eq!(labels[2], "Nf3");
    }

    #[test]
    fn tokenizer_skips_tags_and_results() {
        let tokens = tokenize_pgn("[Event \"x\"]\n1. e4 e5 2. Nf3 *");
        assert_eq!(tokens, vec!["e4", "e5", "Nf3"]);
    }
}
