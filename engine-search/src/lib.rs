use std::collections::HashMap;
use std::time::{Duration, Instant};

use engine_core::{Board, ChessMove, Color, GameStatus, Piece, PieceKind};

const MATE_SCORE: i32 = 100_000;
const MAX_KILLER_PLY: usize = 128;
const ASPIRATION_WINDOW: i32 = 50;

fn tt_capacity_entries_for(hash_mb: usize) -> usize {
    let bytes = hash_mb.max(1) * 1024 * 1024;
    let approx_entry_size = 32usize;
    (bytes / approx_entry_size).max(1_024)
}

fn late_move_reduction(depth: u8, move_index: usize, is_quiet: bool) -> u8 {
    if !is_quiet || depth < 3 || move_index < 3 {
        return 0;
    }
    1 + u8::from(depth >= 7 && move_index >= 8)
}

#[derive(Clone, Debug, Default)]
pub struct SearchLimits {
    pub depth: Option<u8>,
    pub movetime: Option<Duration>,
    pub clock: Option<ClockControl>,
    pub infinite: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct ClockControl {
    pub white_time: Duration,
    pub black_time: Duration,
    pub white_increment: Duration,
    pub black_increment: Duration,
    pub moves_to_go: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct SearchOptions {
    pub max_depth: u8,
    pub hash_mb: usize,
    pub move_overhead: Duration,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            max_depth: 16,
            hash_mb: 16,
            move_overhead: Duration::from_millis(25),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SearchInfo {
    pub depth: u8,
    pub score_cp: i32,
    pub nodes: u64,
    pub elapsed: Duration,
    pub pv: Vec<ChessMove>,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub best_move: Option<ChessMove>,
    pub depth: u8,
    pub score_cp: i32,
    pub nodes: u64,
    pub elapsed: Duration,
    pub pv: Vec<ChessMove>,
}

pub struct Searcher {
    nodes: u64,
    start: Instant,
    deadline: Option<Instant>,
    stopped: bool,
    tt: TranspositionTable,
    killer_moves: Vec<[Option<ChessMove>; 2]>,
    history: HashMap<ChessMove, i32>,
    options: SearchOptions,
}

impl Default for Searcher {
    fn default() -> Self {
        Self {
            nodes: 0,
            start: Instant::now(),
            deadline: None,
            stopped: false,
            tt: TranspositionTable::new(tt_capacity_entries_for(SearchOptions::default().hash_mb)),
            killer_moves: vec![[None, None]; MAX_KILLER_PLY],
            history: HashMap::new(),
            options: SearchOptions::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy, Debug)]
struct TranspositionEntry {
    depth: u8,
    score: i32,
    bound: Bound,
    best_move: Option<ChessMove>,
}

#[derive(Clone, Copy, Debug)]
struct TranspositionSlot {
    key: u64,
    generation: u8,
    entry: TranspositionEntry,
}

#[derive(Debug)]
struct TranspositionTable {
    slots: Vec<Option<TranspositionSlot>>,
    generation: u8,
}

impl TranspositionTable {
    fn new(capacity: usize) -> Self {
        Self {
            slots: vec![None; capacity.max(1)],
            generation: 0,
        }
    }

    fn begin_search(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    fn resize(&mut self, capacity: usize) {
        *self = Self::new(capacity);
    }

    fn get(&self, key: u64) -> Option<&TranspositionEntry> {
        self.slots[self.index(key)]
            .as_ref()
            .filter(|slot| slot.key == key)
            .map(|slot| &slot.entry)
    }

    fn store(&mut self, key: u64, entry: TranspositionEntry) {
        let index = self.index(key);
        let replacement = TranspositionSlot {
            key,
            generation: self.generation,
            entry,
        };

        match self.slots[index] {
            None => self.slots[index] = Some(replacement),
            Some(current) if current.key == key => {
                if entry.depth >= current.entry.depth || entry.bound == Bound::Exact {
                    self.slots[index] = Some(replacement);
                }
            }
            Some(current)
                if current.generation != self.generation || entry.depth > current.entry.depth =>
            {
                self.slots[index] = Some(replacement);
            }
            Some(_) => {}
        }
    }

    #[cfg(test)]
    fn contains_key(&self, key: u64) -> bool {
        self.get(key).is_some()
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.slots.iter().all(Option::is_none)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.slots.iter().flatten().count()
    }

    #[cfg(test)]
    fn values(&self) -> impl Iterator<Item = &TranspositionEntry> {
        self.slots.iter().flatten().map(|slot| &slot.entry)
    }

    fn index(&self, key: u64) -> usize {
        (key as usize) % self.slots.len()
    }
}

impl Searcher {
    pub fn options(&self) -> &SearchOptions {
        &self.options
    }

    pub fn set_options(&mut self, options: SearchOptions) {
        let capacity_changed = self.options.hash_mb != options.hash_mb;
        self.options = options;
        if capacity_changed {
            self.tt.resize(self.tt_capacity_entries());
        }
    }

    pub fn search(&mut self, board: &Board, limits: SearchLimits) -> SearchResult {
        self.search_with_callback(board, limits, |_info| {})
    }

    pub fn search_with_callback<F>(
        &mut self,
        board: &Board,
        limits: SearchLimits,
        mut callback: F,
    ) -> SearchResult
    where
        F: FnMut(&SearchInfo),
    {
        self.nodes = 0;
        self.start = Instant::now();
        self.deadline = self
            .time_budget(board.side_to_move, &limits)
            .map(|limit| self.start + limit);
        self.stopped = false;
        self.tt.begin_search();
        self.killer_moves.fill([None, None]);
        self.history.clear();

        let max_depth = limits
            .depth
            .unwrap_or(self.options.max_depth)
            .max(1)
            .min(self.options.max_depth);
        let mut best_move = None;
        let mut best_score = 0;
        let mut best_pv = Vec::new();
        let mut reached_depth = 0;

        for depth in 1..=max_depth {
            let mut clone = board.clone();
            let (score, pv) = if depth == 1 {
                self.negamax_root(&mut clone, depth, -MATE_SCORE, MATE_SCORE)
            } else {
                self.aspiration_search(&mut clone, depth, best_score)
            };
            if self.stopped {
                break;
            }
            reached_depth = depth;
            best_score = score;
            best_move = pv.first().copied();
            if let Some(root_best_move) = best_move {
                self.store_tt(
                    board.position_hash(),
                    TranspositionEntry {
                        depth,
                        score: best_score,
                        bound: Bound::Exact,
                        best_move: Some(root_best_move),
                    },
                );
            }
            best_pv = pv;
            let info = SearchInfo {
                depth,
                score_cp: best_score,
                nodes: self.nodes,
                elapsed: self.start.elapsed(),
                pv: best_pv.clone(),
            };
            callback(&info);
            if best_score.abs() >= MATE_SCORE - 128 {
                break;
            }
        }

        SearchResult {
            best_move,
            depth: reached_depth,
            score_cp: best_score,
            nodes: self.nodes,
            elapsed: self.start.elapsed(),
            pv: best_pv,
        }
    }

    fn aspiration_search(
        &mut self,
        board: &mut Board,
        depth: u8,
        previous_score: i32,
    ) -> (i32, Vec<ChessMove>) {
        let mut window = ASPIRATION_WINDOW;
        let mut alpha = (previous_score - window).max(-MATE_SCORE);
        let mut beta = (previous_score + window).min(MATE_SCORE);

        loop {
            let (score, pv) = self.negamax_root(board, depth, alpha, beta);
            if self.stopped {
                return (score, pv);
            }
            if score <= alpha {
                window *= 2;
                alpha = (previous_score - window).max(-MATE_SCORE);
                beta = (alpha + window * 2).min(MATE_SCORE);
                continue;
            }
            if score >= beta {
                window *= 2;
                alpha = (beta - window * 2).max(-MATE_SCORE);
                beta = (previous_score + window).min(MATE_SCORE);
                continue;
            }
            return (score, pv);
        }
    }

    fn negamax_root(
        &mut self,
        board: &mut Board,
        depth: u8,
        mut alpha: i32,
        beta: i32,
    ) -> (i32, Vec<ChessMove>) {
        let tt_move = self
            .tt
            .get(board.position_hash())
            .and_then(|entry| entry.best_move);
        let mut moves = board.generate_legal_moves();
        self.order_moves(board, &mut moves, 0, tt_move);
        if moves.is_empty() {
            return (self.evaluate_terminal(board, 0), Vec::new());
        }

        let mut best_score = -MATE_SCORE;
        let mut best_line = Vec::new();
        let original_alpha = alpha;
        for mv in moves {
            if self.should_stop() {
                break;
            }
            let undo = board.make_move(mv).expect("generated move must be legal");
            let (score, mut line) = self.negamax(board, depth.saturating_sub(1), 1, -beta, -alpha);
            let score = -score;
            board.unmake_move(mv, undo);

            if score > best_score {
                best_score = score;
                best_line.clear();
                best_line.push(mv);
                best_line.append(&mut line);
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                self.record_cutoff(0, mv, depth);
                break;
            }
        }
        let bound = if best_score <= original_alpha {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.store_tt(
            board.position_hash(),
            TranspositionEntry {
                depth,
                score: best_score,
                bound,
                best_move: best_line.first().copied(),
            },
        );
        (best_score, best_line)
    }

    fn negamax(
        &mut self,
        board: &mut Board,
        depth: u8,
        ply: i32,
        mut alpha: i32,
        beta: i32,
    ) -> (i32, Vec<ChessMove>) {
        if self.should_stop() {
            return (self.evaluate(board), Vec::new());
        }
        self.nodes += 1;
        let original_alpha = alpha;
        let tt_key = board.position_hash();
        let in_check = board.in_check(board.side_to_move);

        if let Some(entry) = self.tt.get(tt_key).copied()
            && entry.depth >= depth
        {
            match entry.bound {
                Bound::Exact => return (entry.score, vec![]),
                Bound::Lower => alpha = alpha.max(entry.score),
                Bound::Upper => {}
            }
            if alpha >= beta {
                return (entry.score, vec![]);
            }
        }

        match board.game_status() {
            GameStatus::Ongoing => {}
            _ => return (self.evaluate_terminal(board, ply), Vec::new()),
        }

        if depth == 0 {
            return (self.quiescence(board, alpha, beta), Vec::new());
        }

        if !in_check && depth >= 3 && self.has_non_pawn_material(board, board.side_to_move) {
            let null_score = self.try_null_move(board, depth, ply, beta);
            if null_score >= beta {
                return (null_score, Vec::new());
            }
        }

        let tt_move = self.tt.get(tt_key).and_then(|entry| entry.best_move);
        let mut moves = board.generate_legal_moves();
        self.order_moves(board, &mut moves, ply as usize, tt_move);
        if moves.is_empty() {
            return (self.evaluate_terminal(board, ply), Vec::new());
        }

        let mut best_score = -MATE_SCORE;
        let mut best_line = Vec::new();
        for (move_index, mv) in moves.into_iter().enumerate() {
            let is_quiet = self.is_quiet_move(board, mv);
            let undo = board.make_move(mv).expect("generated move must be legal");
            let extension = u8::from(board.in_check(board.side_to_move));
            let next_depth = depth.saturating_sub(1) + extension.min(1);
            let reduction = late_move_reduction(depth, move_index, is_quiet && extension == 0);
            let (child_score, mut line) = self.negamax(
                board,
                next_depth.saturating_sub(reduction),
                ply + 1,
                -beta,
                -alpha,
            );
            let mut score = -child_score;
            if reduction > 0 && score > alpha && !self.stopped {
                let (full_score, full_line) =
                    self.negamax(board, next_depth, ply + 1, -beta, -alpha);
                score = -full_score;
                line = full_line;
            }
            board.unmake_move(mv, undo);

            if score > best_score {
                best_score = score;
                best_line.clear();
                best_line.push(mv);
                best_line.append(&mut line);
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                self.record_cutoff(ply as usize, mv, depth);
                break;
            }
            if self.should_stop() {
                break;
            }
        }
        let bound = if best_score <= original_alpha {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        self.store_tt(
            tt_key,
            TranspositionEntry {
                depth,
                score: best_score,
                bound,
                best_move: best_line.first().copied(),
            },
        );
        (best_score, best_line)
    }

    fn quiescence(&mut self, board: &mut Board, mut alpha: i32, beta: i32) -> i32 {
        if self.should_stop() {
            return self.evaluate(board);
        }
        self.nodes += 1;

        let in_check = board.in_check(board.side_to_move);
        if in_check {
            let mut evasions = board.generate_legal_moves();
            let tt_move = self
                .tt
                .get(board.position_hash())
                .and_then(|entry| entry.best_move);
            self.order_moves(board, &mut evasions, 0, tt_move);
            if evasions.is_empty() {
                return self.evaluate_terminal(board, 0);
            }
            for mv in evasions {
                let undo = board.make_move(mv).expect("generated move must be legal");
                let score = -self.quiescence(board, -beta, -alpha);
                board.unmake_move(mv, undo);
                if score >= beta {
                    return beta;
                }
                alpha = alpha.max(score);
            }
            return alpha;
        }

        let stand_pat = self.evaluate(board);
        if stand_pat >= beta {
            return beta;
        }
        alpha = alpha.max(stand_pat);

        let mut moves = board.generate_capture_moves();
        self.order_moves(board, &mut moves, 0, None);
        for mv in moves {
            if !self.is_promising_quiescence_capture(board, mv, stand_pat, alpha) {
                continue;
            }
            let undo = board.make_move(mv).expect("generated move must be legal");
            let score = -self.quiescence(board, -beta, -alpha);
            board.unmake_move(mv, undo);

            if score >= beta {
                return beta;
            }
            alpha = alpha.max(score);
        }
        alpha
    }

    fn evaluate_terminal(&self, board: &mut Board, ply: i32) -> i32 {
        match board.game_status() {
            GameStatus::Checkmate(color_to_move) => {
                if color_to_move == board.side_to_move {
                    -MATE_SCORE + ply
                } else {
                    MATE_SCORE - ply
                }
            }
            GameStatus::Stalemate
            | GameStatus::DrawByFiftyMoveRule
            | GameStatus::DrawByRepetition => 0,
            GameStatus::Ongoing => self.evaluate(board),
        }
    }

    fn evaluate(&self, board: &Board) -> i32 {
        evaluate_position(board)
    }

    fn order_moves(
        &self,
        board: &Board,
        moves: &mut [ChessMove],
        ply: usize,
        tt_move: Option<ChessMove>,
    ) {
        moves.sort_by_cached_key(|mv| -self.move_order_score(board, *mv, ply, tt_move));
    }

    fn move_order_score(
        &self,
        board: &Board,
        mv: ChessMove,
        ply: usize,
        tt_move: Option<ChessMove>,
    ) -> i32 {
        if tt_move == Some(mv) {
            return 2_000_000;
        }

        let mut score = 0;
        if let Some(victim) = board.piece_at(mv.to) {
            let attacker = board.piece_at(mv.from).map(piece_value).unwrap_or_default();
            score += 1_000_000 + piece_value(victim) * 16 - attacker;
        }
        if board.en_passant() == Some(mv.to)
            && board
                .piece_at(mv.from)
                .is_some_and(|piece| piece.kind == PieceKind::Pawn)
        {
            score += 1_000_000 + piece_kind_value(PieceKind::Pawn) * 16
                - piece_kind_value(PieceKind::Pawn);
        }
        if let Some(promotion) = mv.promotion {
            score += 800_000 + piece_kind_value(promotion);
        }

        if let Some(killers) = self.killer_moves.get(ply) {
            if killers[0] == Some(mv) {
                score += 700_000;
            } else if killers[1] == Some(mv) {
                score += 650_000;
            }
        }

        score + self.history.get(&mv).copied().unwrap_or_default()
    }

    fn record_cutoff(&mut self, ply: usize, mv: ChessMove, depth: u8) {
        if ply < self.killer_moves.len() {
            let entry = &mut self.killer_moves[ply];
            if entry[0] != Some(mv) {
                entry[1] = entry[0];
                entry[0] = Some(mv);
            }
        }
        *self.history.entry(mv).or_insert(0) += i32::from(depth) * i32::from(depth) * 16;
    }

    fn store_tt(&mut self, key: u64, entry: TranspositionEntry) {
        self.tt.store(key, entry);
    }

    fn try_null_move(&mut self, board: &mut Board, depth: u8, ply: i32, beta: i32) -> i32 {
        let mut null_board = board.clone();
        null_board.make_null_move();
        -self
            .negamax(
                &mut null_board,
                depth.saturating_sub(3),
                ply + 1,
                -beta,
                -beta + 1,
            )
            .0
    }

    fn has_non_pawn_material(&self, board: &Board, color: Color) -> bool {
        for idx in 0..64 {
            if let Some(piece) = board.piece_at(engine_core::Square(idx))
                && piece.color == color
                && !matches!(piece.kind, PieceKind::Pawn | PieceKind::King)
            {
                return true;
            }
        }
        false
    }

    fn is_promising_quiescence_capture(
        &self,
        board: &Board,
        mv: ChessMove,
        stand_pat: i32,
        alpha: i32,
    ) -> bool {
        if mv.promotion.is_some() {
            return true;
        }
        let captured_value = board.piece_at(mv.to).map(piece_value).unwrap_or_else(|| {
            if board.en_passant() == Some(mv.to) {
                piece_kind_value(PieceKind::Pawn)
            } else {
                0
            }
        });
        stand_pat + captured_value + 75 >= alpha
    }

    fn is_quiet_move(&self, board: &Board, mv: ChessMove) -> bool {
        mv.promotion.is_none()
            && board.piece_at(mv.to).is_none()
            && !(board.en_passant() == Some(mv.to)
                && board
                    .piece_at(mv.from)
                    .is_some_and(|piece| piece.kind == PieceKind::Pawn))
    }

    fn tt_capacity_entries(&self) -> usize {
        tt_capacity_entries_for(self.options.hash_mb)
    }

    fn time_budget(&self, side_to_move: Color, limits: &SearchLimits) -> Option<Duration> {
        if limits.infinite {
            return None;
        }
        if let Some(movetime) = limits.movetime {
            return Some(movetime);
        }
        let clock = limits.clock?;
        let (time_left, increment) = match side_to_move {
            Color::White => (clock.white_time, clock.white_increment),
            Color::Black => (clock.black_time, clock.black_increment),
        };

        let overhead = self.options.move_overhead.min(time_left);
        let effective_time = time_left.saturating_sub(overhead);
        let moves_to_go = clock.moves_to_go.unwrap_or(30).max(1);
        let slice = effective_time / moves_to_go;
        let base = slice.max(Duration::from_millis(10));
        let bonus = increment / 2;
        let cap = effective_time
            .checked_div(2)
            .unwrap_or(Duration::from_millis(10))
            .max(Duration::from_millis(10));
        Some((base + bonus).min(cap))
    }

    fn should_stop(&mut self) -> bool {
        if self.stopped {
            return true;
        }
        if let Some(deadline) = self.deadline
            && Instant::now() >= deadline
        {
            self.stopped = true;
        }
        self.stopped
    }
}

fn piece_value(piece: Piece) -> i32 {
    piece_kind_value(piece.kind)
}

fn piece_kind_value(kind: PieceKind) -> i32 {
    match kind {
        PieceKind::Pawn => 100,
        PieceKind::Knight => 320,
        PieceKind::Bishop => 330,
        PieceKind::Rook => 500,
        PieceKind::Queen => 900,
        PieceKind::King => 0,
    }
}

fn piece_square_bonus(piece: Piece, idx: u8) -> i32 {
    let rank = idx / 8;
    let file = idx % 8;
    let centered_file = (3_i32 - file as i32).abs().min((4_i32 - file as i32).abs());
    let centered_rank = match piece.color {
        Color::White => (3_i32 - rank as i32).abs().min((4_i32 - rank as i32).abs()),
        Color::Black => (3_i32 - (7 - rank) as i32)
            .abs()
            .min((4_i32 - (7 - rank) as i32).abs()),
    };
    let centrality = 6 - (centered_file + centered_rank);
    match piece.kind {
        PieceKind::Pawn => {
            centrality * 2 + rank as i32 * if piece.color == Color::White { 3 } else { -3 }
        }
        PieceKind::Knight => centrality * 8,
        PieceKind::Bishop => centrality * 5,
        PieceKind::Rook => centrality * 2,
        PieceKind::Queen => centrality * 2,
        PieceKind::King => -centrality * 4,
    }
}

#[derive(Default, Clone, Copy)]
struct EvalFeatures {
    white_score: i32,
    black_score: i32,
}

impl EvalFeatures {
    fn add(&mut self, color: Color, value: i32) {
        match color {
            Color::White => self.white_score += value,
            Color::Black => self.black_score += value,
        }
    }

    fn net(self, side_to_move: Color) -> i32 {
        let score = self.white_score - self.black_score;
        match side_to_move {
            Color::White => score,
            Color::Black => -score,
        }
    }
}

fn evaluate_position(board: &Board) -> i32 {
    let mut features = EvalFeatures::default();
    let mut white_pawn_files = [0u8; 8];
    let mut black_pawn_files = [0u8; 8];
    let mut white_bishops = 0;
    let mut black_bishops = 0;

    for idx in 0..64 {
        let square = engine_core::Square(idx);
        let Some(piece) = board.piece_at(square) else {
            continue;
        };
        features.add(
            piece.color,
            piece_value(piece) + piece_square_bonus(piece, idx),
        );

        if piece.kind == PieceKind::Pawn {
            match piece.color {
                Color::White => white_pawn_files[square.file() as usize] += 1,
                Color::Black => black_pawn_files[square.file() as usize] += 1,
            }
        }
        if piece.kind == PieceKind::Bishop {
            match piece.color {
                Color::White => white_bishops += 1,
                Color::Black => black_bishops += 1,
            }
        }

        features.add(piece.color, activity_bonus(board, square, piece));
    }

    if white_bishops >= 2 {
        features.add(Color::White, 35);
    }
    if black_bishops >= 2 {
        features.add(Color::Black, 35);
    }

    for idx in 0..64 {
        let square = engine_core::Square(idx);
        let Some(piece) = board.piece_at(square) else {
            continue;
        };
        if piece.kind == PieceKind::Pawn {
            features.add(
                piece.color,
                pawn_structure_bonus(
                    board,
                    square,
                    piece.color,
                    &white_pawn_files,
                    &black_pawn_files,
                ),
            );
        }
        if piece.kind == PieceKind::Rook {
            features.add(
                piece.color,
                rook_file_bonus(square, piece.color, &white_pawn_files, &black_pawn_files),
            );
        }
    }

    features.add(Color::White, king_safety_bonus(board, Color::White));
    features.add(Color::Black, king_safety_bonus(board, Color::Black));
    features.net(board.side_to_move)
}

fn activity_bonus(board: &Board, square: engine_core::Square, piece: Piece) -> i32 {
    match piece.kind {
        PieceKind::Knight => count_knight_targets(board, square, piece.color) * 4,
        PieceKind::Bishop => {
            count_slider_targets(
                board,
                square,
                piece.color,
                &[(-1, -1), (-1, 1), (1, -1), (1, 1)],
            ) * 3
        }
        PieceKind::Rook => {
            count_slider_targets(
                board,
                square,
                piece.color,
                &[(-1, 0), (1, 0), (0, -1), (0, 1)],
            ) * 2
        }
        PieceKind::Queen => count_slider_targets(
            board,
            square,
            piece.color,
            &[
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
            ],
        ),
        _ => 0,
    }
}

fn count_knight_targets(board: &Board, square: engine_core::Square, color: Color) -> i32 {
    let mut count = 0;
    for (df, dr) in [
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ] {
        if let Some(target) = square.offset(df, dr)
            && board
                .piece_at(target)
                .is_none_or(|piece| piece.color != color)
        {
            count += 1;
        }
    }
    count
}

fn count_slider_targets(
    board: &Board,
    square: engine_core::Square,
    color: Color,
    directions: &[(i8, i8)],
) -> i32 {
    let mut count = 0;
    for &(df, dr) in directions {
        let mut current = square;
        while let Some(next) = current.offset(df, dr) {
            current = next;
            match board.piece_at(current) {
                Some(piece) if piece.color == color => break,
                Some(_) => {
                    count += 1;
                    break;
                }
                None => count += 1,
            }
        }
    }
    count
}

fn pawn_structure_bonus(
    board: &Board,
    square: engine_core::Square,
    color: Color,
    white_pawn_files: &[u8; 8],
    black_pawn_files: &[u8; 8],
) -> i32 {
    let own_files = match color {
        Color::White => white_pawn_files,
        Color::Black => black_pawn_files,
    };
    let enemy_files = match color {
        Color::White => black_pawn_files,
        Color::Black => white_pawn_files,
    };
    let file = square.file() as usize;
    let mut score = 0;

    if own_files[file] > 1 {
        score -= 14 * i32::from(own_files[file] - 1);
    }

    let left_support = file > 0 && own_files[file - 1] > 0;
    let right_support = file < 7 && own_files[file + 1] > 0;
    if !left_support && !right_support {
        score -= 18;
    }

    let files = file.saturating_sub(1)..=((file + 1).min(7));
    let is_passed = match color {
        Color::White => {
            let mut blocked = false;
            for enemy_file in files {
                for rank in (square.rank() + 1)..8 {
                    if board.piece_at(
                        engine_core::Square::from_file_rank(enemy_file as u8, rank).expect("valid"),
                    ) == Some(Piece {
                        color: Color::Black,
                        kind: PieceKind::Pawn,
                    }) {
                        blocked = true;
                        break;
                    }
                }
                if blocked {
                    break;
                }
            }
            !blocked
        }
        Color::Black => {
            let mut blocked = false;
            for enemy_file in file.saturating_sub(1)..=((file + 1).min(7)) {
                for rank in 0..square.rank() {
                    if board.piece_at(
                        engine_core::Square::from_file_rank(enemy_file as u8, rank).expect("valid"),
                    ) == Some(Piece {
                        color: Color::White,
                        kind: PieceKind::Pawn,
                    }) {
                        blocked = true;
                        break;
                    }
                }
                if blocked {
                    break;
                }
            }
            !blocked
        }
    };
    if is_passed {
        let advancement = match color {
            Color::White => square.rank() as i32,
            Color::Black => (7 - square.rank()) as i32,
        };
        score += 20 + advancement * 10;
    }

    if enemy_files[file] == 0 {
        score += 6;
    }

    score
}

fn rook_file_bonus(
    square: engine_core::Square,
    color: Color,
    white_pawn_files: &[u8; 8],
    black_pawn_files: &[u8; 8],
) -> i32 {
    let file = square.file() as usize;
    let own = match color {
        Color::White => white_pawn_files[file],
        Color::Black => black_pawn_files[file],
    };
    let enemy = match color {
        Color::White => black_pawn_files[file],
        Color::Black => white_pawn_files[file],
    };
    if own == 0 && enemy == 0 {
        20
    } else if own == 0 {
        10
    } else {
        0
    }
}

fn king_safety_bonus(board: &Board, color: Color) -> i32 {
    let Some(king_square) = board.king_square(color) else {
        return 0;
    };
    let home_rank = if color == Color::White { 0 } else { 7 };
    let mut score = 0;
    if king_square.rank() == home_rank && (king_square.file() == 6 || king_square.file() == 2) {
        score += 20;
    }

    let shield_rank = if color == Color::White {
        king_square.rank().saturating_add(1)
    } else {
        king_square.rank().saturating_sub(1)
    };
    for file in king_square.file().saturating_sub(1)..=((king_square.file() + 1).min(7)) {
        if board.piece_at(engine_core::Square::from_file_rank(file, shield_rank).expect("valid"))
            == Some(Piece {
                color,
                kind: PieceKind::Pawn,
            })
        {
            score += 8;
        } else {
            score -= 10;
        }
    }

    score
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use engine_core::{Board, Color};

    use super::{
        Bound, ClockControl, SearchLimits, Searcher, TranspositionEntry, TranspositionTable,
        evaluate_position, late_move_reduction,
    };

    #[test]
    fn finds_simple_mate_in_one() {
        let board = Board::from_fen("6k1/5ppp/8/8/8/5Q2/6PP/6K1 w - - 0 1").unwrap();
        let mut searcher = Searcher::default();
        let result = searcher.search(
            &board,
            SearchLimits {
                depth: Some(2),
                movetime: Some(Duration::from_millis(250)),
                ..SearchLimits::default()
            },
        );
        assert_eq!(
            result.best_move.map(|mv| mv.to_uci()),
            Some("f3a8".to_string())
        );
    }

    #[test]
    fn search_populates_transposition_table() {
        let board = Board::startpos();
        let mut searcher = Searcher::default();
        let result = searcher.search(
            &board,
            SearchLimits {
                depth: Some(3),
                movetime: Some(Duration::from_millis(500)),
                ..SearchLimits::default()
            },
        );
        assert!(result.best_move.is_some());
        assert!(!searcher.tt.is_empty());
        assert!(searcher.tt.contains_key(board.position_hash()));
        assert!(searcher.tt.values().any(|entry| entry.best_move.is_some()));
    }

    #[test]
    fn transposition_table_replaces_a_shallower_collision_without_growing() {
        let mut table = TranspositionTable::new(2);
        table.begin_search();
        table.store(
            1,
            TranspositionEntry {
                depth: 2,
                score: 10,
                bound: Bound::Exact,
                best_move: None,
            },
        );
        table.store(
            3,
            TranspositionEntry {
                depth: 4,
                score: 20,
                bound: Bound::Exact,
                best_move: None,
            },
        );

        assert_eq!(table.len(), 1);
        assert!(table.get(1).is_none());
        assert_eq!(table.get(3).map(|entry| entry.score), Some(20));
    }

    #[test]
    fn late_move_reduction_keeps_early_and_tactical_moves_at_full_depth() {
        assert_eq!(late_move_reduction(5, 0, true), 0);
        assert_eq!(late_move_reduction(5, 4, false), 0);
        assert_eq!(late_move_reduction(2, 4, true), 0);
        assert_eq!(late_move_reduction(5, 4, true), 1);
        assert_eq!(late_move_reduction(8, 10, true), 2);
    }

    #[test]
    fn evaluation_prefers_passed_pawn_and_bishop_pair() {
        let white_edge = Board::from_fen("4k3/8/8/3P4/8/8/4BB2/4K3 w - - 0 1").unwrap();
        let black_edge = Board::from_fen("4k3/4bb2/8/8/3p4/8/8/4K3 b - - 0 1").unwrap();
        assert!(evaluate_position(&white_edge) > 0);
        assert!(evaluate_position(&black_edge) > 0);
    }

    #[test]
    fn finds_hanging_queen_tactic() {
        let board = Board::from_fen("4k3/8/8/8/4q3/8/4Q3/4K3 w - - 0 1").unwrap();
        let mut searcher = Searcher::default();
        let result = searcher.search(
            &board,
            SearchLimits {
                depth: Some(2),
                movetime: Some(Duration::from_millis(250)),
                ..SearchLimits::default()
            },
        );
        assert_eq!(
            result.best_move.map(|mv| mv.to_uci()),
            Some("e2e4".to_string())
        );
        assert!(result.score_cp > 700);
    }

    #[test]
    fn mirrored_position_keeps_side_to_move_perspective() {
        let white = Board::from_fen("4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1").unwrap();
        let black = Board::from_fen("4k3/4q3/8/8/8/8/8/4K3 b - - 0 1").unwrap();
        assert!(evaluate_position(&white) > 0);
        assert!(evaluate_position(&black) > 0);
        assert_eq!(white.side_to_move, Color::White);
    }

    #[test]
    fn clock_budget_uses_side_to_move_clock() {
        let board = Board::startpos();
        let searcher = Searcher::default();
        let budget = searcher.time_budget(
            board.side_to_move,
            &SearchLimits {
                clock: Some(ClockControl {
                    white_time: Duration::from_secs(60),
                    black_time: Duration::from_secs(10),
                    white_increment: Duration::from_secs(2),
                    black_increment: Duration::ZERO,
                    moves_to_go: Some(20),
                }),
                ..SearchLimits::default()
            },
        );
        assert!(budget.is_some());
        assert!(budget.unwrap() > Duration::from_secs(1));
    }

    #[test]
    fn movetime_overrides_clock_budget() {
        let board = Board::startpos();
        let searcher = Searcher::default();
        let budget = searcher.time_budget(
            board.side_to_move,
            &SearchLimits {
                movetime: Some(Duration::from_millis(750)),
                clock: Some(ClockControl {
                    white_time: Duration::from_secs(60),
                    black_time: Duration::from_secs(60),
                    white_increment: Duration::ZERO,
                    black_increment: Duration::ZERO,
                    moves_to_go: Some(30),
                }),
                ..SearchLimits::default()
            },
        );
        assert_eq!(budget, Some(Duration::from_millis(750)));
    }

    #[test]
    fn finds_legal_evasion_while_in_check() {
        let board = Board::from_fen("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1").unwrap();
        let mut searcher = Searcher::default();
        let result = searcher.search(
            &board,
            SearchLimits {
                depth: Some(2),
                movetime: Some(Duration::from_millis(250)),
                ..SearchLimits::default()
            },
        );
        assert_eq!(
            result.best_move.map(|mv| mv.to_uci()),
            Some("e1e2".to_string())
        );
    }
}
