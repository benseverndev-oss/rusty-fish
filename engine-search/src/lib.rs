use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::time::{Duration, Instant};

use engine_core::{Board, ChessMove, Color, GameStatus, Piece, PieceKind, Square, UndoState};
use pyrrhic_rs::{
    Color as TbColor, DtzProbeValue, EngineAdapter, Piece as TbPiece, TableBases, WdlProbeResult,
};

mod nnue;

pub use nnue::{active_features, Nnue, INPUT_DIMENSION};

const MATE_SCORE: i32 = 100_000;
const MAX_KILLER_PLY: usize = 128;
const HISTORY_PROMOTION_STATES: usize = 5;
const HISTORY_SIZE: usize = 64 * 64 * HISTORY_PROMOTION_STATES;
const TT_CLUSTER_SIZE: usize = 4;
const TT_SHARD_COUNT: usize = 64;

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

fn history_index(mv: ChessMove) -> usize {
    let promotion = match mv.promotion {
        None => 0,
        Some(PieceKind::Knight) => 1,
        Some(PieceKind::Bishop) => 2,
        Some(PieceKind::Rook) => 3,
        Some(PieceKind::Queen) => 4,
        Some(PieceKind::Pawn | PieceKind::King) => 0,
    };
    ((usize::from(mv.from.0) * 64 + usize::from(mv.to.0)) * HISTORY_PROMOTION_STATES)
        + promotion
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
    pub syzygy_probe_depth: u8,
    pub syzygy_probe_limit: u8,
    pub threads: usize,
    /// 0 always plays the highest-weight book move; higher values widen
    /// deterministic weighted selection among book alternatives.
    pub book_variety: u8,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            max_depth: 16,
            hash_mb: 16,
            move_overhead: Duration::from_millis(25),
            syzygy_probe_depth: 1,
            syzygy_probe_limit: 7,
            threads: 1,
            book_variety: 0,
        }
    }
}

/// Tunable scalar search parameters. `Default` reproduces the engine's
/// hand-set constants exactly, so an untuned engine is unchanged. These are the
/// knobs the SPSA tuner in `engine-bench` optimises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchParams {
    pub aspiration_window: i32,
    pub razor_margin_base: i32,
    pub razor_margin_scale: i32,
    pub reverse_futility_base: i32,
    pub reverse_futility_scale: i32,
    pub late_move_pruning_base: usize,
    pub late_move_pruning_scale: usize,
    pub null_move_reduction: u8,
    /// Scales the mobility evaluation term, 0–100. 0 disables it (and skips its
    /// cost). Excluded from the SPSA vector; tuned in a later sub-project.
    pub mobility_scale: i32,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            aspiration_window: 50,
            razor_margin_base: 120,
            razor_margin_scale: 80,
            reverse_futility_base: 100,
            reverse_futility_scale: 90,
            late_move_pruning_base: 3,
            late_move_pruning_scale: 2,
            null_move_reduction: 3,
            mobility_scale: 100,
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

#[derive(Clone, Debug, Default)]
pub struct OpeningBook {
    entries: HashMap<u64, Vec<BookMove>>,
}

#[derive(Clone, Copy, Debug)]
struct BookMove {
    mv: ChessMove,
    weight: u32,
}

impl OpeningBook {
    pub fn from_text(text: &str) -> Result<Self, String> {
        let mut lines = text.lines().filter(|line| !line.trim().is_empty());
        let version = lines
            .next()
            .ok_or_else(|| "opening book is empty".to_string())?;
        let v2 = match version {
            "rusty-fish-book v1" => false,
            "rusty-fish-book v2" => true,
            _ => {
                return Err(
                    "opening book must start with `rusty-fish-book v1` or `rusty-fish-book v2`"
                        .to_string(),
                );
            }
        };

        let mut entries = HashMap::new();
        for (line_number, line) in lines.enumerate() {
            let (fen, moves) = line.split_once('\t').ok_or_else(|| {
                format!(
                    "opening book line {} must contain a tab separator",
                    line_number + 2
                )
            })?;
            let board = Board::from_fen(fen)?;
            let moves = moves
                .split_whitespace()
                .map(|entry| {
                    let (uci, weight) = if v2 {
                        let (uci, weight) = entry.split_once(':').ok_or_else(|| {
                            format!(
                                "opening book v2 line {} move must have `uci:weight`",
                                line_number + 2
                            )
                        })?;
                        let weight = weight.parse::<u32>().map_err(|_| {
                            format!(
                                "opening book v2 line {} move weight must be a positive integer",
                                line_number + 2
                            )
                        })?;
                        if weight == 0 {
                            return Err(format!(
                                "opening book v2 line {} move weight must be positive",
                                line_number + 2
                            ));
                        }
                        (uci, weight)
                    } else {
                        (entry, 1)
                    };
                    Ok(BookMove {
                        mv: board.parse_uci_move(uci)?,
                        weight,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if moves.is_empty() {
                return Err(format!(
                    "opening book line {} has no moves",
                    line_number + 2
                ));
            }
            let mut moves = moves;
            if v2 {
                moves.sort_unstable_by(|left, right| {
                    right
                        .weight
                        .cmp(&left.weight)
                        .then_with(|| left.mv.to_uci().cmp(&right.mv.to_uci()))
                });
            }
            entries.insert(board.position_hash(), moves);
        }
        Ok(Self { entries })
    }

    pub fn select(&self, board: &Board) -> Option<ChessMove> {
        self.entries.get(&board.position_hash()).and_then(|moves| {
            moves
                .iter()
                .find(|entry| board.parse_uci_move(&entry.mv.to_uci()).is_ok())
                .map(|entry| entry.mv)
        })
    }

    pub fn select_with_variety(&self, board: &Board, variety: u8) -> Option<ChessMove> {
        if variety == 0 {
            return self.select(board);
        }
        let moves = self.entries.get(&board.position_hash())?;
        let total_weight = moves
            .iter()
            .filter(|entry| board.parse_uci_move(&entry.mv.to_uci()).is_ok())
            .map(|entry| u64::from(entry.weight))
            .sum::<u64>();
        if total_weight == 0 {
            return None;
        }
        let ticket = (board.position_hash() ^ u64::from(variety)) % total_weight;
        let mut upper_bound = 0_u64;
        moves.iter().find_map(|entry| {
            if board.parse_uci_move(&entry.mv.to_uci()).is_err() {
                return None;
            }
            upper_bound += u64::from(entry.weight);
            (ticket < upper_bound).then_some(entry.mv)
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyzygyWdl {
    Loss,
    Draw,
    Win,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyzygyRootProbe {
    pub best_move: ChessMove,
    pub wdl: SyzygyWdl,
    pub dtz: u16,
}

pub struct SyzygyTablebases {
    tables: TableBases<RustyFishTablebaseAdapter>,
    probe_limit: u8,
}

impl SyzygyTablebases {
    pub fn load(path: &str, probe_limit: u8) -> Result<Self, String> {
        if path.split(';').any(|entry| !Path::new(entry).is_dir()) {
            return Err(format!("Syzygy tablebase directory does not exist: {path}"));
        }
        TableBases::<RustyFishTablebaseAdapter>::new(path)
            .map(|tables| Self { tables, probe_limit: probe_limit.clamp(3, 7) })
            .map_err(|error| format!("could not load Syzygy tablebases: {error:?}"))
    }

    pub fn probe_wdl(&self, board: &Board, depth: u8, probe_depth: u8) -> Option<SyzygyWdl> {
        let white = board.occupancy(Color::White);
        let black = board.occupancy(Color::Black);
        if depth < probe_depth || (white | black).count_ones() > self.tables.max_pieces().min(u32::from(self.probe_limit)) {
            return None;
        }
        let ep = board.en_passant().map_or(0, |square| u32::from(square.0));
        let result = self
            .tables
            .probe_wdl(
                white,
                black,
                board.pieces(Color::White, PieceKind::King)
                    | board.pieces(Color::Black, PieceKind::King),
                board.pieces(Color::White, PieceKind::Queen)
                    | board.pieces(Color::Black, PieceKind::Queen),
                board.pieces(Color::White, PieceKind::Rook)
                    | board.pieces(Color::Black, PieceKind::Rook),
                board.pieces(Color::White, PieceKind::Bishop)
                    | board.pieces(Color::Black, PieceKind::Bishop),
                board.pieces(Color::White, PieceKind::Knight)
                    | board.pieces(Color::Black, PieceKind::Knight),
                board.pieces(Color::White, PieceKind::Pawn)
                    | board.pieces(Color::Black, PieceKind::Pawn),
                ep,
                board.side_to_move == Color::White,
            )
            .ok()?;
        Some(syzygy_wdl(result))
    }

    pub fn probe_root(&self, board: &Board) -> Option<SyzygyRootProbe> {
        let white = board.occupancy(Color::White);
        let black = board.occupancy(Color::Black);
        if (white | black).count_ones() > self.tables.max_pieces().min(u32::from(self.probe_limit)) {
            return None;
        }
        let ep = board.en_passant().map_or(0, |square| u32::from(square.0));
        let result = self
            .tables
            .probe_root(
                white,
                black,
                board.pieces(Color::White, PieceKind::King)
                    | board.pieces(Color::Black, PieceKind::King),
                board.pieces(Color::White, PieceKind::Queen)
                    | board.pieces(Color::Black, PieceKind::Queen),
                board.pieces(Color::White, PieceKind::Rook)
                    | board.pieces(Color::Black, PieceKind::Rook),
                board.pieces(Color::White, PieceKind::Bishop)
                    | board.pieces(Color::Black, PieceKind::Bishop),
                board.pieces(Color::White, PieceKind::Knight)
                    | board.pieces(Color::Black, PieceKind::Knight),
                board.pieces(Color::White, PieceKind::Pawn)
                    | board.pieces(Color::Black, PieceKind::Pawn),
                board.halfmove_clock(),
                ep,
                board.side_to_move == Color::White,
            )
            .ok()?;
        let DtzProbeValue::DtzResult(root) = result.root else {
            return None;
        };
        let candidate = ChessMove {
            from: Square(root.from_square),
            to: Square(root.to_square),
            promotion: promotion_from_tablebase(root.promotion),
        };
        Some(SyzygyRootProbe {
            best_move: board.parse_uci_move(&candidate.to_uci()).ok()?,
            wdl: syzygy_wdl(root.wdl),
            dtz: root.dtz,
        })
    }
}

fn syzygy_wdl(result: WdlProbeResult) -> SyzygyWdl {
    match result {
        WdlProbeResult::Win | WdlProbeResult::CursedWin => SyzygyWdl::Win,
        WdlProbeResult::Draw => SyzygyWdl::Draw,
        WdlProbeResult::Loss | WdlProbeResult::BlessedLoss => SyzygyWdl::Loss,
    }
}

fn promotion_from_tablebase(piece: TbPiece) -> Option<PieceKind> {
    match piece {
        TbPiece::Queen => Some(PieceKind::Queen),
        TbPiece::Rook => Some(PieceKind::Rook),
        TbPiece::Bishop => Some(PieceKind::Bishop),
        TbPiece::Knight => Some(PieceKind::Knight),
        TbPiece::Pawn | TbPiece::King => None,
    }
}

#[derive(Clone)]
struct RustyFishTablebaseAdapter;

impl EngineAdapter for RustyFishTablebaseAdapter {
    fn pawn_attacks(color: TbColor, square: u64) -> u64 {
        let square = engine_core::Square(square as u8);
        let rank_delta = if color == TbColor::White { 1 } else { -1 };
        [(-1, rank_delta), (1, rank_delta)]
            .iter()
            .fold(0, |mask, &(file_delta, rank_delta)| {
                square
                    .offset(file_delta, rank_delta)
                    .map_or(mask, |target| mask | (1_u64 << target.0))
            })
    }

    fn knight_attacks(square: u64) -> u64 {
        tablebase_attack_mask(
            engine_core::Square(square as u8),
            &[
                (-2, -1),
                (-2, 1),
                (-1, -2),
                (-1, 2),
                (1, -2),
                (1, 2),
                (2, -1),
                (2, 1),
            ],
        )
    }

    fn bishop_attacks(square: u64, occupied: u64) -> u64 {
        tablebase_sliding_attacks(
            engine_core::Square(square as u8),
            occupied,
            &[(-1, -1), (-1, 1), (1, -1), (1, 1)],
        )
    }

    fn rook_attacks(square: u64, occupied: u64) -> u64 {
        tablebase_sliding_attacks(
            engine_core::Square(square as u8),
            occupied,
            &[(-1, 0), (1, 0), (0, -1), (0, 1)],
        )
    }

    fn queen_attacks(square: u64, occupied: u64) -> u64 {
        Self::bishop_attacks(square, occupied) | Self::rook_attacks(square, occupied)
    }

    fn king_attacks(square: u64) -> u64 {
        tablebase_attack_mask(
            engine_core::Square(square as u8),
            &[
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ],
        )
    }
}

fn tablebase_attack_mask(square: engine_core::Square, deltas: &[(i8, i8)]) -> u64 {
    deltas.iter().fold(0, |mask, &(file_delta, rank_delta)| {
        square
            .offset(file_delta, rank_delta)
            .map_or(mask, |target| mask | (1_u64 << target.0))
    })
}

fn tablebase_sliding_attacks(
    square: engine_core::Square,
    occupied: u64,
    directions: &[(i8, i8)],
) -> u64 {
    let mut attacks = 0;
    for &(file_delta, rank_delta) in directions {
        let mut current = square;
        while let Some(target) = current.offset(file_delta, rank_delta) {
            let bit = 1_u64 << target.0;
            attacks |= bit;
            if occupied & bit != 0 {
                break;
            }
            current = target;
        }
    }
    attacks
}

pub struct Searcher {
    nodes: u64,
    start: Instant,
    deadline: Option<Instant>,
    stopped: bool,
    stop_signal: Option<Arc<AtomicBool>>,
    tt: Arc<SharedTranspositionTable>,
    killer_moves: Vec<[Option<ChessMove>; 2]>,
    history: Vec<i32>,
    counter_moves: Vec<Option<ChessMove>>,
    options: SearchOptions,
    params: SearchParams,
    eval_params: EvalParams,
    nnue: Option<Arc<Nnue>>,
    nnue_accumulator: Option<nnue::Accumulator>,
    nnue_stack: Vec<NnueDelta>,
    opening_book: Option<OpeningBook>,
    syzygy: Option<SyzygyTablebases>,
}

/// One square's piece change from a move: `(square, before, after)`.
type NnueChange = (Square, Option<Piece>, Option<Piece>);

/// The set of square changes a single move makes, used to update the NNUE
/// accumulator incrementally and to reverse it on unmake. A move touches at
/// most four squares (castling moves the king and the rook).
#[derive(Clone, Copy)]
struct NnueDelta {
    changes: [NnueChange; 4],
    len: usize,
}

impl Default for Searcher {
    fn default() -> Self {
        Self {
            nodes: 0,
            start: Instant::now(),
            deadline: None,
            stopped: false,
            stop_signal: None,
            tt: Arc::new(SharedTranspositionTable::new(tt_capacity_entries_for(
                SearchOptions::default().hash_mb,
            ))),
            killer_moves: vec![[None, None]; MAX_KILLER_PLY],
            history: vec![0; HISTORY_SIZE],
            counter_moves: vec![None; HISTORY_SIZE],
            options: SearchOptions::default(),
            params: SearchParams::default(),
            eval_params: EvalParams::default(),
            nnue: None,
            nnue_accumulator: None,
            nnue_stack: Vec::new(),
            opening_book: None,
            syzygy: None,
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
    clusters: Vec<[Option<TranspositionSlot>; TT_CLUSTER_SIZE]>,
    generation: u8,
}

impl TranspositionTable {
    fn new(capacity: usize) -> Self {
        let cluster_count = capacity.max(TT_CLUSTER_SIZE).div_ceil(TT_CLUSTER_SIZE);
        Self {
            clusters: vec![[None; TT_CLUSTER_SIZE]; cluster_count],
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
        self.clusters[self.index(key)]
            .iter()
            .flatten()
            .find(|slot| slot.key == key)
            .map(|slot| &slot.entry)
    }

    fn store(&mut self, key: u64, entry: TranspositionEntry) {
        let index = self.index(key);
        let replacement = TranspositionSlot {
            key,
            generation: self.generation,
            entry,
        };

        let cluster = &mut self.clusters[index];
        if let Some(slot) = cluster
            .iter_mut()
            .find(|slot| slot.is_some_and(|slot| slot.key == key))
        {
            let current = (*slot).expect("matching slot must contain an entry");
            if entry.depth >= current.entry.depth || entry.bound == Bound::Exact {
                *slot = Some(replacement);
            }
            return;
        }

        if let Some(slot) = cluster.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(replacement);
            return;
        }

        let victim_index = cluster
            .iter()
            .enumerate()
            .min_by_key(|(_, slot)| {
                let slot = (*slot).expect("full cluster must contain entries");
                (u8::from(slot.generation == self.generation), slot.entry.depth)
            })
            .map(|(index, _)| index)
            .expect("transposition table cluster cannot be empty");
        let victim = cluster[victim_index].expect("full cluster must contain entries");
        if victim.generation != self.generation || entry.depth > victim.entry.depth {
            cluster[victim_index] = Some(replacement);
        }
    }

    #[cfg(test)]
    fn contains_key(&self, key: u64) -> bool {
        self.get(key).is_some()
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.clusters.iter().flatten().all(Option::is_none)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.clusters.iter().flatten().flatten().count()
    }

    #[cfg(test)]
    fn values(&self) -> impl Iterator<Item = &TranspositionEntry> {
        self.clusters
            .iter()
            .flat_map(|cluster| cluster.iter().flatten())
            .map(|slot| &slot.entry)
    }

    fn index(&self, key: u64) -> usize {
        (key as usize) % self.clusters.len()
    }
}

/// A transposition table that many search threads may probe and store into
/// concurrently. It shards the key space across independently locked
/// [`TranspositionTable`]s: the shard is chosen from the key's high bits while
/// each inner table indexes clusters from the low bits, keeping the two
/// selections decorrelated. Entries are `Copy`, so no borrow is ever held
/// across a lock.
#[derive(Debug)]
struct SharedTranspositionTable {
    shards: Vec<Mutex<TranspositionTable>>,
}

impl SharedTranspositionTable {
    fn new(capacity: usize) -> Self {
        let per_shard = capacity.div_ceil(TT_SHARD_COUNT).max(TT_CLUSTER_SIZE);
        let shards = (0..TT_SHARD_COUNT)
            .map(|_| Mutex::new(TranspositionTable::new(per_shard)))
            .collect();
        Self { shards }
    }

    fn shard(&self, key: u64) -> &Mutex<TranspositionTable> {
        let index = (key >> 48) as usize % self.shards.len();
        &self.shards[index]
    }

    fn begin_search(&self) {
        for shard in &self.shards {
            shard.lock().expect("transposition shard poisoned").begin_search();
        }
    }

    fn resize(&self, capacity: usize) {
        let per_shard = capacity.div_ceil(TT_SHARD_COUNT).max(TT_CLUSTER_SIZE);
        for shard in &self.shards {
            shard.lock().expect("transposition shard poisoned").resize(per_shard);
        }
    }

    fn get(&self, key: u64) -> Option<TranspositionEntry> {
        self.shard(key)
            .lock()
            .expect("transposition shard poisoned")
            .get(key)
            .copied()
    }

    fn store(&self, key: u64, entry: TranspositionEntry) {
        self.shard(key)
            .lock()
            .expect("transposition shard poisoned")
            .store(key, entry);
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.shards
            .iter()
            .all(|shard| shard.lock().expect("transposition shard poisoned").is_empty())
    }

    #[cfg(test)]
    fn contains_key(&self, key: u64) -> bool {
        self.shard(key)
            .lock()
            .expect("transposition shard poisoned")
            .contains_key(key)
    }

    #[cfg(test)]
    fn values(&self) -> impl Iterator<Item = TranspositionEntry> {
        let mut entries = Vec::new();
        for shard in &self.shards {
            entries.extend(
                shard
                    .lock()
                    .expect("transposition shard poisoned")
                    .values()
                    .copied(),
            );
        }
        entries.into_iter()
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

    pub fn search_params(&self) -> &SearchParams {
        &self.params
    }

    pub fn set_search_params(&mut self, params: SearchParams) {
        self.params = params;
    }

    /// Sets the hand-crafted evaluation weights. Only affects the fallback
    /// hand-crafted evaluation, not an installed NNUE network.
    pub fn set_eval_params(&mut self, eval_params: EvalParams) {
        self.eval_params = eval_params;
    }

    /// Installs an NNUE network as the evaluation. Passing `None` restores the
    /// hand-crafted evaluation. The network is shared (read-only) across Lazy
    /// SMP helper threads.
    pub fn set_nnue(&mut self, nnue: Option<Arc<Nnue>>) {
        self.nnue = nnue;
    }

    pub fn has_nnue(&self) -> bool {
        self.nnue.is_some()
    }

    pub fn set_opening_book(&mut self, opening_book: Option<OpeningBook>) {
        self.opening_book = opening_book;
    }

    pub fn set_syzygy_tablebases(&mut self, syzygy: Option<SyzygyTablebases>) {
        self.syzygy = syzygy;
    }

    /// Builds a helper searcher for Lazy SMP. It shares the transposition table
    /// (via the `Arc`) but keeps its own killer/history/counter-move tables and
    /// never consults the opening book or Syzygy tablebases; the primary thread
    /// remains the single source of reported output.
    fn helper(
        tt: Arc<SharedTranspositionTable>,
        options: SearchOptions,
        params: SearchParams,
        eval_params: EvalParams,
        nnue: Option<Arc<Nnue>>,
    ) -> Self {
        Self {
            nodes: 0,
            start: Instant::now(),
            deadline: None,
            stopped: false,
            stop_signal: None,
            tt,
            killer_moves: vec![[None, None]; MAX_KILLER_PLY],
            history: vec![0; HISTORY_SIZE],
            counter_moves: vec![None; HISTORY_SIZE],
            options,
            params,
            eval_params,
            nnue,
            nnue_accumulator: None,
            nnue_stack: Vec::new(),
            opening_book: None,
            syzygy: None,
        }
    }

    /// Runs a helper thread's iterative deepening. It only deepens the shared
    /// transposition table; its results are discarded. It must not bump the
    /// shared generation (the primary thread already did) and it exits as soon
    /// as the shared stop signal fires or the deadline passes.
    fn run_lazy_smp_helper(
        &mut self,
        board: &Board,
        max_depth: u8,
        deadline: Option<Instant>,
        stop_signal: Arc<AtomicBool>,
        index: usize,
    ) {
        self.nodes = 0;
        self.start = Instant::now();
        self.deadline = deadline;
        self.stopped = false;
        self.stop_signal = Some(stop_signal);
        self.killer_moves.fill([None, None]);
        self.history.fill(0);
        self.counter_moves.fill(None);
        self.nnue_refresh(board);

        // Desynchronise the fleet: odd-indexed helpers begin one ply deeper so
        // threads explore the shared tree from different starting points.
        let start_depth = 1 + u8::from(index % 2 == 1);
        let mut best_score = 0;
        for depth in start_depth..=max_depth {
            let mut clone = board.clone();
            let (score, _pv) = if depth == 1 {
                self.negamax_root(&mut clone, depth, -MATE_SCORE, MATE_SCORE)
            } else {
                self.aspiration_search(&mut clone, depth, best_score)
            };
            if self.stopped {
                break;
            }
            best_score = score;
            if best_score.abs() >= MATE_SCORE - 128 {
                break;
            }
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
        self.search_with_callback_and_stop_signal(board, limits, None, callback)
    }

    pub fn search_with_stop_signal(
        &mut self,
        board: &Board,
        limits: SearchLimits,
        stop_signal: Arc<AtomicBool>,
    ) -> SearchResult {
        self.search_with_callback_and_stop_signal(board, limits, Some(stop_signal), |_info| {})
    }

    pub fn search_with_callback_and_stop_signal<F>(
        &mut self,
        board: &Board,
        limits: SearchLimits,
        stop_signal: Option<Arc<AtomicBool>>,
        mut callback: F,
    ) -> SearchResult
    where
        F: FnMut(&SearchInfo),
    {
        if let Some(root) = self
            .syzygy
            .as_ref()
            .and_then(|tables| tables.probe_root(board))
        {
            return root_tablebase_search_result(root);
        }

        if let Some(best_move) = self
            .opening_book
            .as_ref()
            .and_then(|book| book.select_with_variety(board, self.options.book_variety))
        {
            return SearchResult {
                best_move: Some(best_move),
                depth: 0,
                score_cp: 0,
                nodes: 0,
                elapsed: Duration::ZERO,
                pv: vec![best_move],
            };
        }
        self.nodes = 0;
        self.start = Instant::now();
        self.deadline = self
            .time_budget(board.side_to_move, &limits)
            .map(|limit| self.start + limit);
        self.stopped = false;
        self.stop_signal = stop_signal;
        self.tt.begin_search();
        self.killer_moves.fill([None, None]);
        self.history.fill(0);
        self.counter_moves.fill(None);
        self.nnue_refresh(board);

        let max_depth = if limits.infinite {
            u8::MAX
        } else {
            limits
                .depth
                .unwrap_or(self.options.max_depth)
                .max(1)
                .min(self.options.max_depth)
        };
        // Lazy SMP: spawn helper threads that cooperate through the shared
        // transposition table. They share only the table and the stop signal;
        // each keeps its own search heuristics. Nothing non-`Send` crosses the
        // thread boundary because each helper builds its `Searcher` inside its
        // own closure.
        let threads = self.options.threads.max(1);
        let mut helper_handles = Vec::new();
        if threads > 1 && max_depth > 1 && !self.stopped {
            let shared_stop = match self.stop_signal.clone() {
                Some(signal) => signal,
                None => {
                    let signal = Arc::new(AtomicBool::new(false));
                    self.stop_signal = Some(Arc::clone(&signal));
                    signal
                }
            };
            let deadline = self.deadline;
            for index in 1..threads {
                let tt = Arc::clone(&self.tt);
                let options = self.options.clone();
                let params = self.params;
                let eval_params = self.eval_params;
                let nnue = self.nnue.clone();
                let helper_board = board.clone();
                let stop = Arc::clone(&shared_stop);
                helper_handles.push(thread::spawn(move || {
                    Searcher::helper(tt, options, params, eval_params, nnue).run_lazy_smp_helper(
                        &helper_board,
                        max_depth,
                        deadline,
                        stop,
                        index,
                    );
                }));
            }
        }

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

        // Signal helpers to stop and wait for them to unwind before returning.
        if !helper_handles.is_empty() {
            if let Some(signal) = self.stop_signal.as_ref() {
                signal.store(true, Ordering::Relaxed);
            }
            for handle in helper_handles {
                let _ = handle.join();
            }
        }

        let result = SearchResult {
            best_move,
            depth: reached_depth,
            score_cp: best_score,
            nodes: self.nodes,
            elapsed: self.start.elapsed(),
            pv: best_pv,
        };
        self.stop_signal = None;
        result
    }

    fn aspiration_search(
        &mut self,
        board: &mut Board,
        depth: u8,
        previous_score: i32,
    ) -> (i32, Vec<ChessMove>) {
        let mut window = self.params.aspiration_window;
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
        let mut moves = board.generate_legal_move_list();
        self.order_moves(board, moves.as_mut_slice(), 0, tt_move, None);
        if moves.is_empty() {
            return (self.evaluate_terminal(board, 0), Vec::new());
        }

        let mut best_score = -MATE_SCORE;
        let mut best_line = Vec::new();
        let original_alpha = alpha;
        for (move_index, &mv) in moves.as_slice().iter().enumerate() {
            if self.should_stop() {
                break;
            }
            let undo = self.nnue_make(board, mv);
            let child_depth = depth.saturating_sub(1);
            let (mut score, mut line) = if move_index == 0 {
                let (score, line) = self.negamax(board, child_depth, 1, -beta, -alpha, Some(mv), None);
                (-score, line)
            } else {
                let (score, line) =
                    self.negamax(board, child_depth, 1, -alpha - 1, -alpha, Some(mv), None);
                (-score, line)
            };
            if move_index > 0 && score > alpha && score < beta && !self.stopped {
                let (full_score, full_line) =
                    self.negamax(board, child_depth, 1, -beta, -alpha, Some(mv), None);
                score = -full_score;
                line = full_line;
            }
            self.nnue_unmake(board, mv, undo);

            if score > best_score {
                best_score = score;
                best_line.clear();
                best_line.push(mv);
                best_line.append(&mut line);
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                self.record_cutoff(0, mv, depth, None, self.is_quiet_move(board, mv));
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
        previous_move: Option<ChessMove>,
        excluded_move: Option<ChessMove>,
    ) -> (i32, Vec<ChessMove>) {
        if self.should_stop() {
            return (self.evaluate(board), Vec::new());
        }
        self.nodes += 1;
        let original_alpha = alpha;
        let tt_key = board.position_hash();
        let in_check = board.in_check(board.side_to_move);

        if excluded_move.is_none()
            && let Some(entry) = self.tt.get(tt_key)
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

        if board.is_draw_by_rule() {
            return (0, Vec::new());
        }

        if let Some(wdl) = self
            .syzygy
            .as_ref()
            .and_then(|syzygy| syzygy.probe_wdl(board, depth, self.options.syzygy_probe_depth))
        {
            return (syzygy_score(wdl, ply), Vec::new());
        }

        if depth == 0 {
            return (self.quiescence(board, alpha, beta), Vec::new());
        }

        let can_static_prune = can_apply_static_pruning(
            depth,
            in_check,
            alpha,
            beta,
            self.has_non_pawn_material(board, board.side_to_move),
        );
        if can_static_prune {
            let static_eval = self.evaluate(board);
            if depth == 1 && static_eval + razor_margin(&self.params, depth) <= alpha {
                return (self.quiescence(board, alpha, beta), Vec::new());
            }
            if static_eval - reverse_futility_margin(&self.params, depth) >= beta {
                return (static_eval, Vec::new());
            }
        }

        if !in_check && depth >= 3 && self.has_non_pawn_material(board, board.side_to_move) {
            let null_score = self.try_null_move(board, depth, ply, beta);
            if null_score >= beta {
                return (null_score, Vec::new());
            }
        }

        let tt_move = self.tt.get(tt_key).and_then(|entry| entry.best_move);
        let singular_candidate = excluded_move.is_none().then(|| {
            self.tt
                .get(tt_key)
                .filter(|entry| {
                    can_try_singular_extension(
                        depth,
                        in_check,
                        self.has_non_pawn_material(board, board.side_to_move),
                        *entry,
                    )
                })
        }).flatten();
        let mut moves = board.generate_legal_move_list();
        self.order_moves(
            board,
            moves.as_mut_slice(),
            ply as usize,
            tt_move,
            previous_move.and_then(|mv| self.counter_moves[history_index(mv)]),
        );
        if moves.is_empty() {
            return (self.evaluate_terminal(board, ply), Vec::new());
        }

        let mut best_score = -MATE_SCORE;
        let mut best_line = Vec::new();
        for (move_index, &mv) in moves.as_slice().iter().enumerate() {
            if excluded_move == Some(mv) {
                continue;
            }
            let singular_extension = singular_candidate.is_some_and(|entry| {
                entry.best_move == Some(mv)
                    && self
                        .negamax(
                            board,
                            depth / 2,
                            ply,
                            singular_verification_beta(entry.score) - 1,
                            singular_verification_beta(entry.score),
                            previous_move,
                            Some(mv),
                        )
                        .0
                        < singular_verification_beta(entry.score)
            });
            let is_quiet = self.is_quiet_move(board, mv);
            let pawn_extension = passed_pawn_extension(board, mv);
            let is_priority_move = Some(mv) == tt_move
                || self
                    .killer_moves
                    .get(ply as usize)
                    .is_some_and(|killers| killers.contains(&Some(mv)))
                || previous_move
                    .and_then(|previous| self.counter_moves[history_index(previous)])
                    == Some(mv);
            if can_static_prune
                && move_index >= late_move_pruning_limit(&self.params, depth)
                && is_quiet
                && pawn_extension == 0
                && !is_priority_move
            {
                break;
            }
            let undo = self.nnue_make(board, mv);
            let extension = u8::from(board.in_check(board.side_to_move))
                .max(pawn_extension)
                .max(u8::from(singular_extension));
            let next_depth = depth.saturating_sub(1) + extension.min(1);
            let reduction = late_move_reduction(depth, move_index, is_quiet && extension == 0);
            let search_depth = next_depth.saturating_sub(reduction);
            let (mut score, mut line) = if move_index == 0 {
                let (score, line) =
                    self.negamax(board, search_depth, ply + 1, -beta, -alpha, Some(mv), None);
                (-score, line)
            } else {
                let (score, line) = self.negamax(
                    board,
                    search_depth,
                    ply + 1,
                    -alpha - 1,
                    -alpha,
                    Some(mv),
                    None,
                );
                (-score, line)
            };
            if reduction > 0 && score > alpha && !self.stopped {
                let (reduced_score, reduced_line) =
                    self.negamax(board, next_depth, ply + 1, -alpha - 1, -alpha, Some(mv), None);
                score = -reduced_score;
                line = reduced_line;
            }
            if move_index > 0 && score > alpha && score < beta && !self.stopped {
                let (full_score, full_line) =
                    self.negamax(board, next_depth, ply + 1, -beta, -alpha, Some(mv), None);
                score = -full_score;
                line = full_line;
            }
            self.nnue_unmake(board, mv, undo);

            if score > best_score {
                best_score = score;
                best_line.clear();
                best_line.push(mv);
                best_line.append(&mut line);
            }
            alpha = alpha.max(score);
            if alpha >= beta {
                self.record_cutoff(ply as usize, mv, depth, previous_move, is_quiet);
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
        if excluded_move.is_none() {
            self.store_tt(
                tt_key,
                TranspositionEntry {
                    depth,
                    score: best_score,
                    bound,
                    best_move: best_line.first().copied(),
                },
            );
        }
        (best_score, best_line)
    }

    fn quiescence(&mut self, board: &mut Board, mut alpha: i32, beta: i32) -> i32 {
        if self.should_stop() {
            return self.evaluate(board);
        }
        self.nodes += 1;

        let in_check = board.in_check(board.side_to_move);
        if in_check {
            let mut evasions = board.generate_legal_move_list();
            let tt_move = self
                .tt
                .get(board.position_hash())
                .and_then(|entry| entry.best_move);
            self.order_moves(board, evasions.as_mut_slice(), 0, tt_move, None);
            if evasions.is_empty() {
                return self.evaluate_terminal(board, 0);
            }
            for &mv in evasions.as_slice() {
                let undo = self.nnue_make(board, mv);
                let score = -self.quiescence(board, -beta, -alpha);
                self.nnue_unmake(board, mv, undo);
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

        let mut moves = board.generate_capture_move_list();
        self.order_moves(board, moves.as_mut_slice(), 0, None, None);
        for &mv in moves.as_slice() {
            if !self.is_promising_quiescence_capture(board, mv, stand_pat, alpha) {
                continue;
            }
            let undo = self.nnue_make(board, mv);
            let score = -self.quiescence(board, -beta, -alpha);
            self.nnue_unmake(board, mv, undo);

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
        match self.nnue.as_ref() {
            Some(nnue) => match self.nnue_accumulator.as_ref() {
                Some(accumulator) => {
                    debug_assert_eq!(
                        *accumulator,
                        nnue::Accumulator::refresh(nnue, board),
                        "incremental nnue accumulator desynced from the board",
                    );
                    nnue.evaluate_with(accumulator, board.side_to_move)
                }
                // No maintained accumulator (e.g. a direct evaluate outside a
                // search): fall back to a full refresh.
                None => nnue.evaluate(board, board.side_to_move),
            },
            None => evaluate_position(board, self.params.mobility_scale, &self.eval_params),
        }
    }

    /// Rebuilds the NNUE accumulator from scratch for `board` and clears the
    /// incremental delta stack. Called once at the start of a search when a
    /// network is loaded.
    fn nnue_refresh(&mut self, board: &Board) {
        self.nnue_accumulator = self
            .nnue
            .as_ref()
            .map(|nnue| nnue::Accumulator::refresh(nnue, board));
        self.nnue_stack.clear();
    }

    /// Makes a move, updating the NNUE accumulator incrementally when a network
    /// is loaded. Mirror of [`Searcher::nnue_unmake`].
    fn nnue_make(&mut self, board: &mut Board, mv: ChessMove) -> UndoState {
        if self.nnue.is_none() {
            return board.make_move(mv).expect("generated move must be legal");
        }
        let (squares, square_count) = nnue_changed_squares(board, mv);
        let before: [Option<Piece>; 4] =
            std::array::from_fn(|index| board.piece_at(squares[index]));
        let undo = board.make_move(mv).expect("generated move must be legal");

        let nnue = self.nnue.clone().expect("network is loaded");
        let accumulator = self
            .nnue_accumulator
            .as_mut()
            .expect("nnue accumulator is initialised while a network is loaded");
        let mut delta = NnueDelta {
            changes: [(Square(0), None, None); 4],
            len: 0,
        };
        for index in 0..square_count {
            let square = squares[index];
            let old = before[index];
            let new = board.piece_at(square);
            if old == new {
                continue;
            }
            if let Some(piece) = old {
                accumulator.remove_feature(&nnue, Color::White, piece, square);
                accumulator.remove_feature(&nnue, Color::Black, piece, square);
            }
            if let Some(piece) = new {
                accumulator.add_feature(&nnue, Color::White, piece, square);
                accumulator.add_feature(&nnue, Color::Black, piece, square);
            }
            delta.changes[delta.len] = (square, old, new);
            delta.len += 1;
        }
        self.nnue_stack.push(delta);
        undo
    }

    /// Unmakes a move, reversing the incremental accumulator update pushed by
    /// [`Searcher::nnue_make`].
    fn nnue_unmake(&mut self, board: &mut Board, mv: ChessMove, undo: UndoState) {
        board.unmake_move(mv, undo);
        if self.nnue.is_none() {
            return;
        }
        let nnue = self.nnue.clone().expect("network is loaded");
        let delta = self.nnue_stack.pop().expect("balanced nnue make/unmake");
        let accumulator = self
            .nnue_accumulator
            .as_mut()
            .expect("nnue accumulator is initialised while a network is loaded");
        for &(square, old, new) in delta.changes.iter().take(delta.len) {
            // Reverse of nnue_make: we removed `old` and added `new`, so now
            // remove `new` and restore `old`.
            if let Some(piece) = new {
                accumulator.remove_feature(&nnue, Color::White, piece, square);
                accumulator.remove_feature(&nnue, Color::Black, piece, square);
            }
            if let Some(piece) = old {
                accumulator.add_feature(&nnue, Color::White, piece, square);
                accumulator.add_feature(&nnue, Color::Black, piece, square);
            }
        }
    }

    fn order_moves(
        &self,
        board: &Board,
        moves: &mut [ChessMove],
        ply: usize,
        tt_move: Option<ChessMove>,
        counter_move: Option<ChessMove>,
    ) {
        moves.sort_by_cached_key(|mv| {
            -self.move_order_score(board, *mv, ply, tt_move, counter_move)
        });
    }

    fn move_order_score(
        &self,
        board: &Board,
        mv: ChessMove,
        ply: usize,
        tt_move: Option<ChessMove>,
        counter_move: Option<ChessMove>,
    ) -> i32 {
        if tt_move == Some(mv) {
            return 2_000_000;
        }

        let mut score = 0;
        if let Some(victim) = board.piece_at(mv.to) {
            let attacker = board.piece_at(mv.from).map(piece_value).unwrap_or_default();
            let see = static_exchange_evaluation(board, mv);
            score += if see >= 0 {
                1_000_000 + see * 32 + piece_value(victim) * 16 - attacker
            } else {
                100_000 + see
            };
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
        if counter_move == Some(mv) {
            score += 750_000;
        }

        if let Some(killers) = self.killer_moves.get(ply) {
            if killers[0] == Some(mv) {
                score += 700_000;
            } else if killers[1] == Some(mv) {
                score += 650_000;
            }
        }

        score + self.history[history_index(mv)]
    }

    fn record_cutoff(
        &mut self,
        ply: usize,
        mv: ChessMove,
        depth: u8,
        previous_move: Option<ChessMove>,
        is_quiet: bool,
    ) {
        if ply < self.killer_moves.len() {
            let entry = &mut self.killer_moves[ply];
            if entry[0] != Some(mv) {
                entry[1] = entry[0];
                entry[0] = Some(mv);
            }
        }
        let history = &mut self.history[history_index(mv)];
        *history = history.saturating_add(i32::from(depth) * i32::from(depth) * 16);
        if is_quiet {
            if let Some(previous_move) = previous_move {
                self.counter_moves[history_index(previous_move)] = Some(mv);
            }
        }
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
                depth.saturating_sub(self.params.null_move_reduction),
                ply + 1,
                -beta,
                -beta + 1,
                None,
                None,
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
        if self
            .stop_signal
            .as_ref()
            .is_some_and(|signal| signal.load(Ordering::Relaxed))
        {
            self.stopped = true;
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

fn syzygy_score(wdl: SyzygyWdl, ply: i32) -> i32 {
    match wdl {
        SyzygyWdl::Win => MATE_SCORE - 512 - ply,
        SyzygyWdl::Draw => 0,
        SyzygyWdl::Loss => -MATE_SCORE + 512 + ply,
    }
}

fn razor_margin(params: &SearchParams, depth: u8) -> i32 {
    params.razor_margin_base + params.razor_margin_scale * i32::from(depth)
}

fn reverse_futility_margin(params: &SearchParams, depth: u8) -> i32 {
    params.reverse_futility_base + params.reverse_futility_scale * i32::from(depth)
}

fn late_move_pruning_limit(params: &SearchParams, depth: u8) -> usize {
    params.late_move_pruning_base + usize::from(depth) * params.late_move_pruning_scale
}

fn can_apply_static_pruning(
    depth: u8,
    in_check: bool,
    alpha: i32,
    beta: i32,
    has_non_pawn_material: bool,
) -> bool {
    depth <= 3
        && !in_check
        && has_non_pawn_material
        && alpha.abs() < MATE_SCORE - 1_024
        && beta.abs() < MATE_SCORE - 1_024
}

fn can_try_singular_extension(
    depth: u8,
    in_check: bool,
    has_non_pawn_material: bool,
    entry: TranspositionEntry,
) -> bool {
    depth >= 6
        && !in_check
        && has_non_pawn_material
        && entry.depth >= depth.saturating_sub(3)
        && entry.depth < depth
        && entry.bound == Bound::Exact
        && entry.best_move.is_some()
        && entry.score.abs() < MATE_SCORE - 1_024
}

fn singular_verification_beta(tt_score: i32) -> i32 {
    tt_score - 32
}

fn root_tablebase_search_result(root: SyzygyRootProbe) -> SearchResult {
    SearchResult {
        best_move: Some(root.best_move),
        depth: 0,
        score_cp: syzygy_score(root.wdl, 0),
        nodes: 0,
        elapsed: Duration::ZERO,
        pv: vec![root.best_move],
    }
}

fn static_exchange_evaluation(board: &Board, mv: ChessMove) -> i32 {
    let captured_value = board.piece_at(mv.to).map(piece_value).unwrap_or_else(|| {
        if board.en_passant() == Some(mv.to) {
            piece_kind_value(PieceKind::Pawn)
        } else {
            0
        }
    });
    if captured_value == 0 {
        return 0;
    }

    let mut after_capture = board.clone();
    if after_capture.make_move(mv).is_err() {
        return -MATE_SCORE;
    }
    captured_value - best_exchange_gain(&mut after_capture, mv.to)
}

fn best_exchange_gain(board: &mut Board, target: engine_core::Square) -> i32 {
    let mut best_gain = 0;
    let recaptures = board
        .generate_capture_moves()
        .into_iter()
        .filter(|mv| mv.to == target)
        .collect::<Vec<_>>();

    for recapture in recaptures {
        let captured_value = board.piece_at(target).map(piece_value).unwrap_or_default();
        let undo = board
            .make_move(recapture)
            .expect("generated capture must be legal");
        let gain = captured_value - best_exchange_gain(board, target);
        board.unmake_move(recapture, undo);
        best_gain = best_gain.max(gain);
    }
    best_gain
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TaperedScore {
    pub middlegame: i32,
    pub endgame: i32,
}

impl TaperedScore {
    pub const fn new(middlegame: i32, endgame: i32) -> Self {
        Self {
            middlegame,
            endgame,
        }
    }

    pub const fn equal(value: i32) -> Self {
        Self::new(value, value)
    }

    const fn middlegame(value: i32) -> Self {
        Self::new(value, 0)
    }

    const fn endgame(value: i32) -> Self {
        Self::new(0, value)
    }

    const fn add(self, other: Self) -> Self {
        Self::new(
            self.middlegame + other.middlegame,
            self.endgame + other.endgame,
        )
    }

    fn interpolate(self, phase: i32) -> i32 {
        (self.middlegame * (24 - phase) + self.endgame * phase) / 24
    }
}

/// The tunable hand-crafted evaluation weights. `Default` reproduces today's
/// hardcoded constants exactly, so the default eval is byte-identical. The pawn
/// value (the material anchor) and the mobility offsets stay fixed and are not
/// part of this struct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvalParams {
    pub knight: TaperedScore,
    pub bishop: TaperedScore,
    pub rook: TaperedScore,
    pub queen: TaperedScore,
    pub knight_mobility: TaperedScore,
    pub bishop_mobility: TaperedScore,
    pub rook_mobility: TaperedScore,
    pub queen_mobility: TaperedScore,
    pub bishop_pair: i32,
    pub passed_pawn_base: i32,
}

impl Default for EvalParams {
    fn default() -> Self {
        Self {
            knight: TaperedScore::new(323, 322),
            bishop: TaperedScore::new(334, 322),
            rook: TaperedScore::new(503, 499),
            queen: TaperedScore::new(907, 908),
            knight_mobility: TaperedScore::new(5, 4),
            bishop_mobility: TaperedScore::new(3, 2),
            rook_mobility: TaperedScore::new(3, 4),
            queen_mobility: TaperedScore::new(1, 2),
            bishop_pair: 33,
            passed_pawn_base: 21,
        }
    }
}

/// The fixed pawn value: the material anchor, deliberately not tunable.
const PAWN_VALUE: TaperedScore = TaperedScore::equal(100);

fn tapered_piece_value(piece: Piece, params: &EvalParams) -> TaperedScore {
    match piece.kind {
        PieceKind::Pawn => PAWN_VALUE,
        PieceKind::Knight => params.knight,
        PieceKind::Bishop => params.bishop,
        PieceKind::Rook => params.rook,
        PieceKind::Queen => params.queen,
        PieceKind::King => TaperedScore::equal(0),
    }
}

fn piece_square_bonus(piece: Piece, idx: u8) -> TaperedScore {
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
    let value = match piece.kind {
        PieceKind::Pawn => {
            centrality * 2 + rank as i32 * if piece.color == Color::White { 3 } else { -3 }
        }
        PieceKind::Knight => centrality * 8,
        PieceKind::Bishop => centrality * 5,
        PieceKind::Rook => centrality * 2,
        PieceKind::Queen => centrality * 2,
        PieceKind::King => -centrality * 4,
    };
    TaperedScore::equal(value)
}

#[derive(Default, Clone, Copy)]
struct EvalFeatures {
    white_score: i32,
    black_score: i32,
}

impl EvalFeatures {
    fn add(&mut self, color: Color, value: TaperedScore, phase: i32) {
        let value = value.interpolate(phase);
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

/// The squares whose contents a move changes: always `from` and `to`, plus the
/// rook's two squares for castling and the captured pawn's square for en
/// passant. Returns the squares and how many are valid. Used to update the NNUE
/// accumulator incrementally without allocating.
fn nnue_changed_squares(board: &Board, mv: ChessMove) -> ([Square; 4], usize) {
    let mut squares = [mv.from, mv.to, Square(0), Square(0)];
    let mut count = 2;
    if let Some(piece) = board.piece_at(mv.from) {
        let from_file = mv.from.0 % 8;
        let to_file = mv.to.0 % 8;
        let rank_base = mv.from.0 - from_file;
        if piece.kind == PieceKind::King && to_file.abs_diff(from_file) == 2 {
            let (rook_from, rook_to) = if to_file == 6 {
                (rank_base + 7, rank_base + 5) // king-side: h-file rook to f-file
            } else {
                (rank_base, rank_base + 3) // queen-side: a-file rook to d-file
            };
            squares[count] = Square(rook_from);
            squares[count + 1] = Square(rook_to);
            count += 2;
        } else if piece.kind == PieceKind::Pawn
            && board.en_passant() == Some(mv.to)
            && board.piece_at(mv.to).is_none()
        {
            // The captured pawn sits on the to-file at the from-rank.
            squares[count] = Square(rank_base + to_file);
            count += 1;
        }
    }
    (squares, count)
}

/// The hand-crafted evaluation of `board` from the side-to-move's perspective,
/// in centipawns. Exposed as the teacher signal for NNUE training.
pub fn hand_crafted_evaluation(board: &Board) -> i32 {
    evaluate_position(board, 0, &EvalParams::default())
}

fn mobility_score(kind: PieceKind, mobility: i32, params: &EvalParams) -> TaperedScore {
    // Centered by a per-piece offset so an average-mobility piece scores near
    // zero (material already accounts for having the piece). The weights are
    // tunable (`EvalParams`); the offsets stay fixed.
    let (weight, offset) = match kind {
        PieceKind::Knight => (params.knight_mobility, 4),
        PieceKind::Bishop => (params.bishop_mobility, 6),
        PieceKind::Rook => (params.rook_mobility, 7),
        PieceKind::Queen => (params.queen_mobility, 13),
        _ => (TaperedScore::new(0, 0), 0),
    };
    let centered = mobility - offset;
    TaperedScore::new(weight.middlegame * centered, weight.endgame * centered)
}

fn evaluate_position(board: &Board, mobility_scale: i32, params: &EvalParams) -> i32 {
    let mut features = EvalFeatures::default();
    let mut white_pawn_files = [0u8; 8];
    let mut black_pawn_files = [0u8; 8];
    let mut white_bishops = 0;
    let mut black_bishops = 0;
    let endgame_phase = endgame_phase(board);

    for idx in 0..64 {
        let square = engine_core::Square(idx);
        let Some(piece) = board.piece_at(square) else {
            continue;
        };
        features.add(
            piece.color,
            tapered_piece_value(piece, params).add(piece_square_bonus(piece, idx)),
            endgame_phase,
        );

        if mobility_scale != 0
            && matches!(
                piece.kind,
                PieceKind::Knight | PieceKind::Bishop | PieceKind::Rook | PieceKind::Queen
            )
        {
            let mobility =
                (board.attacks(square, piece) & !board.occupancy(piece.color)).count_ones() as i32;
            let raw = mobility_score(piece.kind, mobility, params);
            let scaled = TaperedScore::new(
                raw.middlegame * mobility_scale / 100,
                raw.endgame * mobility_scale / 100,
            );
            features.add(piece.color, scaled, endgame_phase);
        }

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

        features.add(
            piece.color,
            TaperedScore::equal(activity_bonus(board, square, piece)),
            endgame_phase,
        );
        if piece.kind == PieceKind::King {
            features.add(
                piece.color,
                TaperedScore::endgame(king_endgame_activity(square)),
                endgame_phase,
            );
        }
    }

    if white_bishops >= 2 {
        features.add(
            Color::White,
            TaperedScore::equal(params.bishop_pair),
            endgame_phase,
        );
    }
    if black_bishops >= 2 {
        features.add(
            Color::Black,
            TaperedScore::equal(params.bishop_pair),
            endgame_phase,
        );
    }

    for idx in 0..64 {
        let square = engine_core::Square(idx);
        let Some(piece) = board.piece_at(square) else {
            continue;
        };
        if piece.kind == PieceKind::Pawn {
            features.add(
                piece.color,
                TaperedScore::equal(pawn_structure_bonus(
                    board,
                    square,
                    piece.color,
                    &white_pawn_files,
                    &black_pawn_files,
                    params,
                )),
                endgame_phase,
            );
        }
        if piece.kind == PieceKind::Rook {
            features.add(
                piece.color,
                TaperedScore::equal(rook_file_bonus(
                    square,
                    piece.color,
                    &white_pawn_files,
                    &black_pawn_files,
                )),
                endgame_phase,
            );
        }
    }

    features.add(
        Color::White,
        TaperedScore::middlegame(king_safety_bonus(board, Color::White)),
        endgame_phase,
    );
    features.add(
        Color::Black,
        TaperedScore::middlegame(king_safety_bonus(board, Color::Black)),
        endgame_phase,
    );
    features.add(
        Color::White,
        TaperedScore::equal(threat_bonus(board, Color::White)),
        endgame_phase,
    );
    features.add(
        Color::Black,
        TaperedScore::equal(threat_bonus(board, Color::Black)),
        endgame_phase,
    );
    features.net(board.side_to_move)
}

fn endgame_phase(board: &Board) -> i32 {
    let mut remaining = 0;
    for index in 0..64 {
        let Some(piece) = board.piece_at(engine_core::Square(index)) else {
            continue;
        };
        remaining += match piece.kind {
            PieceKind::Knight | PieceKind::Bishop => 1,
            PieceKind::Rook => 2,
            PieceKind::Queen => 4,
            PieceKind::Pawn | PieceKind::King => 0,
        };
    }
    24 - remaining.min(24)
}

fn king_endgame_activity(square: engine_core::Square) -> i32 {
    let file_distance = (square.file() as i32 - 3)
        .abs()
        .min((square.file() as i32 - 4).abs());
    let rank_distance = (square.rank() as i32 - 3)
        .abs()
        .min((square.rank() as i32 - 4).abs());
    24 - (file_distance + rank_distance) * 6
}

fn threat_bonus(board: &Board, attacker: Color) -> i32 {
    let mut score = 0;
    for index in 0..64 {
        let square = engine_core::Square(index);
        let Some(piece) = board.piece_at(square) else {
            continue;
        };
        if piece.color != attacker.opposite()
            || !board.is_square_attacked(square, attacker)
            || board.is_square_attacked(square, piece.color)
        {
            continue;
        }
        score += piece_value(piece) / 24;
    }
    score
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

fn is_passed_pawn(board: &Board, square: engine_core::Square, color: Color) -> bool {
    let ranks = match color {
        Color::White => square.rank() + 1..8,
        Color::Black => 0..square.rank(),
    };
    for file in square.file().saturating_sub(1)..=((square.file() + 1).min(7)) {
        for rank in ranks.clone() {
            if board.piece_at(
                engine_core::Square::from_file_rank(file, rank).expect("in bounds"),
            ) == Some(Piece {
                color: color.opposite(),
                kind: PieceKind::Pawn,
            }) {
                return false;
            }
        }
    }
    true
}

fn passed_pawn_extension(board: &Board, mv: ChessMove) -> u8 {
    let Some(piece) = board.piece_at(mv.from) else {
        return 0;
    };
    let advanced = match piece.color {
        Color::White => mv.to.rank() == 6,
        Color::Black => mv.to.rank() == 1,
    };
    u8::from(
        piece.kind == PieceKind::Pawn
            && mv.promotion.is_none()
            && advanced
            && is_passed_pawn(board, mv.from, piece.color),
    )
}

fn pawn_structure_bonus(
    board: &Board,
    square: engine_core::Square,
    color: Color,
    white_pawn_files: &[u8; 8],
    black_pawn_files: &[u8; 8],
    params: &EvalParams,
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

    let is_passed = is_passed_pawn(board, square, color);
    if is_passed {
        let advancement = match color {
            Color::White => square.rank() as i32,
            Color::Black => (7 - square.rank()) as i32,
        };
        score += params.passed_pawn_base + advancement * 10;
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

    let shield_rank = match color {
        Color::White if king_square.rank() < 7 => Some(king_square.rank() + 1),
        Color::Black if king_square.rank() > 0 => Some(king_square.rank() - 1),
        Color::White | Color::Black => None,
    };
    if let Some(shield_rank) = shield_rank {
        for file in king_square.file().saturating_sub(1)..=((king_square.file() + 1).min(7)) {
            if board.piece_at(engine_core::Square::from_file_rank(file, shield_rank).expect("in bounds"))
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
    }

    score
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, atomic::AtomicBool};

    use std::time::Duration;

    use engine_core::{Board, ChessMove, Color, PieceKind};
    use pyrrhic_rs::{Piece as TbPiece, WdlProbeResult};

    use super::{
        Bound, ClockControl, EvalParams, MATE_SCORE, Nnue, OpeningBook, SearchLimits, SearchOptions,
        SearchParams, Searcher, SharedTranspositionTable, SyzygyRootProbe,
        SyzygyTablebases, SyzygyWdl, TaperedScore, TranspositionEntry, TranspositionTable,
        evaluate_position, history_index, late_move_reduction, passed_pawn_extension,
        promotion_from_tablebase, root_tablebase_search_result, static_exchange_evaluation,
        syzygy_score, syzygy_wdl, threat_bonus, late_move_pruning_limit, razor_margin,
        reverse_futility_margin, can_apply_static_pruning, can_try_singular_extension,
        singular_verification_beta,
    };

    #[test]
    fn default_search_options_use_one_thread() {
        assert_eq!(SearchOptions::default().threads, 1);
    }

    /// A spread of positions that exercise every eval term: startpos, an open
    /// middlegame, a closed one, a pawn endgame, a bishop-pair position, and a
    /// passed-pawn position. Used to freeze today's default-eval output so the
    /// `EvalParams` threading stays byte-identical.
    const EVAL_CORPUS: [&str; 6] = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        "rnbqk2r/ppp1bppp/3p1n2/4p3/2PpP3/2NP1N2/PP3PPP/R1BQKB1R w KQkq - 0 1",
        "8/5pk1/6p1/8/8/6P1/5PK1/8 w - - 0 1",
        "2b1k3/8/8/8/8/8/8/2B1KB2 w - - 0 1",
        "4k3/8/8/3P4/8/8/8/4K3 w - - 0 1",
    ];

    /// The tuned default eval's exact scores for `EVAL_CORPUS`, baked from the
    /// SPSA-tuned `EvalParams::default()`. Re-baked when the shipped default eval
    /// flipped to the tuned weights; the default eval must stay byte-identical.
    const FROZEN_TUNED_EVAL_SCORES: [i32; 6] = [168, 155, 129, 42, 396, 172];

    #[test]
    fn default_tuned_eval_is_byte_identical() {
        let scores: Vec<i32> = EVAL_CORPUS
            .iter()
            .map(|fen| {
                let board = Board::from_fen(fen).unwrap();
                evaluate_position(&board, 0, &EvalParams::default())
            })
            .collect();
        // TWO-STEP REBAKE PROBE: reveals the tuned scores in the CI log, then this
        // panic is replaced by an assert against FROZEN_TUNED_EVAL_SCORES.
        assert_eq!(
            scores,
            FROZEN_TUNED_EVAL_SCORES.to_vec(),
            "REBAKE_SCORES {scores:?}"
        );
    }

    #[test]
    fn default_eval_params_match_the_original_constants() {
        let params = EvalParams::default();
        assert_eq!(params.knight, TaperedScore::new(323, 322));
        assert_eq!(params.bishop, TaperedScore::new(334, 322));
        assert_eq!(params.rook, TaperedScore::new(503, 499));
        assert_eq!(params.queen, TaperedScore::new(907, 908));
        assert_eq!(params.knight_mobility, TaperedScore::new(5, 4));
        assert_eq!(params.bishop_mobility, TaperedScore::new(3, 2));
        assert_eq!(params.rook_mobility, TaperedScore::new(3, 4));
        assert_eq!(params.queen_mobility, TaperedScore::new(1, 2));
        assert_eq!(params.bishop_pair, 33);
        assert_eq!(params.passed_pawn_base, 21);
    }

    #[test]
    fn shared_transposition_table_round_trips_across_shards() {
        let table = SharedTranspositionTable::new(4_096);
        // Keys chosen to land in distinct shards (shard = key >> 48).
        for key in [1u64, 1 << 48, (7 << 48) | 9, (63 << 48) | 5] {
            table.store(
                key,
                TranspositionEntry {
                    depth: 4,
                    score: 12,
                    bound: Bound::Exact,
                    best_move: None,
                },
            );
            assert_eq!(table.get(key).map(|entry| entry.score), Some(12));
        }
    }

    #[test]
    fn multi_threaded_search_finds_the_winning_capture() {
        // White rook on h1 can capture the undefended black queen on h4.
        let board = Board::from_fen("4k3/8/8/8/7q/8/8/4K2R w K - 0 1").unwrap();
        let mut options = SearchOptions::default();
        options.threads = 4;
        let mut searcher = Searcher::default();
        searcher.set_options(options);
        let result = searcher.search(
            &board,
            SearchLimits {
                depth: Some(6),
                ..SearchLimits::default()
            },
        );
        assert_eq!(result.best_move.map(|mv| mv.to_uci()), Some("h1h4".to_string()));
    }

    #[test]
    fn single_and_multi_threaded_search_agree_on_a_clear_best_move() {
        let board = Board::from_fen("4k3/8/8/8/7q/8/8/4K2R w K - 0 1").unwrap();
        let limits = SearchLimits {
            depth: Some(6),
            ..SearchLimits::default()
        };

        let mut single = Searcher::default();
        let single_result = single.search(&board, limits.clone());

        let mut options = SearchOptions::default();
        options.threads = 3;
        let mut multi = Searcher::default();
        multi.set_options(options);
        let multi_result = multi.search(&board, limits);

        assert_eq!(single_result.best_move, multi_result.best_move);
    }

    #[test]
    fn history_index_distinguishes_move_destinations_and_promotions() {
        let quiet = ChessMove::from_uci("a2a3").unwrap();
        let different_destination = ChessMove::from_uci("a2a4").unwrap();
        let queen_promotion = ChessMove::from_uci("a7a8q").unwrap();
        let knight_promotion = ChessMove::from_uci("a7a8n").unwrap();

        assert_ne!(history_index(quiet), history_index(different_destination));
        assert_ne!(history_index(queen_promotion), history_index(knight_promotion));
    }

    #[test]
    fn tapered_scores_interpolate_between_phase_endpoints() {
        let score = TaperedScore::new(120, 40);
        assert_eq!(score.interpolate(0), 120);
        assert_eq!(score.interpolate(24), 40);
        assert_eq!(score.interpolate(12), 80);
        assert_eq!(TaperedScore::equal(17).interpolate(9), 17);
    }

    #[test]
    fn quiet_cutoff_records_and_prioritizes_a_counter_move() {
        let previous_move = ChessMove::from_uci("e2e4").unwrap();
        let counter_move = ChessMove::from_uci("d7d5").unwrap();
        let another_quiet_move = ChessMove::from_uci("g8f6").unwrap();
        let mut searcher = Searcher::default();

        searcher.record_cutoff(1, counter_move, 4, Some(previous_move), true);

        assert_eq!(
            searcher.counter_moves[history_index(previous_move)],
            Some(counter_move)
        );
        let board = Board::startpos();
        assert!(
            searcher.move_order_score(&board, counter_move, 0, None, Some(counter_move))
                > searcher.move_order_score(&board, another_quiet_move, 0, None, Some(counter_move))
        );
    }

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
    fn external_stop_signal_cancels_search_before_a_root_move() {
        let stop_signal = Arc::new(AtomicBool::new(true));
        let mut searcher = Searcher::default();
        let result = searcher.search_with_stop_signal(
            &Board::startpos(),
            SearchLimits {
                infinite: true,
                ..SearchLimits::default()
            },
            stop_signal,
        );
        assert_eq!(result.depth, 0);
        assert!(result.best_move.is_none());
    }

    #[test]
    fn transposition_table_keeps_collisions_until_a_cluster_is_full() {
        let mut table = TranspositionTable::new(4);
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
            2,
            TranspositionEntry {
                depth: 3,
                score: 15,
                bound: Bound::Exact,
                best_move: None,
            },
        );
        table.store(
            3,
            TranspositionEntry {
                depth: 3,
                score: 18,
                bound: Bound::Exact,
                best_move: None,
            },
        );
        table.store(
            4,
            TranspositionEntry {
                depth: 3,
                score: 19,
                bound: Bound::Exact,
                best_move: None,
            },
        );

        assert_eq!(table.len(), 4);
        assert_eq!(table.get(1).map(|entry| entry.score), Some(10));
        assert_eq!(table.get(4).map(|entry| entry.score), Some(19));

        table.store(
            5,
            TranspositionEntry {
                depth: 4,
                score: 20,
                bound: Bound::Exact,
                best_move: None,
            },
        );

        assert_eq!(table.len(), 4);
        assert!(table.get(1).is_none());
        assert_eq!(table.get(2).map(|entry| entry.score), Some(15));
        assert_eq!(table.get(5).map(|entry| entry.score), Some(20));
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
    fn versioned_opening_book_selects_a_legal_move_for_the_position() {
        let book = OpeningBook::from_text(
            "rusty-fish-book v1\nrnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\te2e4 d2d4\n",
        )
        .unwrap();
        assert_eq!(
            book.select(&Board::startpos()).map(|mv| mv.to_uci()),
            Some("e2e4".to_string())
        );
    }

    #[test]
    fn v2_opening_book_uses_the_highest_weight_move_at_zero_variety() {
        let book = OpeningBook::from_text(
            "rusty-fish-book v2\nrnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -\te2e4:9 d2d4:1\n",
        )
        .unwrap();
        assert_eq!(
            book.select(&Board::startpos()).map(|mv| mv.to_uci()),
            Some("e2e4".to_string())
        );
    }

    #[test]
    fn v2_opening_book_selects_a_weighted_legal_move_at_nonzero_variety() {
        let book = OpeningBook::from_text(
            "rusty-fish-book v2\nrnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -\te2e4:9 d2d4:1\n",
        )
        .unwrap();
        assert!(matches!(
            book.select_with_variety(&Board::startpos(), 100)
                .map(|mv| mv.to_uci())
                .as_deref(),
            Some("e2e4" | "d2d4")
        ));
    }

    #[test]
    fn search_uses_a_configured_opening_book_before_searching() {
        let mut searcher = Searcher::default();
        searcher.set_opening_book(Some(
            OpeningBook::from_text(
                "rusty-fish-book v1\nrnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\td2d4\n",
            )
            .unwrap(),
        ));
        let result = searcher.search(&Board::startpos(), SearchLimits::default());
        assert_eq!(
            result.best_move.map(|mv| mv.to_uci()),
            Some("d2d4".to_string())
        );
        assert_eq!(result.depth, 0);
    }

    #[test]
    fn search_honors_the_configured_book_variety() {
        const BOOK: &str =
            "rusty-fish-book v2\nrnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -\te2e4:9 d2d4:1\n";
        let expected = OpeningBook::from_text(BOOK)
            .unwrap()
            .select_with_variety(&Board::startpos(), 100)
            .map(|mv| mv.to_uci());

        let mut searcher = Searcher::default();
        searcher.set_options(SearchOptions {
            book_variety: 100,
            ..SearchOptions::default()
        });
        searcher.set_opening_book(Some(OpeningBook::from_text(BOOK).unwrap()));

        let result = searcher.search(&Board::startpos(), SearchLimits::default());
        assert_eq!(result.best_move.map(|mv| mv.to_uci()), expected);
        assert_eq!(result.depth, 0);
    }

    #[test]
    fn search_uses_the_highest_weight_book_move_at_zero_variety() {
        let mut searcher = Searcher::default();
        searcher.set_opening_book(Some(
            OpeningBook::from_text(
                "rusty-fish-book v2\nrnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -\te2e4:9 d2d4:1\n",
            )
            .unwrap(),
        ));

        let result = searcher.search(&Board::startpos(), SearchLimits::default());
        assert_eq!(
            result.best_move.map(|mv| mv.to_uci()),
            Some("e2e4".to_string())
        );
    }

    #[test]
    fn root_tablebase_result_uses_the_exact_move_and_existing_score_scale() {
        let board = Board::startpos();
        let root = SyzygyRootProbe {
            best_move: board.parse_uci_move("e2e4").unwrap(),
            wdl: SyzygyWdl::Win,
            dtz: 1,
        };

        let result = root_tablebase_search_result(root);

        assert_eq!(result.best_move, Some(root.best_move));
        assert_eq!(result.score_cp, syzygy_score(SyzygyWdl::Win, 0));
        assert_eq!(result.nodes, 0);
    }

    #[test]
    fn conservative_pruning_margins_increase_with_depth() {
        let params = SearchParams::default();
        assert!(razor_margin(&params, 2) > razor_margin(&params, 1));
        assert!(reverse_futility_margin(&params, 3) > reverse_futility_margin(&params, 2));
        assert!(late_move_pruning_limit(&params, 3) > late_move_pruning_limit(&params, 2));
    }

    #[test]
    fn incremental_nnue_search_matches_refresh_across_move_types() {
        // In debug builds, evaluate() asserts the incrementally maintained
        // accumulator equals a full refresh at every evaluated node, so these
        // searches exercise castling, en passant, and promotion deltas end to
        // end. A desync would panic the assertion.
        let net = Arc::new(Nnue::from_seed(2024, 16));
        let fens = [
            "r3k2r/pppq1ppp/2np1n2/2b1p3/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
        ];
        for fen in fens {
            let board = Board::from_fen(fen).unwrap();
            let mut searcher = Searcher::default();
            searcher.set_nnue(Some(Arc::clone(&net)));
            let result = searcher.search(
                &board,
                SearchLimits {
                    depth: Some(4),
                    ..SearchLimits::default()
                },
            );
            assert!(result.best_move.is_some(), "search returns a move for {fen}");
        }
    }

    #[test]
    fn nnue_evaluation_overrides_the_handcrafted_score() {
        let board = Board::startpos();
        let mut searcher = Searcher::default();
        let handcrafted = searcher.evaluate(&board);

        let net = Arc::new(Nnue::from_seed(12_345, 32));
        let expected = net.evaluate(&board, board.side_to_move);
        searcher.set_nnue(Some(net));
        assert!(searcher.has_nnue());
        assert_eq!(searcher.evaluate(&board), expected);

        // Removing the network restores the hand-crafted evaluation exactly.
        searcher.set_nnue(None);
        assert!(!searcher.has_nnue());
        assert_eq!(searcher.evaluate(&board), handcrafted);
    }

    #[test]
    fn default_search_params_match_the_original_constants() {
        let params = SearchParams::default();
        assert_eq!(params.aspiration_window, 50);
        assert_eq!(razor_margin(&params, 1), 200);
        assert_eq!(razor_margin(&params, 2), 280);
        assert_eq!(reverse_futility_margin(&params, 1), 190);
        assert_eq!(late_move_pruning_limit(&params, 3), 9);
        assert_eq!(params.null_move_reduction, 3);
        assert_eq!(params.mobility_scale, 100);
    }

    #[test]
    fn custom_search_params_change_margin_scaling() {
        let params = SearchParams {
            razor_margin_base: 200,
            razor_margin_scale: 100,
            ..SearchParams::default()
        };
        assert_eq!(razor_margin(&params, 2), 400);
    }

    #[test]
    fn pruning_policy_excludes_check_and_mate_windows() {
        assert!(!can_apply_static_pruning(2, true, 0, 50, true));
        assert!(!can_apply_static_pruning(
            2,
            false,
            MATE_SCORE - 512,
            MATE_SCORE,
            true,
        ));
        assert!(can_apply_static_pruning(2, false, 0, 50, true));
    }

    #[test]
    fn singular_extension_requires_an_unresolved_exact_non_mate_tt_entry() {
        let entry = TranspositionEntry {
            depth: 5,
            score: 40,
            bound: Bound::Exact,
            best_move: Some(ChessMove::from_uci("e2e4").unwrap()),
        };
        assert!(can_try_singular_extension(6, false, true, entry));
        assert!(!can_try_singular_extension(
            6,
            false,
            true,
            TranspositionEntry { depth: 6, ..entry },
        ));
        assert!(!can_try_singular_extension(6, true, true, entry));
    }

    #[test]
    fn singular_extension_uses_a_fixed_verification_margin() {
        assert_eq!(singular_verification_beta(80), 48);
    }

    #[test]
    fn syzygy_loader_reports_a_missing_tablebase_path_without_affecting_search() {
        assert!(SyzygyTablebases::load("missing-syzygy-tablebases", 7).is_err());
    }

    #[test]
    fn checksummed_kqvk_corpus_returns_exact_win_and_dtz() {
        let Ok(path) = std::env::var("RUSTY_FISH_SYZYGY_TEST_DIR") else {
            return;
        };
        let tables = SyzygyTablebases::load(&path, 3).expect("load checksummed KQvK corpus");
        let board = Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1").unwrap();
        let root = tables.probe_root(&board).expect("KQvK root probe");
        assert_eq!(root.wdl, SyzygyWdl::Win);
        assert_eq!(root.dtz, 1);
    }

    #[test]
    fn tablebase_promotion_conversion_matches_uci_piece_kinds() {
        assert_eq!(
            promotion_from_tablebase(TbPiece::Queen),
            Some(PieceKind::Queen)
        );
        assert_eq!(promotion_from_tablebase(TbPiece::Pawn), None);
    }

    #[test]
    fn tablebase_wdl_categories_keep_cursed_results_on_the_winning_side() {
        assert_eq!(syzygy_wdl(WdlProbeResult::CursedWin), SyzygyWdl::Win);
        assert_eq!(syzygy_wdl(WdlProbeResult::BlessedLoss), SyzygyWdl::Loss);
    }

    #[test]
    fn static_exchange_evaluation_rejects_a_losing_queen_capture() {
        let board = Board::from_fen("3rk3/8/8/3p4/3Q4/8/8/4K3 w - - 0 1").unwrap();
        let mv = board.parse_uci_move("d4d5").unwrap();
        assert!(static_exchange_evaluation(&board, mv) < 0);
    }

    #[test]
    fn evaluation_prefers_passed_pawn_and_bishop_pair() {
        let white_edge = Board::from_fen("4k3/8/8/3P4/8/8/4BB2/4K3 w - - 0 1").unwrap();
        let black_edge = Board::from_fen("4k3/4bb2/8/8/3p4/8/8/4K3 b - - 0 1").unwrap();
        assert!(evaluate_position(&white_edge, 0, &EvalParams::default()) > 0);
        assert!(evaluate_position(&black_edge, 0, &EvalParams::default()) > 0);
    }

    #[test]
    fn passed_pawn_extension_requires_an_advanced_unblocked_pawn_push() {
        let white = Board::from_fen("4k3/8/3P4/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(
            passed_pawn_extension(&white, white.parse_uci_move("d6d7").unwrap()),
            1
        );

        let black = Board::from_fen("4k3/8/8/8/8/3p4/8/4K3 b - - 0 1").unwrap();
        assert_eq!(
            passed_pawn_extension(&black, black.parse_uci_move("d3d2").unwrap()),
            1
        );

        let blocked = Board::from_fen("4k3/2p5/3P4/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(
            passed_pawn_extension(&blocked, blocked.parse_uci_move("d6d7").unwrap()),
            0
        );

        let promotion = Board::from_fen("4k3/3P4/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(
            passed_pawn_extension(&promotion, promotion.parse_uci_move("d7d8q").unwrap()),
            0
        );
    }

    #[test]
    fn endgame_evaluation_rewards_an_active_king() {
        let active = Board::from_fen("4k3/8/8/8/4K3/8/4P3/8 w - - 0 1").unwrap();
        let passive = Board::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
        assert!(
            evaluate_position(&active, 0, &EvalParams::default())
                > evaluate_position(&passive, 0, &EvalParams::default())
        );
    }

    #[test]
    fn mobility_scale_rewards_the_more_active_side() {
        // White knight on d4 (8 targets), black knight on a8 (2 targets).
        let board = Board::from_fen("n6k/8/8/8/3N4/8/8/7K w - - 0 1").unwrap();
        let off = evaluate_position(&board, 0, &EvalParams::default());
        let on = evaluate_position(&board, 100, &EvalParams::default());
        // White is to move, so a positive mobility difference raises the score.
        assert!(on > off, "mobility should favor the side with the more active knight: on={on} off={off}");
    }

    #[test]
    fn king_safety_handles_a_king_on_the_board_edge() {
        let board = Board::from_fen("6K1/8/8/8/8/8/8/4k3 w - - 0 1").unwrap();
        let _ = evaluate_position(&board, 0, &EvalParams::default());
    }

    #[test]
    fn evaluation_rewards_attacking_an_undefended_queen() {
        let threatened = Board::from_fen("4k3/8/8/8/3q4/5N2/8/4K3 w - - 0 1").unwrap();
        let safe = Board::from_fen("4k3/8/8/8/3q4/7N/8/4K3 w - - 0 1").unwrap();
        assert!(threat_bonus(&threatened, Color::White) > 0);
        assert_eq!(threat_bonus(&safe, Color::White), 0);
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
        assert!(evaluate_position(&white, 0, &EvalParams::default()) > 0);
        assert!(evaluate_position(&black, 0, &EvalParams::default()) > 0);
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
