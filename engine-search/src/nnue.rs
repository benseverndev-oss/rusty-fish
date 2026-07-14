//! NNUE (efficiently updatable neural network) evaluation.
//!
//! This module provides the inference machinery for a perspective network: the
//! feature encoding, an accumulator with both a from-scratch refresh and
//! incremental update primitives, a quantised forward pass, and a versioned
//! file format. It is deliberately independent of a trained network — a
//! deterministic seeded generator lets tests and CI exercise the whole pipeline
//! — and the engine only uses it when a network is explicitly loaded.

use engine_core::{Board, Color, Piece, PieceKind, Square};

/// Number of input features: (own / their) x 6 piece kinds x 64 squares.
pub const INPUT_DIMENSION: usize = 2 * 6 * 64;

const MAGIC: &[u8; 4] = b"RFNN";
const FORMAT_VERSION: u32 = 1;

/// Clipped-ReLU upper bound for accumulator activations.
const ACTIVATION_CLIP: i32 = 127;
/// Divisor applied to the integer output to reach centipawns.
const OUTPUT_SCALE: i32 = 64;
/// NNUE scores are clamped to this magnitude so they can never look like a mate
/// score to the search.
const EVAL_CLAMP: i32 = 20_000;

fn piece_kind_index(kind: PieceKind) -> usize {
    match kind {
        PieceKind::Pawn => 0,
        PieceKind::Knight => 1,
        PieceKind::Bishop => 2,
        PieceKind::Rook => 3,
        PieceKind::Queen => 4,
        PieceKind::King => 5,
    }
}

/// Maps a piece on a square to its feature index for a given perspective. The
/// square is vertically flipped for the black perspective and the colour is
/// taken relative to the perspective, so each side sees "my pieces" vs "their
/// pieces" consistently.
pub fn feature_index(perspective: Color, piece: Piece, square: Square) -> usize {
    let relative_square = match perspective {
        Color::White => usize::from(square.0),
        Color::Black => usize::from(square.0 ^ 56),
    };
    let relative_color = usize::from(piece.color != perspective);
    (relative_color * 6 + piece_kind_index(piece.kind)) * 64 + relative_square
}

/// Returns the active feature indices for `board` from `perspective` (one per
/// piece on the board). Exposed for the NNUE trainer.
pub fn active_features(board: &Board, perspective: Color) -> Vec<usize> {
    let mut features = Vec::with_capacity(32);
    for index in 0..64 {
        let square = Square(index);
        if let Some(piece) = board.piece_at(square) {
            features.push(feature_index(perspective, piece, square));
        }
    }
    features
}

/// Two per-perspective accumulators. Values are summed into `i32` so a full
/// board can never overflow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Accumulator {
    white: Vec<i32>,
    black: Vec<i32>,
}

impl Accumulator {
    /// A zeroed accumulator seeded with the feature-transformer bias.
    fn empty(net: &Nnue) -> Self {
        let bias: Vec<i32> = net.feature_bias.iter().map(|value| i32::from(*value)).collect();
        Self {
            white: bias.clone(),
            black: bias,
        }
    }

    fn perspective_mut(&mut self, perspective: Color) -> &mut [i32] {
        match perspective {
            Color::White => &mut self.white,
            Color::Black => &mut self.black,
        }
    }

    /// Adds a piece's contribution to one perspective's accumulator.
    pub fn add_feature(&mut self, net: &Nnue, perspective: Color, piece: Piece, square: Square) {
        let feature = feature_index(perspective, piece, square);
        let hidden = net.hidden;
        let column = &net.feature_weights[feature * hidden..feature * hidden + hidden];
        for (value, weight) in self.perspective_mut(perspective).iter_mut().zip(column) {
            *value += i32::from(*weight);
        }
    }

    /// Removes a piece's contribution from one perspective's accumulator. The
    /// inverse of [`Accumulator::add_feature`]; the search's make/unmake hook
    /// drives both to keep the accumulator in sync incrementally.
    pub fn remove_feature(&mut self, net: &Nnue, perspective: Color, piece: Piece, square: Square) {
        let feature = feature_index(perspective, piece, square);
        let hidden = net.hidden;
        let column = &net.feature_weights[feature * hidden..feature * hidden + hidden];
        for (value, weight) in self.perspective_mut(perspective).iter_mut().zip(column) {
            *value -= i32::from(*weight);
        }
    }

    /// Rebuilds both accumulators from scratch for the given board.
    pub fn refresh(net: &Nnue, board: &Board) -> Self {
        let mut accumulator = Self::empty(net);
        for index in 0..64 {
            let square = Square(index);
            if let Some(piece) = board.piece_at(square) {
                accumulator.add_feature(net, Color::White, piece, square);
                accumulator.add_feature(net, Color::Black, piece, square);
            }
        }
        accumulator
    }
}

/// A quantised perspective network: immutable integer weights only, so it is
/// cheap to share across search threads behind an `Arc`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nnue {
    hidden: usize,
    feature_weights: Vec<i16>,
    feature_bias: Vec<i16>,
    output_weights: Vec<i16>,
    output_bias: i32,
}

impl Nnue {
    /// The hidden layer width of this network.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Evaluates the position from `side_to_move`'s perspective, in centipawns.
    pub fn evaluate(&self, board: &Board, side_to_move: Color) -> i32 {
        let accumulator = Accumulator::refresh(self, board);
        self.forward(&accumulator, side_to_move)
    }

    /// Evaluates from a prebuilt (incrementally maintained) accumulator, in
    /// centipawns from `side_to_move`'s perspective.
    pub fn evaluate_with(&self, accumulator: &Accumulator, side_to_move: Color) -> i32 {
        self.forward(accumulator, side_to_move)
    }

    fn forward(&self, accumulator: &Accumulator, side_to_move: Color) -> i32 {
        let (own, opponent) = match side_to_move {
            Color::White => (&accumulator.white, &accumulator.black),
            Color::Black => (&accumulator.black, &accumulator.white),
        };
        let mut output = self.output_bias;
        for (activation, weight) in own.iter().zip(&self.output_weights[..self.hidden]) {
            output += clipped_relu(*activation) * i32::from(*weight);
        }
        for (activation, weight) in opponent.iter().zip(&self.output_weights[self.hidden..]) {
            output += clipped_relu(*activation) * i32::from(*weight);
        }
        (output / OUTPUT_SCALE).clamp(-EVAL_CLAMP, EVAL_CLAMP)
    }

    /// Serialises the network to the `RFNN` little-endian format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(self.hidden as u32).to_le_bytes());
        for weight in &self.feature_weights {
            bytes.extend_from_slice(&weight.to_le_bytes());
        }
        for weight in &self.feature_bias {
            bytes.extend_from_slice(&weight.to_le_bytes());
        }
        for weight in &self.output_weights {
            bytes.extend_from_slice(&weight.to_le_bytes());
        }
        bytes.extend_from_slice(&self.output_bias.to_le_bytes());
        bytes
    }

    /// Parses a network from the `RFNN` format, rejecting malformed data.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut cursor = Cursor::new(bytes);
        let magic = cursor.take(4)?;
        if magic != MAGIC {
            return Err("not an RFNN network (bad magic)".to_string());
        }
        let version = cursor.read_u32()?;
        if version != FORMAT_VERSION {
            return Err(format!("unsupported RFNN version {version}"));
        }
        let hidden = cursor.read_u32()? as usize;
        if hidden == 0 {
            return Err("RFNN hidden size must be non-zero".to_string());
        }
        let feature_weights = cursor.read_i16s(INPUT_DIMENSION * hidden)?;
        let feature_bias = cursor.read_i16s(hidden)?;
        let output_weights = cursor.read_i16s(2 * hidden)?;
        let output_bias = cursor.read_i32()?;
        if !cursor.is_empty() {
            return Err("trailing bytes after RFNN network".to_string());
        }
        Ok(Self {
            hidden,
            feature_weights,
            feature_bias,
            output_weights,
            output_bias,
        })
    }

    /// Loads a network from a file path.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("failed to read NNUE file {path}: {error}"))?;
        Self::from_bytes(&bytes)
    }

    /// Builds a network from explicit quantised parameters, validating that the
    /// weight vectors have the lengths implied by `hidden`. Used by the trainer
    /// to assemble a network after fitting.
    pub fn from_parameters(
        hidden: usize,
        feature_weights: Vec<i16>,
        feature_bias: Vec<i16>,
        output_weights: Vec<i16>,
        output_bias: i32,
    ) -> Result<Self, String> {
        if hidden == 0 {
            return Err("hidden size must be non-zero".to_string());
        }
        if feature_weights.len() != INPUT_DIMENSION * hidden {
            return Err(format!(
                "feature_weights must have {} entries",
                INPUT_DIMENSION * hidden
            ));
        }
        if feature_bias.len() != hidden {
            return Err(format!("feature_bias must have {hidden} entries"));
        }
        if output_weights.len() != 2 * hidden {
            return Err(format!("output_weights must have {} entries", 2 * hidden));
        }
        Ok(Self {
            hidden,
            feature_weights,
            feature_bias,
            output_weights,
            output_bias,
        })
    }

    /// Builds a deterministic network with small pseudo-random weights. This is
    /// for tests and pipeline exercise only — it is not a trained network.
    pub fn from_seed(seed: u64, hidden: usize) -> Self {
        assert!(hidden > 0, "hidden size must be non-zero");
        let mut rng = SplitMix64::new(seed);
        let feature_weights = (0..INPUT_DIMENSION * hidden)
            .map(|_| rng.small_weight(8))
            .collect();
        let feature_bias = (0..hidden).map(|_| rng.small_weight(4)).collect();
        let output_weights = (0..2 * hidden).map(|_| rng.small_weight(16)).collect();
        let output_bias = i32::from(rng.small_weight(32));
        Self {
            hidden,
            feature_weights,
            feature_bias,
            output_weights,
            output_bias,
        }
    }
}

fn clipped_relu(value: i32) -> i32 {
    value.clamp(0, ACTIVATION_CLIP)
}

/// Minimal cursor over a byte slice for the little-endian loader.
struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        let end = self
            .offset
            .checked_add(count)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| "RFNN network is truncated".to_string())?;
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let bytes = self.take(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i32(&mut self) -> Result<i32, String> {
        let bytes = self.take(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i16s(&mut self, count: usize) -> Result<Vec<i16>, String> {
        let bytes = self.take(count * 2)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect())
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

/// SplitMix64 — a tiny deterministic PRNG used only to synthesise test networks.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A small signed weight in `[-magnitude, magnitude]`.
    fn small_weight(&mut self, magnitude: i64) -> i16 {
        let span = magnitude * 2 + 1;
        let value = (self.next_u64() % span as u64) as i64 - magnitude;
        value as i16
    }
}

#[cfg(test)]
mod tests {
    use super::{feature_index, Accumulator, Nnue, INPUT_DIMENSION};
    use engine_core::{Board, Color, Piece, PieceKind, Square};

    fn white_pawn() -> Piece {
        Piece {
            color: Color::White,
            kind: PieceKind::Pawn,
        }
    }

    #[test]
    fn feature_indices_stay_in_range_and_flip_by_perspective() {
        let piece = white_pawn();
        for index in 0..64u8 {
            let square = Square(index);
            assert!(feature_index(Color::White, piece, square) < INPUT_DIMENSION);
            assert!(feature_index(Color::Black, piece, square) < INPUT_DIMENSION);
        }
        // A white pawn on a2 seen from White mirrors a black pawn on a7 seen
        // from Black: both the square (vertical flip) and the colour are taken
        // relative to the perspective.
        let a2 = Square(8);
        let a7 = Square(48);
        let black_pawn = Piece {
            color: Color::Black,
            kind: PieceKind::Pawn,
        };
        assert_eq!(
            feature_index(Color::White, piece, a2),
            feature_index(Color::Black, black_pawn, a7),
        );
    }

    #[test]
    fn incremental_updates_match_a_full_refresh() {
        let net = Nnue::from_seed(2024, 32);
        let board = Board::startpos();
        let refreshed = Accumulator::refresh(&net, &board);

        // Build the same accumulator incrementally, piece by piece.
        let mut incremental = Accumulator::empty_for_test(&net);
        for index in 0..64u8 {
            let square = Square(index);
            if let Some(piece) = board.piece_at(square) {
                incremental.add_feature(&net, Color::White, piece, square);
                incremental.add_feature(&net, Color::Black, piece, square);
            }
        }
        assert_eq!(incremental, refreshed);

        // Adding then removing a feature is a no-op.
        let mut toggled = refreshed.clone();
        let piece = white_pawn();
        let square = Square(20);
        toggled.add_feature(&net, Color::White, piece, square);
        toggled.remove_feature(&net, Color::White, piece, square);
        assert_eq!(toggled, refreshed);
    }

    #[test]
    fn forward_pass_is_deterministic_and_bounded() {
        let net = Nnue::from_seed(7, 64);
        let board = Board::startpos();
        let first = net.evaluate(&board, Color::White);
        let second = net.evaluate(&board, Color::White);
        assert_eq!(first, second);
        assert!(first.abs() <= 20_000);
    }

    #[test]
    fn network_round_trips_through_its_byte_format() {
        let net = Nnue::from_seed(99, 48);
        let restored = Nnue::from_bytes(&net.to_bytes()).expect("valid network round-trips");
        assert_eq!(net, restored);
    }

    #[test]
    fn loader_rejects_malformed_networks() {
        assert!(Nnue::from_bytes(b"not a network").is_err());
        let mut bytes = Nnue::from_seed(1, 16).to_bytes();
        bytes.push(0); // trailing garbage
        assert!(Nnue::from_bytes(&bytes).is_err());
        let truncated = &Nnue::from_seed(1, 16).to_bytes()[..10];
        assert!(Nnue::from_bytes(truncated).is_err());
    }

    impl Accumulator {
        fn empty_for_test(net: &Nnue) -> Self {
            Self::empty(net)
        }
    }
}
