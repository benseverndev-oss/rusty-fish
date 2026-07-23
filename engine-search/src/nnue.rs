//! NNUE (efficiently updatable neural network) evaluation.
//!
//! This module provides the inference machinery for a perspective network: the
//! feature encoding, an accumulator with both a from-scratch refresh and
//! incremental update primitives, a quantised forward pass, and a versioned
//! file format. It is deliberately independent of a trained network — a
//! deterministic seeded generator lets tests and CI exercise the whole pipeline
//! — and the engine only uses it when a network is explicitly loaded.

use std::sync::{Arc, LazyLock};

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
        accumulate(self.perspective_mut(perspective), column, true);
    }

    /// Removes a piece's contribution from one perspective's accumulator. The
    /// inverse of [`Accumulator::add_feature`]; the search's make/unmake hook
    /// drives both to keep the accumulator in sync incrementally.
    pub fn remove_feature(&mut self, net: &Nnue, perspective: Color, piece: Piece, square: Square) {
        let feature = feature_index(perspective, piece, square);
        let hidden = net.hidden;
        let column = &net.feature_weights[feature * hidden..feature * hidden + hidden];
        accumulate(self.perspective_mut(perspective), column, false);
    }

    /// Applies a batch of square changes to both perspectives in a single fused
    /// pass each: every accumulator lane is read and written once no matter how
    /// many features the move touched. Each change is `(square, removed, added)`
    /// — the piece leaving the square (subtracted) and the piece arriving (added).
    /// The search drives make with `(square, old, new)` and unmake with the pair
    /// swapped. Bit-exact with repeated [`Accumulator::add_feature`]/
    /// [`Accumulator::remove_feature`] calls.
    pub fn apply_changes(
        &mut self,
        net: &Nnue,
        changes: &[(Square, Option<Piece>, Option<Piece>)],
    ) {
        let hidden = net.hidden;
        for perspective in [Color::White, Color::Black] {
            let mut adds = [0usize; 4];
            let mut add_count = 0;
            let mut subs = [0usize; 4];
            let mut sub_count = 0;
            for &(square, removed, added) in changes {
                if let Some(piece) = removed {
                    subs[sub_count] = feature_index(perspective, piece, square);
                    sub_count += 1;
                }
                if let Some(piece) = added {
                    adds[add_count] = feature_index(perspective, piece, square);
                    add_count += 1;
                }
            }
            apply_feature_deltas(
                &net.feature_weights,
                hidden,
                self.perspective_mut(perspective),
                &adds[..add_count],
                &subs[..sub_count],
            );
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
        output += crelu_dot(own, &self.output_weights[..self.hidden]);
        output += crelu_dot(opponent, &self.output_weights[self.hidden..]);
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

/// The engine's default NNUE, compiled into the binary. Parsed once and shared.
static BUNDLED_NETWORK: LazyLock<Arc<Nnue>> = LazyLock::new(|| {
    Arc::new(
        Nnue::from_bytes(include_bytes!("../../assets/nnue/rusty-fish-net.rfnn"))
            .expect("bundled NNUE asset is a valid RFNN network"),
    )
});

/// A shared handle to the bundled default network.
pub fn bundled_network() -> Arc<Nnue> {
    BUNDLED_NETWORK.clone()
}

fn clipped_relu(value: i32) -> i32 {
    value.clamp(0, ACTIVATION_CLIP)
}

// --- Vectorised inner loops -------------------------------------------------
//
// The feature-transformer update and the output dot product are the hot per-node
// cost of NNUE evaluation. Both fold i16 weights into i32 sums, and because each
// term cannot overflow i32 (a clipped activation is at most 127 and a weight fits
// i16), the running sum is order-independent modulo 2^32. That makes the AVX2
// paths *bit-exact* to the scalar ones — the incremental-vs-refresh and
// forward-determinism tests hold them to it. AVX2 is chosen at runtime; every
// other target (and any non-AVX2 x86) uses the scalar fallback.

/// Adds (`add`) or subtracts the i16 `weights` column into the i32 `accumulator`,
/// lane for lane. `accumulator` and `weights` must have equal length.
#[inline]
fn accumulate(accumulator: &mut [i32], weights: &[i16], add: bool) {
    debug_assert_eq!(accumulator.len(), weights.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 is available on this CPU and the slices share a length.
            unsafe {
                accumulate_avx2(accumulator, weights, add);
            }
            return;
        }
    }
    accumulate_scalar(accumulator, weights, add);
}

/// Applies several feature columns to one accumulator in a single fused pass:
/// `add` columns are summed in, `sub` columns subtracted. Fusing keeps the
/// accumulator resident (one load + one store per lane) instead of re-streaming
/// it once per feature. `adds`/`subs` hold feature indices into `weights`.
#[inline]
fn apply_feature_deltas(
    weights: &[i16],
    hidden: usize,
    accumulator: &mut [i32],
    adds: &[usize],
    subs: &[usize],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 is available; feature indices index full `hidden`-length
            // columns within `weights`, matched to the accumulator length.
            unsafe {
                apply_feature_deltas_avx2(weights, hidden, accumulator, adds, subs);
            }
            return;
        }
    }
    for &feature in adds {
        let column = &weights[feature * hidden..feature * hidden + hidden];
        for (value, weight) in accumulator.iter_mut().zip(column) {
            *value += i32::from(*weight);
        }
    }
    for &feature in subs {
        let column = &weights[feature * hidden..feature * hidden + hidden];
        for (value, weight) in accumulator.iter_mut().zip(column) {
            *value -= i32::from(*weight);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_feature_deltas_avx2(
    weights: &[i16],
    hidden: usize,
    accumulator: &mut [i32],
    adds: &[usize],
    subs: &[usize],
) {
    use core::arch::x86_64::*;
    let chunks = hidden / 8;
    let acc_ptr = accumulator.as_mut_ptr();
    let weight_ptr = weights.as_ptr();
    for chunk in 0..chunks {
        let offset = chunk * 8;
        let mut value = _mm256_loadu_si256(acc_ptr.add(offset) as *const __m256i);
        for &feature in adds {
            let column = _mm_loadu_si128(weight_ptr.add(feature * hidden + offset) as *const __m128i);
            value = _mm256_add_epi32(value, _mm256_cvtepi16_epi32(column));
        }
        for &feature in subs {
            let column = _mm_loadu_si128(weight_ptr.add(feature * hidden + offset) as *const __m128i);
            value = _mm256_sub_epi32(value, _mm256_cvtepi16_epi32(column));
        }
        _mm256_storeu_si256(acc_ptr.add(offset) as *mut __m256i, value);
    }
    for index in (chunks * 8)..hidden {
        let mut value = *accumulator.get_unchecked(index);
        for &feature in adds {
            value += i32::from(*weights.get_unchecked(feature * hidden + index));
        }
        for &feature in subs {
            value -= i32::from(*weights.get_unchecked(feature * hidden + index));
        }
        *accumulator.get_unchecked_mut(index) = value;
    }
}

fn accumulate_scalar(accumulator: &mut [i32], weights: &[i16], add: bool) {
    if add {
        for (value, weight) in accumulator.iter_mut().zip(weights) {
            *value += i32::from(*weight);
        }
    } else {
        for (value, weight) in accumulator.iter_mut().zip(weights) {
            *value -= i32::from(*weight);
        }
    }
}

/// Clipped-ReLU dot product: `sum(clip(activation) * weight)` over equal-length
/// slices, returning the i32 sum (wrapping, matching the scalar accumulation).
#[inline]
fn crelu_dot(activations: &[i32], weights: &[i16]) -> i32 {
    debug_assert_eq!(activations.len(), weights.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 is available on this CPU and the slices share a length.
            return unsafe { crelu_dot_avx2(activations, weights) };
        }
    }
    crelu_dot_scalar(activations, weights)
}

fn crelu_dot_scalar(activations: &[i32], weights: &[i16]) -> i32 {
    let mut sum = 0i32;
    for (activation, weight) in activations.iter().zip(weights) {
        sum += clipped_relu(*activation) * i32::from(*weight);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_avx2(accumulator: &mut [i32], weights: &[i16], add: bool) {
    use core::arch::x86_64::*;
    let len = accumulator.len();
    let chunks = len / 8;
    let acc_ptr = accumulator.as_mut_ptr();
    let weight_ptr = weights.as_ptr();
    for chunk in 0..chunks {
        let offset = chunk * 8;
        let acc = _mm256_loadu_si256(acc_ptr.add(offset) as *const __m256i);
        // Load 8 i16 weights and sign-extend them to 8 i32 lanes.
        let widened = _mm256_cvtepi16_epi32(_mm_loadu_si128(weight_ptr.add(offset) as *const __m128i));
        let updated = if add {
            _mm256_add_epi32(acc, widened)
        } else {
            _mm256_sub_epi32(acc, widened)
        };
        _mm256_storeu_si256(acc_ptr.add(offset) as *mut __m256i, updated);
    }
    for index in (chunks * 8)..len {
        let weight = i32::from(*weights.get_unchecked(index));
        let value = accumulator.get_unchecked_mut(index);
        if add {
            *value += weight;
        } else {
            *value -= weight;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn crelu_dot_avx2(activations: &[i32], weights: &[i16]) -> i32 {
    use core::arch::x86_64::*;
    let len = activations.len();
    let chunks = len / 8;
    let act_ptr = activations.as_ptr();
    let weight_ptr = weights.as_ptr();
    let zero = _mm256_setzero_si256();
    let clip = _mm256_set1_epi32(ACTIVATION_CLIP);
    let mut acc = _mm256_setzero_si256();
    for chunk in 0..chunks {
        let offset = chunk * 8;
        let activation = _mm256_loadu_si256(act_ptr.add(offset) as *const __m256i);
        let clipped = _mm256_min_epi32(_mm256_max_epi32(activation, zero), clip);
        let widened = _mm256_cvtepi16_epi32(_mm_loadu_si128(weight_ptr.add(offset) as *const __m128i));
        let product = _mm256_mullo_epi32(clipped, widened);
        acc = _mm256_add_epi32(acc, product);
    }
    // Horizontal sum of the eight i32 lanes.
    let low = _mm256_castsi256_si128(acc);
    let high = _mm256_extracti128_si256(acc, 1);
    let mut summed = _mm_add_epi32(low, high);
    summed = _mm_hadd_epi32(summed, summed);
    summed = _mm_hadd_epi32(summed, summed);
    let mut sum = _mm_cvtsi128_si32(summed);
    for index in (chunks * 8)..len {
        sum += clipped_relu(*activations.get_unchecked(index)) * i32::from(*weights.get_unchecked(index));
    }
    sum
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
    use super::{bundled_network, feature_index, Accumulator, Nnue, INPUT_DIMENSION};
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
    fn bundled_network_round_trips() {
        let net = bundled_network();
        assert_eq!(net.hidden(), 1024); // the shipped champion's hidden width
        // Re-serialising the parsed net reproduces the committed asset bytes exactly.
        assert_eq!(
            net.to_bytes(),
            include_bytes!("../../assets/nnue/rusty-fish-net.rfnn").to_vec()
        );
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
