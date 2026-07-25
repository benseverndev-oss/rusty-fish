//! Learned move-ordering policy (Phase 4): a tiny MLP that predicts, from a move's
//! *ordering-time* context, the probability the move causes a beta cutoff — i.e. that
//! it deserves to be searched first. Loaded from the `RFPO` format the in-process Rust
//! trainer (`engine-bench`'s `train-policy`) exports. Inference is the same shape as
//! the learned-LMR net (a handful of FLOPs), cheap enough to score every move.
//!
//! This module is the inference/format half only. Wiring the policy into
//! `move_order_score` (as `order = classical + learned_correction`, mirroring learned
//! LMR's `reduction = classical + correction`) is a separate, gated change; with no
//! policy installed the search is byte-for-byte the classical orderer.
//!
//! RFPO v1 layout (little-endian), identical in shape to RFLM but with its own magic so
//! the two model files can never be loaded into the wrong slot: magic `b"RFPO"` | u32
//! version=1 | u32 input_dim | u32 hidden | mean[input_dim] f32 | scale[input_dim] f32
//! | w1[hidden*input_dim] f32 (row-major [hidden, input]) | b1[hidden] f32 | w2[hidden]
//! f32 | b2 f32. `input_dim` is checked against [`POLICY_FEATURES`] on load, so widening
//! the feature set and swapping any bundled asset have to happen in the same commit.

const MAGIC: &[u8; 4] = b"RFPO";
const VERSION: u32 = 1;

/// The ordering-time context features, in the exact order the trainer uses (matching
/// `telemetry::POLICY_FEATURE_COLUMNS`): depth, ply, pv_node, node_in_check,
/// static_eval, is_quiet, is_capture, is_promotion, is_tt_move, is_killer, is_counter,
/// history_score, tt_depth, order_score, see, mover_piece, captured_piece.
///
/// These are exactly the signals available to the move orderer *before* a move is
/// searched, and cheap to compute there. Deliberately excluded: `move_index` (the
/// classical rank — the thing being improved, so circular), and `reduction` /
/// `extension` / `gives_check` (decided or costly only after ordering).
pub const POLICY_FEATURES: usize = 17;

/// Indices of the unbounded feature columns, clamped before standardization exactly as
/// the trainer clamps them. Kept in lockstep with `telemetry::POLICY_FEATURE_CLAMPS`
/// (checked by a test there): the trainer fits its standardization on clamped values,
/// so an unclamped mate eval, runaway history entry, or the huge TT-move order score
/// would land far outside the distribution the stored mean/scale describe.
const CLAMPS: [(usize, f32); 4] = [
    (4, 2_000.0),      // static_eval
    (11, 20_000.0),    // history_score
    (13, 2_000_000.0), // order_score (TT move ~= 2_000_000, kept distinct)
    (14, 2_000.0),     // see
];

/// A loaded move-ordering policy. Immutable after load, cheap to share behind `Arc`.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyModel {
    hidden: usize,
    mean: [f32; POLICY_FEATURES],
    scale: [f32; POLICY_FEATURES],
    w1: Vec<f32>, // [hidden * POLICY_FEATURES], row-major (hidden rows of POLICY_FEATURES)
    b1: Vec<f32>, // [hidden]
    w2: Vec<f32>, // [hidden]
    b2: f32,
}

fn take<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], String> {
    let end = *cursor + len;
    let slice = bytes.get(*cursor..end).ok_or_else(|| "RFPO truncated".to_string())?;
    *cursor = end;
    Ok(slice)
}

fn read_u32(bytes: &[u8], cursor: &mut usize) -> Result<u32, String> {
    Ok(u32::from_le_bytes(take(bytes, cursor, 4)?.try_into().unwrap()))
}

fn read_f32(bytes: &[u8], cursor: &mut usize) -> Result<f32, String> {
    Ok(f32::from_le_bytes(take(bytes, cursor, 4)?.try_into().unwrap()))
}

fn read_f32s(bytes: &[u8], cursor: &mut usize, count: usize) -> Result<Vec<f32>, String> {
    (0..count).map(|_| read_f32(bytes, cursor)).collect()
}

impl PolicyModel {
    /// Parse an RFPO v1 byte buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut cursor = 0usize;
        if take(bytes, &mut cursor, 4)? != MAGIC {
            return Err("not an RFPO file (bad magic)".to_string());
        }
        let version = read_u32(bytes, &mut cursor)?;
        if version != VERSION {
            return Err(format!("unsupported RFPO version {version}"));
        }
        let input_dim = read_u32(bytes, &mut cursor)? as usize;
        if input_dim != POLICY_FEATURES {
            return Err(format!("RFPO input_dim {input_dim} != {POLICY_FEATURES}"));
        }
        let hidden = read_u32(bytes, &mut cursor)? as usize;
        if hidden == 0 {
            return Err("RFPO hidden must be >= 1".to_string());
        }
        let mean_v = read_f32s(bytes, &mut cursor, POLICY_FEATURES)?;
        let scale_v = read_f32s(bytes, &mut cursor, POLICY_FEATURES)?;
        let w1 = read_f32s(bytes, &mut cursor, hidden * POLICY_FEATURES)?;
        let b1 = read_f32s(bytes, &mut cursor, hidden)?;
        let w2 = read_f32s(bytes, &mut cursor, hidden)?;
        let b2 = read_f32(bytes, &mut cursor)?;
        let mut mean = [0f32; POLICY_FEATURES];
        let mut scale = [0f32; POLICY_FEATURES];
        mean.copy_from_slice(&mean_v);
        scale.copy_from_slice(&scale_v);
        Ok(Self { hidden, mean, scale, w1, b1, w2, b2 })
    }

    /// Load an RFPO model from a file.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|error| format!("failed to read {path}: {error}"))?;
        Self::from_bytes(&bytes)
    }

    /// Assemble a model directly from trained parameters (used by the trainer to
    /// serialize and to round-trip through [`to_bytes`](Self::to_bytes)). `w1` is
    /// row-major `[hidden, POLICY_FEATURES]`.
    pub fn from_parameters(
        mean: [f32; POLICY_FEATURES],
        scale: [f32; POLICY_FEATURES],
        w1: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: f32,
    ) -> Result<Self, String> {
        let hidden = b1.len();
        if hidden == 0 {
            return Err("policy hidden must be >= 1".to_string());
        }
        if w1.len() != hidden * POLICY_FEATURES {
            return Err(format!(
                "w1 has {} entries, expected {}",
                w1.len(),
                hidden * POLICY_FEATURES
            ));
        }
        if w2.len() != hidden {
            return Err(format!("w2 has {} entries, expected {hidden}", w2.len()));
        }
        Ok(Self { hidden, mean, scale, w1, b1, w2, b2 })
    }

    /// Serialize to the RFPO v1 byte layout (see module docs). Round-trips with
    /// [`from_bytes`](Self::from_bytes).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&(POLICY_FEATURES as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.hidden as u32).to_le_bytes());
        for value in self.mean.iter().chain(self.scale.iter()) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in self.w1.iter().chain(self.b1.iter()).chain(self.w2.iter()) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&self.b2.to_le_bytes());
        bytes
    }

    /// Additive move-ordering correction, in classical order-score units, from the
    /// cutoff probability and a tunable magnitude `bound`. `(2p - 1) * bound` maps
    /// `p in [0, 1]` onto `[-bound, +bound]`: a move the policy is confident cuts is
    /// ordered earlier, one it doubts later, and `p = 0.5` is exactly neutral (zero
    /// correction). Integer output so the policy magnitude can live in `SearchParams`
    /// and be gated by A/B while the model itself stays a pure predictor — the same
    /// split learned LMR uses for its thresholds.
    pub fn order_correction(&self, feats: &[f32; POLICY_FEATURES], bound: i32) -> i32 {
        let raw = (2.0 * self.cutoff_prob(feats) - 1.0) * bound as f32;
        (raw.round() as i32).clamp(-bound, bound)
    }

    /// P(move causes a cutoff) for the raw (un-normalized) feature vector. Clamps the
    /// unbounded columns exactly as the trainer does, standardizes with the stored
    /// mean/scale, then runs the forward pass (matching the trainer).
    pub fn cutoff_prob(&self, raw: &[f32; POLICY_FEATURES]) -> f32 {
        let mut feats = *raw;
        for (index, bound) in CLAMPS {
            feats[index] = feats[index].clamp(-bound, bound);
        }
        // Standardize once per feature into a stack buffer, then the hidden loop is a
        // plain dot product. This computes the exact same `(feat - mean) * scale` value
        // as before — just hoisted out of the `hidden`-way inner loop instead of
        // recomputed for every hidden unit — so the result is bit-identical (the
        // zero-correction byte-identical guarantee still holds) while the standardize
        // arithmetic drops from `hidden * POLICY_FEATURES` to `POLICY_FEATURES`.
        let mut normalized = [0f32; POLICY_FEATURES];
        for i in 0..POLICY_FEATURES {
            normalized[i] = (feats[i] - self.mean[i]) * self.scale[i];
        }
        let mut out = self.b2;
        for j in 0..self.hidden {
            let mut h = self.b1[j];
            let row = j * POLICY_FEATURES;
            for i in 0..POLICY_FEATURES {
                h += self.w1[row + i] * normalized[i];
            }
            if h > 0.0 {
                out += self.w2[j] * h; // ReLU: non-positive hidden units contribute 0
            }
        }
        1.0 / (1.0 + (-out).exp())
    }
}

/// The feature indices clamped before standardization, exposed so the trainer applies
/// the identical clamps when it builds its dataset (single-sourced with `CLAMPS`).
pub const POLICY_CLAMP_INDICES: [(usize, f32); 4] = CLAMPS;

/// Default magnitude of the move-ordering correction, in classical order-score units.
/// Chosen to reorder meaningfully *within* the quiet-move history band (history scores
/// run to ~20000) and nudge near bucket edges, while staying far below the priority
/// gaps (TT/capture/killer buckets are hundreds of thousands apart) so the policy
/// refines ordering rather than overriding the proven structure. Tunable like the LMR
/// thresholds; the real value is whatever a gate prefers.
pub const DEFAULT_POLICY_ORDER_BOUND: i32 = 4000;

/// Default number of top-of-order moves the policy re-ranks. `0` means "all moves":
/// the reference behavior, so a run that does not opt into a cap reproduces the
/// full-policy ordering. The real value is whatever a gate prefers once the inference
/// tax is traded off against ordering quality — a small `K` is the lever that makes
/// per-node inference cheap.
pub const DEFAULT_POLICY_ORDER_TOP_K: usize = 0;

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an RFPO byte buffer through the public serializer so tests exercise the
    /// real round-trip. Hidden size is inferred from `b1`. Identity standardization
    /// (mean 0, scale 1).
    fn build(w1: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: f32) -> Vec<u8> {
        PolicyModel::from_parameters([0.0; POLICY_FEATURES], [1.0; POLICY_FEATURES], w1, b1, w2, b2)
            .expect("valid parameters")
            .to_bytes()
    }

    #[test]
    fn rfpo_round_trips_and_forward_matches_hand_computation() {
        // One hidden unit; w1 = all-ones, b1 = 0, w2 = [1], b2 = 0.
        let w1 = vec![1.0f32; POLICY_FEATURES];
        let bytes = build(w1, vec![0.0], vec![1.0], 0.0);
        let model = PolicyModel::from_bytes(&bytes).expect("parse");
        // A single unit feature => hidden h = 1.0 (relu) => out = 1.0 => sigmoid(1).
        let mut feats = [0.0f32; POLICY_FEATURES];
        feats[0] = 1.0;
        let p = model.cutoff_prob(&feats);
        let expected = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((p - expected).abs() < 1e-6, "p={p} expected={expected}");
    }

    #[test]
    fn probability_stays_in_unit_interval() {
        let w1 = vec![0.0f32; POLICY_FEATURES];
        for b2 in [-5.0f32, -2.0, 0.0, 2.0, 5.0] {
            let model =
                PolicyModel::from_bytes(&build(w1.clone(), vec![0.0], vec![0.0], b2)).unwrap();
            let p = model.cutoff_prob(&[0.0; POLICY_FEATURES]);
            assert!((0.0..=1.0).contains(&p), "probability {p} out of [0,1] for b2={b2}");
        }
    }

    #[test]
    fn rejects_bad_magic_and_dim() {
        assert!(PolicyModel::from_bytes(b"XXXX....").is_err());
        let mut bytes = build(vec![0.0; POLICY_FEATURES], vec![0.0], vec![0.0], 0.0);
        bytes[8] = 9; // corrupt input_dim (byte after magic+version)
        assert!(PolicyModel::from_bytes(&bytes).is_err());
    }

    #[test]
    fn order_correction_is_signed_neutral_at_half_and_bounded() {
        let bound = 1000;
        // One hidden unit h = relu(x0), out = h - 2. A larger feature keeps the unit
        // active and lifts the logit above 0 (p > 0.5 => earlier); a small feature
        // leaves out = -2 (p < 0.5 => later). Both within ±bound. (A single ReLU unit
        // with a *negative* pre-activation would clamp to 0 and can't go negative, so
        // the negative case comes from the bias, not from driving the unit negative.)
        let mut w1 = vec![0.0f32; POLICY_FEATURES];
        w1[0] = 1.0;
        let model = PolicyModel::from_bytes(&build(w1, vec![0.0], vec![1.0], -2.0)).unwrap();
        let mut hi = [0.0f32; POLICY_FEATURES];
        hi[0] = 5.0;
        let lo = [0.0f32; POLICY_FEATURES];
        let c_hi = model.order_correction(&hi, bound);
        let c_lo = model.order_correction(&lo, bound);
        assert!(c_hi > 0 && c_lo < 0, "expected signed correction, got hi={c_hi} lo={c_lo}");
        assert!((-bound..=bound).contains(&c_hi) && (-bound..=bound).contains(&c_lo));

        // All-zero net with b2 = 0 => p = 0.5 for every input => exactly neutral. This
        // is the property the search's byte-identical test relies on.
        let neutral =
            PolicyModel::from_bytes(&build(vec![0.0; POLICY_FEATURES], vec![0.0], vec![0.0], 0.0))
                .unwrap();
        assert_eq!(neutral.order_correction(&[7.0; POLICY_FEATURES], bound), 0);
    }

    #[test]
    fn unbounded_columns_are_clamped() {
        // A unit weight on each clamped column: a value far past the bound must score
        // exactly as the bound does (the largest value the trainer ever standardized on).
        for (index, bound) in POLICY_CLAMP_INDICES {
            let mut w1 = vec![0.0f32; POLICY_FEATURES];
            w1[index] = 1.0;
            let model =
                PolicyModel::from_bytes(&build(w1, vec![0.0], vec![1.0], 0.0)).unwrap();
            let mut at_bound = [0.0f32; POLICY_FEATURES];
            at_bound[index] = bound;
            let mut way_past = [0.0f32; POLICY_FEATURES];
            way_past[index] = bound * 1_000.0;
            assert_eq!(
                model.cutoff_prob(&at_bound),
                model.cutoff_prob(&way_past),
                "column {index} not clamped at {bound}"
            );
        }
    }
}
