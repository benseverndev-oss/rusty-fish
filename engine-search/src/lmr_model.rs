//! Learned-LMR model (Phase 2): a tiny MLP that predicts P(a move raises alpha)
//! from the search context, loaded from the `RFLM` format `modal/train_lmr.py`
//! exports. The search turns the probability into a clamped reduction correction
//! (`reduction = classical + clamp(correction, -1, +2)`). Inference is a handful of
//! FLOPs (18 -> hidden -> 1), cheap enough for the hot loop.
//!
//! RFLM v1 layout (little-endian): magic `b"RFLM"` | u32 version=1 | u32 input_dim
//! | u32 hidden | mean[input_dim] f32 | scale[input_dim] f32 | w1[hidden*input_dim]
//! f32 (row-major [hidden, input]) | b1[hidden] f32 | w2[hidden] f32 | b2 f32.
//!
//! `input_dim` is checked against [`LMR_FEATURES`] on load, so widening the feature
//! set and swapping the bundled asset have to happen in the same commit.

use std::sync::LazyLock;

const MAGIC: &[u8; 4] = b"RFLM";
const VERSION: u32 = 1;

/// The context features, in the exact order the trainer used (train_lmr.py
/// `FEATURE_COLS`): depth, ply, move_index, is_quiet, is_priority, pv_node,
/// gives_check, static_eval, extension, reduction, history_score, is_tt_move,
/// is_killer, is_counter, is_capture, is_promotion, node_in_check, tt_depth.
///
/// The first ten are the v1 set. The last eight were added after the v1 model
/// saturated at val AUC ~0.94 against both more data (depth-9/192M -> 0.9378) and
/// more capacity (hidden 64 -> 0.9396): it was feature-limited, not data- or
/// capacity-limited. Widening to 18 moved val AUC to 0.9463 at the same hidden 16.
pub const LMR_FEATURES: usize = 18;

/// Index of `static_eval` in the feature vector.
const FEAT_STATIC_EVAL: usize = 7;
/// Index of `history_score` in the feature vector.
const FEAT_HISTORY: usize = 10;
/// Clamp bounds mirroring `train_lmr.py`'s `load_telemetry_sample`. Both scalars are
/// unbounded in the search (mate scores; uncapped history), and the trainer clamps
/// them before fitting the standardization, so inference has to clamp identically or
/// the extremes land far outside the distribution the model was standardized on.
const STATIC_EVAL_CLAMP: f32 = 2000.0;
const HISTORY_CLAMP: f32 = 20000.0;

/// A loaded learned-LMR model. Immutable after load, cheap to share behind `Arc`.
#[derive(Clone, Debug, PartialEq)]
pub struct LmrModel {
    hidden: usize,
    mean: [f32; LMR_FEATURES],
    scale: [f32; LMR_FEATURES],
    w1: Vec<f32>, // [hidden * LMR_FEATURES], row-major (hidden rows of LMR_FEATURES)
    b1: Vec<f32>, // [hidden]
    w2: Vec<f32>, // [hidden]
    b2: f32,
}

fn take<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8], String> {
    let end = *cursor + len;
    let slice = bytes.get(*cursor..end).ok_or_else(|| "RFLM truncated".to_string())?;
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

impl LmrModel {
    /// Parse an RFLM v1 byte buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut cursor = 0usize;
        if take(bytes, &mut cursor, 4)? != MAGIC {
            return Err("not an RFLM file (bad magic)".to_string());
        }
        let version = read_u32(bytes, &mut cursor)?;
        if version != VERSION {
            return Err(format!("unsupported RFLM version {version}"));
        }
        let input_dim = read_u32(bytes, &mut cursor)? as usize;
        if input_dim != LMR_FEATURES {
            return Err(format!("RFLM input_dim {input_dim} != {LMR_FEATURES}"));
        }
        let hidden = read_u32(bytes, &mut cursor)? as usize;
        if hidden == 0 {
            return Err("RFLM hidden must be >= 1".to_string());
        }
        let mean_v = read_f32s(bytes, &mut cursor, LMR_FEATURES)?;
        let scale_v = read_f32s(bytes, &mut cursor, LMR_FEATURES)?;
        let w1 = read_f32s(bytes, &mut cursor, hidden * LMR_FEATURES)?;
        let b1 = read_f32s(bytes, &mut cursor, hidden)?;
        let w2 = read_f32s(bytes, &mut cursor, hidden)?;
        let b2 = read_f32(bytes, &mut cursor)?;
        let mut mean = [0f32; LMR_FEATURES];
        let mut scale = [0f32; LMR_FEATURES];
        mean.copy_from_slice(&mean_v);
        scale.copy_from_slice(&scale_v);
        Ok(Self { hidden, mean, scale, w1, b1, w2, b2 })
    }

    /// Load an RFLM model from a file.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|error| format!("failed to read {path}: {error}"))?;
        Self::from_bytes(&bytes)
    }

    /// P(move raises alpha) for the raw (un-normalized) feature vector. Clamps the two
    /// unbounded scalars exactly as the trainer does, standardizes with the stored
    /// mean/scale, then runs the forward pass (matching the trainer).
    pub fn raise_alpha_prob(&self, raw: &[f32; LMR_FEATURES]) -> f32 {
        let mut feats = *raw;
        feats[FEAT_STATIC_EVAL] =
            feats[FEAT_STATIC_EVAL].clamp(-STATIC_EVAL_CLAMP, STATIC_EVAL_CLAMP);
        feats[FEAT_HISTORY] = feats[FEAT_HISTORY].clamp(-HISTORY_CLAMP, HISTORY_CLAMP);
        let mut out = self.b2;
        for j in 0..self.hidden {
            let mut h = self.b1[j];
            let row = j * LMR_FEATURES;
            for i in 0..LMR_FEATURES {
                let normalized = (feats[i] - self.mean[i]) * self.scale[i];
                h += self.w1[row + i] * normalized;
            }
            if h > 0.0 {
                out += self.w2[j] * h; // ReLU: non-positive hidden units contribute 0
            }
        }
        1.0 / (1.0 + (-out).exp())
    }

    /// Reduction correction in plies, always in `[-1, +2]`, from per-mille P(raise
    /// alpha) thresholds. Integers so the policy can live in `SearchParams` and be
    /// tuned by gated A/B — the model stays a pure predictor, the policy is tunable.
    pub fn reduction_correction_with(
        &self,
        feats: &[f32; LMR_FEATURES],
        unreduce_permille: i32,
        reduce2_permille: i32,
        reduce1_permille: i32,
    ) -> i8 {
        let permille = (self.raise_alpha_prob(feats) * 1000.0) as i32;
        if permille >= unreduce_permille {
            -1
        } else if permille < reduce2_permille {
            2
        } else if permille < reduce1_permille {
            1
        } else {
            0
        }
    }

    /// [`reduction_correction_with`](Self::reduction_correction_with) at the default
    /// thresholds (the settings the bundled model was swept for).
    pub fn reduction_correction(&self, feats: &[f32; LMR_FEATURES]) -> i8 {
        self.reduction_correction_with(
            feats,
            DEFAULT_LMR_UNREDUCE_PERMILLE,
            DEFAULT_LMR_REDUCE2_PERMILLE,
            DEFAULT_LMR_REDUCE1_PERMILLE,
        )
    }
}

/// Default correction thresholds, per-mille P(raise alpha). `SearchParams` mirrors
/// these and is what the search actually reads, so they can be tuned per-run.
///
/// Tuned by gated A/B at equal movetime (50 ms/move, sharded SPRT). Reducing *harder*
/// than the original guess pays, matching what the telemetry said: classical LMR errs
/// on only 2.31% of the moves it reduces, so the predictably-safe majority can take
/// another ply.
///
/// **These are per-model.** The thresholds slice a probability distribution, and a
/// retrained model recalibrates — so a model swap without a re-sweep is a regression,
/// not a neutral change. Measured, at 16384 games (SE ~±2.3 Elo):
///
///   model              unreduce/reduce2/reduce1 -> Elo vs classical
///   v1 (10 features)   500 / 100 / 220          -> +57.7
///   v2 (18 features)   500 / 100 / 220          -> +50.6   (v1's thresholds: -7.1)
///   v2 (18 features)   500 / 250 / 450 (ADOPTED)-> +57.2   (re-swept: recovered)
///
/// The earlier 4096-game sweep (+38.3 -> +49.6 -> +56.5 -> +57.3) established the
/// direction, but its steps were 1-2 sigma at that sample size and its fine ordering
/// should not be read as resolved.
///
/// `reduce1` has little headroom left: at >= `unreduce` it is unreachable, because the
/// unreduce branch is tested first.
pub const DEFAULT_LMR_UNREDUCE_PERMILLE: i32 = 500;
pub const DEFAULT_LMR_REDUCE2_PERMILLE: i32 = 250;
pub const DEFAULT_LMR_REDUCE1_PERMILLE: i32 = 450;

/// The engine's default learned-LMR model, compiled into the binary and parsed once.
/// Adopted 2026-07-24 after gating +38.3 Elo (equal movetime, 4096 games, AcceptH1),
/// then re-trained on the 18-feature telemetry (`d8-v2feat`, val AUC 0.9463).
static BUNDLED_LMR_MODEL: LazyLock<LmrModel> = LazyLock::new(|| {
    LmrModel::from_bytes(include_bytes!("../../assets/lmr/rusty-fish-lmr.rflm"))
        .expect("bundled LMR asset is a valid RFLM model")
});

/// The bundled default learned-LMR model (a cheap clone of the parsed-once model).
pub fn bundled_lmr_model() -> LmrModel {
    BUNDLED_LMR_MODEL.clone()
}

/// Bias that puts an all-zero-weight model in the middle of the no-correction band
/// `[reduce1, unreduce)`, expressed as a logit so the forward pass reproduces exactly
/// that probability for every input.
///
/// Derived from the threshold constants rather than hardcoded: a fixture pinned to
/// one set of thresholds silently stops testing what it claims the moment they are
/// re-swept, which is exactly what happened when the v2 model moved the band.
#[cfg(test)]
pub(crate) fn neutral_correction_bias() -> f32 {
    let permille =
        (DEFAULT_LMR_REDUCE1_PERMILLE + DEFAULT_LMR_UNREDUCE_PERMILLE) as f32 / 2.0;
    let probability = permille / 1000.0;
    (probability / (1.0 - probability)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an RFLM byte buffer for a given (hidden) net so tests exercise the real
    /// loader. `w1`/`b1`/`w2` are flat; identity-ish standardization (mean 0, scale 1).
    fn build_rflm(hidden: usize, w1: &[f32], b1: &[f32], w2: &[f32], b2: f32) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&VERSION.to_le_bytes());
        bytes.extend_from_slice(&(LMR_FEATURES as u32).to_le_bytes());
        bytes.extend_from_slice(&(hidden as u32).to_le_bytes());
        for _ in 0..LMR_FEATURES {
            bytes.extend_from_slice(&0f32.to_le_bytes()); // mean
        }
        for _ in 0..LMR_FEATURES {
            bytes.extend_from_slice(&1f32.to_le_bytes()); // scale
        }
        for &v in w1 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for &v in b1 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for &v in w2 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes.extend_from_slice(&b2.to_le_bytes());
        bytes
    }

    #[test]
    fn rflm_round_trips_and_forward_matches_hand_computation() {
        // One hidden unit; w1 = all-ones, b1 = 0, w2 = [1], b2 = 0.
        let w1 = vec![1.0f32; LMR_FEATURES];
        let bytes = build_rflm(1, &w1, &[0.0], &[1.0], 0.0);
        let model = LmrModel::from_bytes(&bytes).expect("parse");
        // feats sum to 1.0 -> hidden h = 1.0 (relu) -> out = 1.0 -> sigmoid(1).
        let mut feats = [0.0f32; LMR_FEATURES];
        feats[0] = 1.0;
        let p = model.raise_alpha_prob(&feats);
        let expected = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((p - expected).abs() < 1e-6, "p={p} expected={expected}");
    }

    #[test]
    fn correction_stays_in_range() {
        let w1 = vec![0.0f32; LMR_FEATURES];
        for b2 in [-5.0f32, -2.0, 0.0, 2.0, 5.0] {
            let model = LmrModel::from_bytes(&build_rflm(1, &w1, &[0.0], &[0.0], b2)).unwrap();
            let c = model.reduction_correction(&[0.0; LMR_FEATURES]);
            assert!((-1..=2).contains(&c), "correction {c} out of range for b2={b2}");
        }
    }

    #[test]
    fn zero_weights_with_neutral_bias_gives_zero_correction() {
        // All-zero weights => out = b2 for every input, so the bias alone decides the
        // probability. `neutral_correction_bias` places it mid-band, which is the
        // model the search's byte-identical test installs (its only possible effect on
        // the search is via the correction, so a zero correction must change nothing).
        let model = LmrModel::from_bytes(&build_rflm(
            1,
            &vec![0.0; LMR_FEATURES],
            &[0.0],
            &[0.0],
            neutral_correction_bias(),
        ))
        .unwrap();
        assert_eq!(model.reduction_correction(&[42.0; LMR_FEATURES]), 0);
    }

    #[test]
    fn rejects_bad_magic_and_dim() {
        assert!(LmrModel::from_bytes(b"XXXX....").is_err());
        let mut bytes = build_rflm(1, &vec![0.0; LMR_FEATURES], &[0.0], &[0.0], 0.0);
        bytes[8] = 9; // corrupt input_dim (byte after magic+version)
        assert!(LmrModel::from_bytes(&bytes).is_err());
    }

    #[test]
    fn bundled_lmr_model_parses_and_predicts_sanely() {
        let model = bundled_lmr_model();
        // depth, ply, move_index, is_quiet, is_priority, pv_node, gives_check,
        // static_eval, extension, reduction, history, tt/killer/counter/capture/promo,
        // node_in_check, tt_depth.
        let feats = [
            6.0, 4.0, 5.0, 1.0, 0.0, 0.0, 0.0, 20.0, 0.0, 1.0, 350.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 5.0,
        ];
        let p = model.raise_alpha_prob(&feats);
        assert!((0.0..=1.0).contains(&p), "probability {p} out of [0,1]");
        assert!((-1..=2).contains(&model.reduction_correction(&feats)));
    }

    #[test]
    fn unbounded_scalars_are_clamped_like_the_trainer() {
        // A single unit weight on static_eval: a mate-sized eval must score exactly as
        // the clamp bound does, which is the largest value the trainer ever saw.
        let mut w1 = vec![0.0f32; LMR_FEATURES];
        w1[FEAT_STATIC_EVAL] = 1.0;
        let model = LmrModel::from_bytes(&build_rflm(1, &w1, &[0.0], &[1.0], 0.0)).unwrap();
        let mut at_bound = [0.0f32; LMR_FEATURES];
        at_bound[FEAT_STATIC_EVAL] = STATIC_EVAL_CLAMP;
        let mut way_past = [0.0f32; LMR_FEATURES];
        way_past[FEAT_STATIC_EVAL] = 30_000.0; // a mate score
        assert_eq!(model.raise_alpha_prob(&at_bound), model.raise_alpha_prob(&way_past));

        let mut w1 = vec![0.0f32; LMR_FEATURES];
        w1[FEAT_HISTORY] = 1.0;
        let model = LmrModel::from_bytes(&build_rflm(1, &w1, &[0.0], &[1.0], 0.0)).unwrap();
        let mut at_bound = [0.0f32; LMR_FEATURES];
        at_bound[FEAT_HISTORY] = -HISTORY_CLAMP;
        let mut way_past = [0.0f32; LMR_FEATURES];
        way_past[FEAT_HISTORY] = -1_000_000.0;
        assert_eq!(model.raise_alpha_prob(&at_bound), model.raise_alpha_prob(&way_past));
    }
}
