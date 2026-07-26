//! In-process Rust trainer for the Phase 4 move-ordering policy — a port of what
//! `modal/train_lmr.py` does for the LMR net, but with no Python and no Modal in the
//! loop. For a ~300-parameter MLP the gradient work is milliseconds; the only reason
//! the LMR trainer lived on Modal was TSV parsing, which is already Rust (`.npy`
//! export). Keeping training in-process too means the whole train -> gate loop runs in
//! one binary, inside the sandbox, with no GPU and no external service.
//!
//! Model: [`engine_search::POLICY_FEATURES`] standardized features -> `hidden` ReLU ->
//! 1 logit. Trained with class-weighted BCE-with-logits and Adam, exactly like the LMR
//! net, and serialized to `RFPO` via [`PolicyModel`] so the trainer and the engine's
//! inference agree by construction (a round-trip test pins it).

use std::io::{BufRead, BufReader};

use engine_search::{
    PolicyModel, LMR_FILTER_COLUMN, POLICY_FEATURES, POLICY_FEATURE_CLAMPS, POLICY_FEATURE_COLUMNS,
    POLICY_TARGET_COLUMN,
};

/// Deterministic SplitMix64 -> `f32` in `[0, 1)`. A fixed seed makes a training run
/// reproducible without pulling in an RNG dependency; the exact stream doesn't matter,
/// only that it's stable and decently mixed for weight init and shuffling.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f32` in `[0, 1)` (24-bit mantissa resolution).
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    /// Uniform `f32` in `[-half, half)`.
    fn centered(&mut self, half: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * half
    }

    /// Fisher-Yates shuffle.
    fn shuffle(&mut self, slice: &mut [u32]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            slice.swap(i, j);
        }
    }
}

/// Knobs for a training run. Defaults mirror `train_lmr.py`.
#[derive(Clone, Copy, Debug)]
pub struct PolicyTrainConfig {
    pub hidden: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    /// Keep 1 row in `stride` (decorrelates rows drawn from the same search).
    pub stride: u64,
    /// Stop collecting after this many kept rows (bounds memory on huge datasets).
    pub max_rows: usize,
    pub seed: u64,
}

impl Default for PolicyTrainConfig {
    fn default() -> Self {
        Self {
            hidden: 16,
            epochs: 12,
            batch_size: 8192,
            learning_rate: 1e-3,
            stride: 1,
            max_rows: 10_000_000,
            seed: 0,
        }
    }
}

/// What a training run learned, printed by the CLI so a campaign can read it back. An
/// `AUC` well above 0.5 is the signal that "search this move first" is learnable from
/// ordering-time features — the precondition for the policy to be worth wiring in.
#[derive(Clone, Copy, Debug)]
pub struct PolicyTrainSummary {
    pub rows: usize,
    pub scanned: u64,
    pub base_rate: f32,
    pub val_accuracy: f32,
    pub val_auc: f32,
}

fn header_index(header: &[&str], name: &str) -> Result<usize, String> {
    header
        .iter()
        .position(|column| *column == name)
        .ok_or_else(|| format!("telemetry header is missing the `{name}` column"))
}

/// Stride-sample a telemetry TSV into an in-memory `(X, y)` dataset: `X` is a flat
/// `rows * POLICY_FEATURES` row-major buffer, `y` is the per-row target. Columns are
/// resolved by *name* against the file's own header (so a schema change can't silently
/// mis-map), `lmp_pruned` rows are dropped (never searched — their outcome is not an
/// observation), and the unbounded columns are clamped exactly as inference clamps them.
fn load_dataset<R: std::io::Read>(
    reader: R,
    stride: u64,
    max_rows: usize,
) -> Result<(Vec<f32>, Vec<f32>, u64), String> {
    if stride == 0 {
        return Err("invalid stride 0: need stride >= 1".to_string());
    }
    let mut lines = BufReader::with_capacity(1 << 20, reader).lines();
    let header_line = lines
        .next()
        .transpose()
        .map_err(|error| format!("failed to read telemetry header: {error}"))?
        .ok_or_else(|| "telemetry input is empty".to_string())?;
    let header: Vec<&str> = header_line.trim_end().split('\t').collect();

    let target_index = header_index(&header, POLICY_TARGET_COLUMN)?;
    let filter_index = header_index(&header, LMR_FILTER_COLUMN)?;
    // TSV column -> feature slot, so each row is one forward scan with no random access.
    let mut slot_for: Vec<Option<usize>> = vec![None; header.len()];
    for (slot, name) in POLICY_FEATURE_COLUMNS.iter().enumerate() {
        slot_for[header_index(&header, name)?] = Some(slot);
    }
    // Clamp bounds aligned to feature position, indexed directly during the scan.
    let mut clamps = [f32::INFINITY; POLICY_FEATURES];
    for (name, bound) in POLICY_FEATURE_CLAMPS {
        let at = POLICY_FEATURE_COLUMNS
            .iter()
            .position(|column| *column == name)
            .ok_or_else(|| format!("clamped column `{name}` is not a policy feature"))?;
        clamps[at] = bound;
    }

    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut scanned = 0u64;
    for line in lines {
        let line = line.map_err(|error| format!("failed to read telemetry: {error}"))?;
        let index = scanned;
        scanned += 1;
        if index % stride != 0 {
            continue;
        }
        let mut row = [0f32; POLICY_FEATURES];
        let mut target = 0f32;
        let mut columns = 0usize;
        let mut keep = true;
        for (column, field) in line.trim_end().split('\t').enumerate() {
            columns += 1;
            if column == filter_index && field == "1" {
                keep = false; // late-move-pruned: never searched, so not an observation
                break;
            }
            if column == target_index {
                match field.parse::<f32>() {
                    Ok(value) => target = value,
                    Err(_) => {
                        keep = false;
                        break;
                    }
                }
            }
            if let Some(slot) = slot_for.get(column).copied().flatten() {
                match field.parse::<f32>() {
                    Ok(value) => row[slot] = value.clamp(-clamps[slot], clamps[slot]),
                    Err(_) => {
                        keep = false;
                        break;
                    }
                }
            }
        }
        // A short row is an older schema rather than a parse error; drop it either way.
        if !keep || columns < header.len() {
            continue;
        }
        x.extend_from_slice(&row);
        y.push(target);
        if y.len() >= max_rows {
            break;
        }
    }
    Ok((x, y, scanned))
}

/// Column means and inverse-std scales, standardizing `x` in place. Constant columns
/// (a bool that never varied in the sample) get scale 1 so they don't divide by ~0.
fn standardize(x: &mut [f32], rows: usize) -> ([f32; POLICY_FEATURES], [f32; POLICY_FEATURES]) {
    let mut mean = [0f32; POLICY_FEATURES];
    let mut scale = [1f32; POLICY_FEATURES];
    if rows == 0 {
        return (mean, scale);
    }
    for row in x.chunks_exact(POLICY_FEATURES) {
        for i in 0..POLICY_FEATURES {
            mean[i] += row[i];
        }
    }
    for m in &mut mean {
        *m /= rows as f32;
    }
    let mut var = [0f32; POLICY_FEATURES];
    for row in x.chunks_exact(POLICY_FEATURES) {
        for i in 0..POLICY_FEATURES {
            let d = row[i] - mean[i];
            var[i] += d * d;
        }
    }
    for i in 0..POLICY_FEATURES {
        let std = (var[i] / rows as f32).sqrt();
        scale[i] = if std < 1e-6 { 1.0 } else { 1.0 / std };
    }
    for row in x.chunks_exact_mut(POLICY_FEATURES) {
        for i in 0..POLICY_FEATURES {
            row[i] = (row[i] - mean[i]) * scale[i];
        }
    }
    (mean, scale)
}

/// One hidden layer, trained by Adam. Weights held flat, row-major `[hidden, F]`.
struct Mlp {
    hidden: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: f32,
}

impl Mlp {
    /// PyTorch-style init: `U(-1/sqrt(fan_in), 1/sqrt(fan_in))` for each layer.
    fn new(hidden: usize, rng: &mut Rng) -> Self {
        let k1 = 1.0 / (POLICY_FEATURES as f32).sqrt();
        let k2 = 1.0 / (hidden as f32).sqrt();
        let w1 = (0..hidden * POLICY_FEATURES).map(|_| rng.centered(k1)).collect();
        let b1 = (0..hidden).map(|_| rng.centered(k1)).collect();
        let w2 = (0..hidden).map(|_| rng.centered(k2)).collect();
        let b2 = rng.centered(k2);
        Self { hidden, w1, b1, w2, b2 }
    }

    /// Logit for one standardized sample, returning the pre-activations too (needed for
    /// the ReLU gradient in backprop).
    fn forward(&self, x: &[f32], hpre: &mut [f32], h: &mut [f32]) -> f32 {
        let mut z = self.b2;
        for j in 0..self.hidden {
            let row = j * POLICY_FEATURES;
            let mut pre = self.b1[j];
            for i in 0..POLICY_FEATURES {
                pre += self.w1[row + i] * x[i];
            }
            hpre[j] = pre;
            let act = if pre > 0.0 { pre } else { 0.0 };
            h[j] = act;
            z += self.w2[j] * act;
        }
        z
    }

    fn to_policy_model(
        &self,
        mean: [f32; POLICY_FEATURES],
        scale: [f32; POLICY_FEATURES],
    ) -> Result<PolicyModel, String> {
        PolicyModel::from_parameters(
            mean,
            scale,
            self.w1.clone(),
            self.b1.clone(),
            self.w2.clone(),
            self.b2,
        )
    }
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// Adam moment buffers, one scalar per parameter, laid out to match the `Mlp` fields.
struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: i32,
    m_w1: Vec<f32>,
    v_w1: Vec<f32>,
    m_b1: Vec<f32>,
    v_b1: Vec<f32>,
    m_w2: Vec<f32>,
    v_w2: Vec<f32>,
    m_b2: f32,
    v_b2: f32,
}

impl Adam {
    fn new(model: &Mlp, lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m_w1: vec![0.0; model.w1.len()],
            v_w1: vec![0.0; model.w1.len()],
            m_b1: vec![0.0; model.b1.len()],
            v_b1: vec![0.0; model.b1.len()],
            m_w2: vec![0.0; model.w2.len()],
            v_w2: vec![0.0; model.w2.len()],
            m_b2: 0.0,
            v_b2: 0.0,
        }
    }

    fn step_scalar(&self, m: &mut f32, v: &mut f32, grad: f32, bias1: f32, bias2: f32) -> f32 {
        *m = self.beta1 * *m + (1.0 - self.beta1) * grad;
        *v = self.beta2 * *v + (1.0 - self.beta2) * grad * grad;
        let mhat = *m / bias1;
        let vhat = *v / bias2;
        self.lr * mhat / (vhat.sqrt() + self.eps)
    }

    /// Apply one Adam update from averaged gradients.
    fn apply(&mut self, model: &mut Mlp, g_w1: &[f32], g_b1: &[f32], g_w2: &[f32], g_b2: f32) {
        self.t += 1;
        let bias1 = 1.0 - self.beta1.powi(self.t);
        let bias2 = 1.0 - self.beta2.powi(self.t);
        for k in 0..model.w1.len() {
            let (mut m, mut v) = (self.m_w1[k], self.v_w1[k]);
            model.w1[k] -= self.step_scalar(&mut m, &mut v, g_w1[k], bias1, bias2);
            self.m_w1[k] = m;
            self.v_w1[k] = v;
        }
        for k in 0..model.b1.len() {
            let (mut m, mut v) = (self.m_b1[k], self.v_b1[k]);
            model.b1[k] -= self.step_scalar(&mut m, &mut v, g_b1[k], bias1, bias2);
            self.m_b1[k] = m;
            self.v_b1[k] = v;
        }
        for k in 0..model.w2.len() {
            let (mut m, mut v) = (self.m_w2[k], self.v_w2[k]);
            model.w2[k] -= self.step_scalar(&mut m, &mut v, g_w2[k], bias1, bias2);
            self.m_w2[k] = m;
            self.v_w2[k] = v;
        }
        let (mut m, mut v) = (self.m_b2, self.v_b2);
        model.b2 -= self.step_scalar(&mut m, &mut v, g_b2, bias1, bias2);
        self.m_b2 = m;
        self.v_b2 = v;
    }
}

/// ROC AUC via the rank formula (Mann-Whitney), no dependency. `NaN` if a class is
/// absent in the validation split.
fn roc_auc(y: &[f32], scores: &[f32]) -> f32 {
    let n = y.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut rank = vec![0f64; n];
    // Average ranks over ties so equal scores don't bias the statistic.
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && scores[order[j]] == scores[order[i]] {
            j += 1;
        }
        let avg = ((i + 1 + j) as f64) / 2.0; // mean of ranks (i+1..=j), 1-based
        for &idx in &order[i..j] {
            rank[idx] = avg;
        }
        i = j;
    }
    let n_pos = y.iter().filter(|&&v| v == 1.0).count();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return f32::NAN;
    }
    let sum_pos: f64 = (0..n).filter(|&k| y[k] == 1.0).map(|k| rank[k]).sum();
    ((sum_pos - (n_pos * (n_pos + 1)) as f64 / 2.0) / (n_pos as f64 * n_neg as f64)) as f32
}

/// Fit the policy from a telemetry TSV and return the serialized `RFPO` bytes plus
/// metrics. Standardization, class-weighted BCE, Adam, and the 90/10 val split mirror
/// `train_lmr.py`; the forward pass is [`PolicyModel`]'s, so training and inference match.
pub fn train_policy<R: std::io::Read>(
    reader: R,
    config: PolicyTrainConfig,
) -> Result<(Vec<u8>, PolicyTrainSummary), String> {
    let (mut x, y, scanned) = load_dataset(reader, config.stride, config.max_rows)?;
    let rows = y.len();
    if rows == 0 {
        return Err("no usable telemetry rows after filtering".to_string());
    }
    let (mean, scale) = standardize(&mut x, rows);

    // 90/10 split on a seeded permutation.
    let mut rng = Rng::new(config.seed);
    let mut perm: Vec<u32> = (0..rows as u32).collect();
    rng.shuffle(&mut perm);
    let n_val = (rows / 10).max(1);
    let (val_idx, train_idx) = perm.split_at(n_val);
    let train_idx = train_idx.to_vec();

    let base_rate = y.iter().sum::<f32>() / rows as f32;
    // Class weight on the positive term so the ~few-percent cutoff rate isn't ignored.
    let pos_weight = (1.0 - base_rate) / base_rate.max(1e-6);

    let mut model = Mlp::new(config.hidden, &mut rng);
    let mut opt = Adam::new(&model, config.learning_rate);
    let hidden = config.hidden;
    let mut hpre = vec![0f32; hidden];
    let mut h = vec![0f32; hidden];
    let mut g_w1 = vec![0f32; model.w1.len()];
    let mut g_b1 = vec![0f32; hidden];
    let mut g_w2 = vec![0f32; hidden];

    let mut epoch_order = train_idx.clone();
    for _ in 0..config.epochs {
        rng.shuffle(&mut epoch_order);
        for batch in epoch_order.chunks(config.batch_size) {
            g_w1.iter_mut().for_each(|v| *v = 0.0);
            g_b1.iter_mut().for_each(|v| *v = 0.0);
            g_w2.iter_mut().for_each(|v| *v = 0.0);
            let mut g_b2 = 0f32;
            let inv = 1.0 / batch.len() as f32;
            for &sample in batch {
                let s = sample as usize;
                let xi = &x[s * POLICY_FEATURES..(s + 1) * POLICY_FEATURES];
                let z = model.forward(xi, &mut hpre, &mut h);
                let p = sigmoid(z);
                let yi = y[s];
                // d/dz of class-weighted BCE-with-logits: (1-y)*p - pos_weight*y*(1-p).
                let g_z = ((1.0 - yi) * p - pos_weight * yi * (1.0 - p)) * inv;
                g_b2 += g_z;
                for j in 0..hidden {
                    g_w2[j] += g_z * h[j];
                    let dh = g_z * model.w2[j];
                    let dpre = if hpre[j] > 0.0 { dh } else { 0.0 };
                    g_b1[j] += dpre;
                    let row = j * POLICY_FEATURES;
                    for i in 0..POLICY_FEATURES {
                        g_w1[row + i] += dpre * xi[i];
                    }
                }
            }
            opt.apply(&mut model, &g_w1, &g_b1, &g_w2, g_b2);
        }
    }

    // Validation metrics.
    let mut val_scores = Vec::with_capacity(n_val);
    let mut val_targets = Vec::with_capacity(n_val);
    let mut correct = 0usize;
    for &sample in val_idx {
        let s = sample as usize;
        let xi = &x[s * POLICY_FEATURES..(s + 1) * POLICY_FEATURES];
        let z = model.forward(xi, &mut hpre, &mut h);
        let p = sigmoid(z);
        let yi = y[s];
        if (p > 0.5) == (yi == 1.0) {
            correct += 1;
        }
        val_scores.push(p);
        val_targets.push(yi);
    }
    let val_accuracy = correct as f32 / n_val as f32;
    let val_auc = roc_auc(&val_targets, &val_scores);

    let bytes = model.to_policy_model(mean, scale)?.to_bytes();
    let summary = PolicyTrainSummary {
        rows,
        scanned,
        base_rate,
        val_accuracy,
        val_auc,
    };
    Ok((bytes, summary))
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_search::TELEMETRY_TSV_HEADER;

    /// Build a one-row TSV line with a given `caused_cutoff` label and `see`, all other
    /// columns zero. Only the columns the loader reads need to be meaningful.
    fn tsv_row(header: &[&str], see: i32, caused_cutoff: u8) -> String {
        header
            .iter()
            .map(|&col| match col {
                "pos_id" => "0".to_string(),
                "see" => see.to_string(),
                "caused_cutoff" => caused_cutoff.to_string(),
                _ => "0".to_string(),
            })
            .collect::<Vec<_>>()
            .join("\t")
    }

    /// A dataset where `see > 0` perfectly predicts a cutoff: the trainer must drive
    /// validation AUC to (near) 1, proving the full pipeline — load, standardize,
    /// forward, backward, Adam — learns a real signal, and that the exported bytes load
    /// back through `PolicyModel` and rank a positive above a negative.
    #[test]
    fn trainer_learns_a_separable_signal_and_round_trips() {
        let header: Vec<&str> = TELEMETRY_TSV_HEADER.split('\t').collect();
        let mut tsv = String::from(TELEMETRY_TSV_HEADER);
        tsv.push('\n');
        // 600 rows, alternating classes: positive iff see > 0.
        for i in 0..600 {
            let positive = i % 2 == 0;
            let see = if positive { 500 } else { -500 };
            tsv.push_str(&tsv_row(&header, see, u8::from(positive)));
            tsv.push('\n');
        }
        let config = PolicyTrainConfig {
            hidden: 8,
            epochs: 40,
            batch_size: 128,
            learning_rate: 5e-3,
            stride: 1,
            max_rows: 10_000,
            seed: 1,
        };
        let (bytes, summary) = train_policy(tsv.as_bytes(), config).expect("training");
        assert_eq!(summary.rows, 600);
        assert!(
            summary.val_auc > 0.95,
            "AUC {} — trainer failed to learn the separable signal",
            summary.val_auc
        );

        let model = PolicyModel::from_bytes(&bytes).expect("exported RFPO parses");
        let see_slot = POLICY_FEATURE_COLUMNS.iter().position(|c| *c == "see").unwrap();
        let mut pos = [0f32; POLICY_FEATURES];
        pos[see_slot] = 500.0;
        let mut neg = [0f32; POLICY_FEATURES];
        neg[see_slot] = -500.0;
        assert!(
            model.cutoff_prob(&pos) > model.cutoff_prob(&neg),
            "positive SEE should rank above negative after training"
        );
    }

    #[test]
    fn empty_after_filtering_is_an_error() {
        let mut tsv = String::from(TELEMETRY_TSV_HEADER);
        tsv.push('\n');
        assert!(train_policy(tsv.as_bytes(), PolicyTrainConfig::default()).is_err());
    }

    #[test]
    fn auc_is_half_for_random_scores() {
        // Interleaved identical structure, label independent of the only varying feature.
        let y = [1.0, 0.0, 1.0, 0.0];
        let s = [0.3f32, 0.3, 0.7, 0.7]; // ties across classes => 0.5
        let auc = roc_auc(&y, &s);
        assert!((auc - 0.5).abs() < 1e-6, "auc={auc}");
    }
}
