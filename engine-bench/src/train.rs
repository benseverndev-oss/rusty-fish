//! A self-contained bootstrap trainer for the NNUE network.
//!
//! It generates training positions by random self-play, labels each with the
//! hand-crafted evaluation (the "teacher"), fits a float network by stochastic
//! gradient descent using the same arithmetic structure as the quantised
//! inference, and exports a quantised `RFNN` network.
//!
//! This validates the whole training -> export -> load -> play pipeline without
//! any external dependency. Because it distils the hand-crafted evaluation it
//! only approximates that evaluation's strength; swapping the teacher for a
//! deeper search, game outcomes, or an external engine is the path to a network
//! that exceeds it.

use engine_core::{Board, Color};
use engine_search::{
    active_features, hand_crafted_evaluation, Nnue, SearchLimits, Searcher, INPUT_DIMENSION,
};

/// Search scores can reach mate values; training targets are clamped to this
/// magnitude so the regression stays well-conditioned.
const TARGET_CLAMP: i32 = 10_000;

/// Centipawns-to-win-probability scale for the training loss. Targets and
/// predictions are squashed through `sigmoid(cp / WDL_SCALE)` before the loss is
/// taken, which bounds every gradient regardless of target magnitude — so
/// extreme tactical labels no longer dominate the fit (nor make it diverge).
const WDL_SCALE: f32 = 400.0;

/// Divisor mirroring the quantised inference's `OUTPUT_SCALE`, so the float
/// model and the exported integer model share the same output scaling.
const OUTPUT_SCALE: f32 = 64.0;
/// Clipped-ReLU upper bound, matching the inference activation clamp.
const ACTIVATION_CLIP: f32 = 127.0;

fn opposite(color: Color) -> Color {
    match color {
        Color::White => Color::Black,
        Color::Black => Color::White,
    }
}

/// One training position: the active features from each perspective and the
/// side-to-move-relative teacher score in centipawns.
#[derive(Clone, Debug)]
pub struct TrainingSample {
    pub own: Vec<usize>,
    pub opp: Vec<usize>,
    pub target: f32,
}

/// Generates training samples by walking a random (seeded) legal game from each
/// seed position, recording every position with a teacher label.
///
/// When `label_depth` is `None` the label is the static hand-crafted
/// evaluation. When it is `Some(depth)` the label is a depth-`depth` search
/// score (with the hand-crafted evaluation at the leaves), which distils search
/// knowledge into the static network and is a stronger teacher than the static
/// evaluation alone.
pub fn generate_training_samples(
    seeds: &[&str],
    plies: u32,
    seed: u64,
    label_depth: Option<u8>,
) -> Result<Vec<TrainingSample>, String> {
    let mut rng = Lcg::new(seed);
    let mut labeler = Searcher::default();
    let mut samples = Vec::new();
    for fen in seeds {
        let mut board = Board::from_fen(fen)?;
        for _ in 0..plies {
            let stm = board.side_to_move;
            let target = match label_depth {
                None => hand_crafted_evaluation(&board),
                Some(depth) => labeler
                    .search(
                        &board,
                        SearchLimits {
                            depth: Some(depth),
                            ..SearchLimits::default()
                        },
                    )
                    .score_cp
                    .clamp(-TARGET_CLAMP, TARGET_CLAMP),
            };
            samples.push(TrainingSample {
                own: active_features(&board, stm),
                opp: active_features(&board, opposite(stm)),
                target: target as f32,
            });
            let moves = board.generate_legal_move_list();
            if moves.is_empty() {
                break;
            }
            let choice = (rng.next_u64() % moves.as_slice().len() as u64) as usize;
            let mv = moves.as_slice()[choice];
            board.make_move(mv)?;
        }
    }
    Ok(samples)
}

#[derive(Clone, Copy, Debug)]
pub struct TrainConfig {
    pub hidden: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            hidden: 64,
            epochs: 20,
            // The WDL-sigmoid gradient is small (bounded), so it needs a much
            // larger step than the old raw-centipawn MSE loss.
            learning_rate: 8_000.0,
            seed: 0x51A5_2C0D_E5EE_D001,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrainReport {
    pub initial_loss: f32,
    pub final_loss: f32,
    pub samples: usize,
}

/// A float network with the same shape as [`Nnue`], trained by SGD.
struct FloatNet {
    hidden: usize,
    feature_weights: Vec<f32>,
    feature_bias: Vec<f32>,
    output_weights: Vec<f32>,
    output_bias: f32,
}

impl FloatNet {
    fn new(hidden: usize, rng: &mut Lcg) -> Self {
        // Small symmetric initialisation keeps early activations in range.
        let mut init = |count: usize, scale: f32| {
            (0..count)
                .map(|_| (rng.next_unit() - 0.5) * scale)
                .collect::<Vec<f32>>()
        };
        Self {
            hidden,
            feature_weights: init(INPUT_DIMENSION * hidden, 0.2),
            feature_bias: init(hidden, 0.2),
            output_weights: init(2 * hidden, 0.2),
            output_bias: 0.0,
        }
    }

    fn accumulate(&self, features: &[usize]) -> Vec<f32> {
        let mut acc = self.feature_bias.clone();
        for &feature in features {
            let base = feature * self.hidden;
            for (value, weight) in acc.iter_mut().zip(&self.feature_weights[base..base + self.hidden]) {
                *value += weight;
            }
        }
        acc
    }

    /// Forward pass returning the prediction plus the pre-activation
    /// accumulators, which the backward pass needs for the clip gradient.
    fn forward(&self, sample: &TrainingSample) -> (f32, Vec<f32>, Vec<f32>) {
        let own = self.accumulate(&sample.own);
        let opp = self.accumulate(&sample.opp);
        let mut output = self.output_bias;
        for i in 0..self.hidden {
            output += clip(own[i]) * self.output_weights[i];
            output += clip(opp[i]) * self.output_weights[self.hidden + i];
        }
        (output / OUTPUT_SCALE, own, opp)
    }

    fn sgd_step(&mut self, sample: &TrainingSample, learning_rate: f32) -> f32 {
        let (prediction, own, opp) = self.forward(sample);
        // Win-probability (WDL) loss: squash both prediction and target through a
        // sigmoid so the loss lives in [0, 1] and its gradient is bounded — a
        // huge tactical target contributes a bounded, non-dominating step.
        let predicted_wp = win_probability(prediction);
        let target_wp = win_probability(sample.target);
        let error_wp = predicted_wp - target_wp;
        // d(loss)/d(prediction) = error_wp * sigmoid'(pred) then chain through the
        // OUTPUT_SCALE divisor to reach d(loss)/d(output).
        let grad_output =
            error_wp * predicted_wp * (1.0 - predicted_wp) / WDL_SCALE / OUTPUT_SCALE;

        self.output_bias -= learning_rate * grad_output;
        for i in 0..self.hidden {
            let own_act = clip(own[i]);
            let opp_act = clip(opp[i]);
            let grad_own_w = grad_output * own_act;
            let grad_opp_w = grad_output * opp_act;

            // Gradients flowing back into each accumulator through the clip.
            let grad_own_acc = grad_output * self.output_weights[i] * clip_grad(own[i]);
            let grad_opp_acc = grad_output * self.output_weights[self.hidden + i] * clip_grad(opp[i]);

            self.output_weights[i] -= learning_rate * grad_own_w;
            self.output_weights[self.hidden + i] -= learning_rate * grad_opp_w;

            self.feature_bias[i] -= learning_rate * (grad_own_acc + grad_opp_acc);
            for &feature in &sample.own {
                let idx = feature * self.hidden + i;
                self.feature_weights[idx] -= learning_rate * grad_own_acc;
            }
            for &feature in &sample.opp {
                let idx = feature * self.hidden + i;
                self.feature_weights[idx] -= learning_rate * grad_opp_acc;
            }
        }
        0.5 * error_wp * error_wp
    }

    fn mean_loss(&self, samples: &[TrainingSample]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let total: f32 = samples
            .iter()
            .map(|sample| {
                let (prediction, _, _) = self.forward(sample);
                let error_wp = win_probability(prediction) - win_probability(sample.target);
                0.5 * error_wp * error_wp
            })
            .sum();
        total / samples.len() as f32
    }

    fn quantize(&self) -> Result<Nnue, String> {
        let feature_weights = self.feature_weights.iter().map(round_i16).collect();
        let feature_bias = self.feature_bias.iter().map(round_i16).collect();
        let output_weights = self.output_weights.iter().map(round_i16).collect();
        let output_bias = self.output_bias.round() as i32;
        Nnue::from_parameters(
            self.hidden,
            feature_weights,
            feature_bias,
            output_weights,
            output_bias,
        )
    }
}

/// Trains a network on the samples and returns the exported quantised network
/// alongside a report of the loss reduction.
pub fn train_nnue(
    samples: &[TrainingSample],
    config: TrainConfig,
) -> Result<(Nnue, TrainReport), String> {
    if samples.is_empty() {
        return Err("cannot train on an empty sample set".to_string());
    }
    let mut rng = Lcg::new(config.seed);
    let mut net = FloatNet::new(config.hidden, &mut rng);
    let initial_loss = net.mean_loss(samples);

    let mut order: Vec<usize> = (0..samples.len()).collect();
    for _ in 0..config.epochs {
        rng.shuffle(&mut order);
        for &index in &order {
            net.sgd_step(&samples[index], config.learning_rate);
        }
    }

    let final_loss = net.mean_loss(samples);
    let network = net.quantize()?;
    Ok((
        network,
        TrainReport {
            initial_loss,
            final_loss,
            samples: samples.len(),
        },
    ))
}

/// Maps a centipawn evaluation to a win probability in `(0, 1)`.
fn win_probability(centipawns: f32) -> f32 {
    1.0 / (1.0 + (-centipawns / WDL_SCALE).exp())
}

fn clip(value: f32) -> f32 {
    value.clamp(0.0, ACTIVATION_CLIP)
}

fn clip_grad(value: f32) -> f32 {
    if value > 0.0 && value < ACTIVATION_CLIP {
        1.0
    } else {
        0.0
    }
}

fn round_i16(value: &f32) -> i16 {
    value.round().clamp(-32768.0, 32767.0) as i16
}

/// A tiny deterministic linear-congruential PRNG for reproducible training.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0xD1B5_4A32_D192_ED03,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// A float in `[0, 1)`.
    fn next_unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn shuffle(&mut self, items: &mut [usize]) {
        for i in (1..items.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{generate_training_samples, train_nnue, TrainConfig};

    const SEEDS: &[&str] = &[
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3",
    ];

    #[test]
    fn training_generates_labelled_samples() {
        let samples = generate_training_samples(SEEDS, 8, 1, None).expect("samples");
        assert!(!samples.is_empty());
        // Every sample records the pieces on the board from both perspectives.
        assert!(samples.iter().all(|sample| !sample.own.is_empty() && !sample.opp.is_empty()));
    }

    #[test]
    fn deep_search_labels_differ_from_the_static_evaluation() {
        let static_samples = generate_training_samples(SEEDS, 6, 3, None).expect("static");
        let search_samples = generate_training_samples(SEEDS, 6, 3, Some(3)).expect("search");
        assert_eq!(static_samples.len(), search_samples.len());
        // Same positions (identical features), but the depth-3 search labels
        // reflect tactics the static evaluation misses, so some targets differ.
        assert!(
            static_samples
                .iter()
                .zip(&search_samples)
                .any(|(a, b)| (a.target - b.target).abs() > f32::EPSILON),
            "deep-search labels should differ from static labels on some positions",
        );
    }

    #[test]
    fn deep_search_labels_do_not_diverge() {
        // Depth-labelled targets are large and high-variance; before gradient
        // clipping they made SGD diverge (loss increasing). Training must now
        // reduce the loss on them.
        let samples = generate_training_samples(SEEDS, 8, 11, Some(2)).expect("samples");
        let config = TrainConfig {
            hidden: 32,
            epochs: 30,
            learning_rate: 8_000.0,
            seed: 4_242,
        };
        let (_net, report) = train_nnue(&samples, config).expect("training succeeds");
        assert!(
            report.final_loss < report.initial_loss,
            "deep-search training should reduce loss, not diverge: {} -> {}",
            report.initial_loss,
            report.final_loss
        );
    }

    #[test]
    fn training_reduces_loss_and_beats_a_zero_predictor() {
        let samples = generate_training_samples(SEEDS, 12, 7, None).expect("samples");
        // A zero-centipawn predictor maps to a 0.5 win probability everywhere.
        let zero_loss: f32 = samples
            .iter()
            .map(|sample| {
                let error = 0.5 - super::win_probability(sample.target);
                0.5 * error * error
            })
            .sum::<f32>()
            / samples.len() as f32;

        let config = TrainConfig {
            hidden: 32,
            epochs: 40,
            learning_rate: 8_000.0,
            seed: 20_260_714,
        };
        let (net, report) = train_nnue(&samples, config).expect("training succeeds");

        assert!(
            report.final_loss < report.initial_loss,
            "SGD should reduce loss: {} -> {}",
            report.initial_loss,
            report.final_loss
        );
        // The fitted network predicts the teacher better than always guessing 0.
        assert!(
            report.final_loss < zero_loss,
            "final loss {} should beat the zero-predictor loss {zero_loss}",
            report.final_loss
        );

        // The exported network round-trips and evaluates within bounds.
        let restored =
            engine_search::Nnue::from_bytes(&net.to_bytes()).expect("exported net round-trips");
        let board = engine_core::Board::startpos();
        assert!(restored.evaluate(&board, engine_core::Color::White).abs() <= 20_000);
    }
}
