//! Tier 1 of decision-mutation label augmentation (Phase 4): replay a search node's
//! alpha-beta cutoff logic *offline* under a mutated move order, from the v4 telemetry
//! columns (`node_id`, `move_score`, `node_alpha`, `node_beta`), synthesizing labels for
//! orderings the base search never took — with no extra search.
//!
//! What replay from recorded scores can and cannot do (worth being precise, it shapes
//! the whole idea):
//!
//! - **`caused_cutoff` gains no new signal.** A move cuts iff `move_score >= node_beta`.
//!   Among the *searched* moves a node only ever records one such move (the loop breaks at
//!   the first cutoff), and every searched non-cutter genuinely has `move_score <
//!   node_beta`. So the recorded `caused_cutoff` already *is* the counterfactual
//!   "would this move cut if ordered first?" for every searched move. The moves that could
//!   be new cutters are the *unsearched* tail (after the cutoff, or LMP-pruned) — and
//!   those have no recorded score, so only re-search (Tier 2) can label them.
//! - **`raised_alpha` gains real new signal.** Whether a move raises alpha depends on the
//!   running alpha, which depends on order. Replaying a node under different orders
//!   produces genuinely different, valid `raised_alpha` labels for the same
//!   (order-independent) features — the denser target the plan flags.
//!
//! So this pass emits full rows (features untouched) with `raised_alpha`, `caused_cutoff`
//! and `move_index` recomputed for a mutated order, and reports how faithfully replaying
//! the *actual* order reproduces the recorded labels (the recorded-score model's
//! fidelity). It is also the substrate Tier 2 (sampled re-search) builds on.

use std::io::{BufRead, BufReader, Write};

/// Column indices resolved by name against the telemetry header, so a schema change
/// re-maps instead of silently shifting.
struct Columns {
    pos_id: usize,
    node_id: usize,
    move_index: usize,
    lmp_pruned: usize,
    raised_alpha: usize,
    caused_cutoff: usize,
    move_score: usize,
    node_alpha: usize,
    node_beta: usize,
    count: usize,
}

fn column(header: &[&str], name: &str) -> Result<usize, String> {
    header
        .iter()
        .position(|column| *column == name)
        .ok_or_else(|| format!("telemetry header is missing the `{name}` column (need v4 telemetry)"))
}

impl Columns {
    fn resolve(header: &[&str]) -> Result<Self, String> {
        Ok(Self {
            pos_id: column(header, "pos_id")?,
            node_id: column(header, "node_id")?,
            move_index: column(header, "move_index")?,
            lmp_pruned: column(header, "lmp_pruned")?,
            raised_alpha: column(header, "raised_alpha")?,
            caused_cutoff: column(header, "caused_cutoff")?,
            move_score: column(header, "move_score")?,
            node_alpha: column(header, "node_alpha")?,
            node_beta: column(header, "node_beta")?,
            count: header.len(),
        })
    }
}

/// One searched move of a node, carrying its raw TSV fields (re-emitted with labels
/// overwritten) and the parsed values replay needs.
#[derive(Clone)]
struct NodeMove {
    fields: Vec<String>,
    move_index: i64,
    move_score: i64,
    rec_raised_alpha: bool,
    rec_caused_cutoff: bool,
}

/// Deterministic SplitMix64, seeded per node so a node's permutations are reproducible
/// and vary across nodes without any RNG dependency.
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

    /// Fisher-Yates over an index slice.
    fn shuffle(&mut self, slice: &mut [usize]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            slice.swap(i, j);
        }
    }
}

/// Replays a node's cutoff logic over `moves` taken in `order` (indices into `moves`),
/// starting the running alpha at `node_alpha`. Returns, per position in `order`, the
/// `(raised_alpha, caused_cutoff)` label — stopping after the first cutter (its entry is
/// included; later moves are "unsearched" in this counterfactual and omitted). Mirrors the
/// search's own arithmetic: a move raises alpha iff its score beats the running alpha, and
/// cuts iff its score reaches `node_beta`.
fn replay(
    moves: &[NodeMove],
    order: &[usize],
    node_alpha: i64,
    node_beta: i64,
) -> Vec<(bool, bool)> {
    let mut running_alpha = node_alpha;
    let mut labels = Vec::with_capacity(order.len());
    for &idx in order {
        let score = moves[idx].move_score;
        let raised = score > running_alpha;
        let cutoff = score >= node_beta;
        labels.push((raised, cutoff));
        if cutoff {
            break;
        }
        if raised {
            running_alpha = score;
        }
    }
    labels
}

/// What a mutation pass produced, reported so a caller can trust the augmentation.
#[derive(Clone, Copy, Debug, Default)]
pub struct MutateSummary {
    /// Nodes with >= 1 searched move seen.
    pub nodes: u64,
    /// Nodes with >= 2 searched moves (the ones a mutation can reorder).
    pub mutable_nodes: u64,
    /// Synthetic rows written.
    pub emitted_rows: u64,
    /// Nodes whose actual-order replay reproduced every recorded label exactly.
    pub faithful_nodes: u64,
    /// Nodes checked for fidelity (same as `nodes`).
    pub checked_nodes: u64,
}

impl MutateSummary {
    /// Fraction of nodes whose recorded labels the replay reproduces from the recorded
    /// scores — the recorded-score model's fidelity. `None` if nothing was checked.
    pub fn fidelity(&self) -> Option<f64> {
        (self.checked_nodes > 0).then(|| self.faithful_nodes as f64 / self.checked_nodes as f64)
    }
}

fn parse_bool(field: &str) -> bool {
    field == "1"
}

/// Flush one node's accumulated searched moves: check actual-order fidelity, then emit
/// `mutations_per_node` shuffled orderings with recomputed labels.
fn flush_node<W: Write>(
    writer: &mut W,
    cols: &Columns,
    moves: &mut Vec<NodeMove>,
    node_key: u64,
    mutations_per_node: u64,
    seed: u64,
    summary: &mut MutateSummary,
) -> Result<(), String> {
    if moves.is_empty() {
        return Ok(());
    }
    summary.nodes += 1;
    summary.checked_nodes += 1;
    // Actual order = recorded move_index ascending.
    moves.sort_by_key(|m| m.move_index);
    let node_alpha = moves[0].fields[cols.node_alpha]
        .parse::<i64>()
        .map_err(|_| "bad node_alpha".to_string())?;
    let node_beta = moves[0].fields[cols.node_beta]
        .parse::<i64>()
        .map_err(|_| "bad node_beta".to_string())?;

    // Fidelity: replay the actual order and compare to the recorded labels.
    let actual: Vec<usize> = (0..moves.len()).collect();
    let replayed = replay(moves, &actual, node_alpha, node_beta);
    let faithful = replayed.len() == moves.len()
        && replayed
            .iter()
            .zip(moves.iter())
            .all(|((ra, cc), m)| *ra == m.rec_raised_alpha && *cc == m.rec_caused_cutoff);
    if faithful {
        summary.faithful_nodes += 1;
    }

    if moves.len() >= 2 {
        summary.mutable_nodes += 1;
        let mut rng = Rng::new(seed ^ node_key.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        for _ in 0..mutations_per_node {
            let mut order: Vec<usize> = (0..moves.len()).collect();
            rng.shuffle(&mut order);
            let labels = replay(moves, &order, node_alpha, node_beta);
            for (new_index, ((raised, cutoff), &idx)) in labels.iter().zip(order.iter()).enumerate() {
                let mut fields = moves[idx].fields.clone();
                fields[cols.move_index] = new_index.to_string();
                fields[cols.raised_alpha] = u8::from(*raised).to_string();
                fields[cols.caused_cutoff] = u8::from(*cutoff).to_string();
                writeln!(writer, "{}", fields.join("\t"))
                    .map_err(|error| format!("failed to write mutated row: {error}"))?;
                summary.emitted_rows += 1;
            }
        }
    }
    moves.clear();
    Ok(())
}

/// Reads a v4 telemetry TSV, groups rows by `(pos_id, node_id)`, and writes
/// `mutations_per_node` shuffled-order replays per node (searched moves only) to
/// `writer`, in the *same* schema so the output concatenates with real telemetry and
/// feeds `train-policy` unchanged. The header is written first. Rows are streamed and a
/// node is flushed as soon as its group ends — the stream is grouped (all of one node's
/// rows share a `(pos_id, node_id)` and, within a search, arrive before the id is reused),
/// which `run_gen_search_telemetry` guarantees since `node_id` is the entry node counter.
pub fn gen_mutated_labels<R: std::io::Read, W: Write>(
    reader: R,
    mut writer: W,
    mutations_per_node: u64,
    seed: u64,
) -> Result<MutateSummary, String> {
    let mut lines = BufReader::with_capacity(1 << 20, reader).lines();
    let header_line = lines
        .next()
        .transpose()
        .map_err(|error| format!("failed to read telemetry header: {error}"))?
        .ok_or_else(|| "telemetry input is empty".to_string())?;
    let header: Vec<&str> = header_line.trim_end().split('\t').collect();
    let cols = Columns::resolve(&header)?;
    writeln!(writer, "{}", header_line.trim_end())
        .map_err(|error| format!("failed to write header: {error}"))?;

    let mut summary = MutateSummary::default();
    let mut group: Vec<NodeMove> = Vec::new();
    let mut current_key: Option<(u64, u64)> = None;

    for line in lines {
        let line = line.map_err(|error| format!("failed to read telemetry: {error}"))?;
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            continue;
        }
        let fields: Vec<String> = trimmed.split('\t').map(str::to_string).collect();
        if fields.len() != cols.count {
            continue; // short row: an older schema; skip rather than mis-map
        }
        let pos_id = fields[cols.pos_id].parse::<u64>().map_err(|_| "bad pos_id".to_string())?;
        let node_id = fields[cols.node_id].parse::<u64>().map_err(|_| "bad node_id".to_string())?;
        let key = (pos_id, node_id);
        if current_key != Some(key) {
            if let Some((pos, node)) = current_key {
                let node_key = pos.wrapping_mul(0x1000_0000_0000_0001).wrapping_add(node);
                flush_node(
                    &mut writer, &cols, &mut group, node_key, mutations_per_node, seed, &mut summary,
                )?;
            }
            current_key = Some(key);
        }
        // Only searched moves carry a real score; LMP-pruned rows were never searched.
        if parse_bool(&fields[cols.lmp_pruned]) {
            continue;
        }
        let move_index = fields[cols.move_index].parse::<i64>().map_err(|_| "bad move_index".to_string())?;
        let move_score = fields[cols.move_score].parse::<i64>().map_err(|_| "bad move_score".to_string())?;
        let rec_raised_alpha = parse_bool(&fields[cols.raised_alpha]);
        let rec_caused_cutoff = parse_bool(&fields[cols.caused_cutoff]);
        group.push(NodeMove { fields, move_index, move_score, rec_raised_alpha, rec_caused_cutoff });
    }
    if let Some((pos, node)) = current_key {
        let node_key = pos.wrapping_mul(0x1000_0000_0000_0001).wrapping_add(node);
        flush_node(
            &mut writer, &cols, &mut group, node_key, mutations_per_node, seed, &mut summary,
        )?;
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_search::TELEMETRY_TSV_HEADER;

    /// Build a v4 telemetry row from a header, setting only the columns replay reads.
    fn row(
        header: &[&str],
        pos_id: u64,
        node_id: u64,
        move_index: u64,
        lmp_pruned: bool,
        raised_alpha: bool,
        caused_cutoff: bool,
        move_score: i64,
        node_alpha: i64,
        node_beta: i64,
    ) -> String {
        header
            .iter()
            .map(|&col| match col {
                "pos_id" => pos_id.to_string(),
                "node_id" => node_id.to_string(),
                "move_index" => move_index.to_string(),
                "lmp_pruned" => u8::from(lmp_pruned).to_string(),
                "raised_alpha" => u8::from(raised_alpha).to_string(),
                "caused_cutoff" => u8::from(caused_cutoff).to_string(),
                "move_score" => move_score.to_string(),
                "node_alpha" => node_alpha.to_string(),
                "node_beta" => node_beta.to_string(),
                _ => "0".to_string(),
            })
            .collect::<Vec<_>>()
            .join("\t")
    }

    /// A node where moves 0,1 don't reach beta and move 2 cuts. Alpha 0, beta 100,
    /// scores [10, 40, 120]. Actual-order replay must reproduce: move0 raises (10>0),
    /// move1 raises (40>10), move2 cuts (120>=100). The tool must report fidelity 1.0 and
    /// emit rows whose labels obey the same arithmetic under shuffles.
    #[test]
    fn replay_reproduces_recorded_labels_and_mutates() {
        let header: Vec<&str> = TELEMETRY_TSV_HEADER.split('\t').collect();
        let mut tsv = String::from(TELEMETRY_TSV_HEADER);
        tsv.push('\n');
        // (raised, cutoff) recorded to match the arithmetic so fidelity is exact.
        tsv.push_str(&row(&header, 0, 1, 0, false, true, false, 10, 0, 100));
        tsv.push('\n');
        tsv.push_str(&row(&header, 0, 1, 1, false, true, false, 40, 0, 100));
        tsv.push('\n');
        tsv.push_str(&row(&header, 0, 1, 2, false, true, true, 120, 0, 100));
        tsv.push('\n');

        let mut out = Vec::new();
        let summary = gen_mutated_labels(tsv.as_bytes(), &mut out, 5, 7).expect("mutate");
        assert_eq!(summary.nodes, 1);
        assert_eq!(summary.mutable_nodes, 1);
        assert_eq!(summary.faithful_nodes, 1, "actual-order replay must reproduce records");
        assert_eq!(summary.fidelity(), Some(1.0));

        // Every emitted row must obey replay arithmetic: caused_cutoff iff move_score>=100.
        let text = String::from_utf8(out).unwrap();
        let mut lines = text.lines();
        let out_header: Vec<&str> = lines.next().unwrap().split('\t').collect();
        let cols = Columns::resolve(&out_header).unwrap();
        let mut saw_cutoff = false;
        let mut rows = 0;
        for line in lines {
            let f: Vec<&str> = line.split('\t').collect();
            let score: i64 = f[cols.move_score].parse().unwrap();
            let cutoff = f[cols.caused_cutoff] == "1";
            assert_eq!(cutoff, score >= 100, "cutoff label must equal move_score>=beta");
            saw_cutoff |= cutoff;
            rows += 1;
        }
        assert_eq!(rows as u64, summary.emitted_rows);
        assert!(saw_cutoff, "the cutter (score 120) should be reached in some shuffle");
    }

    /// A cut node with a single searched move can't be reordered — no rows, but it still
    /// counts toward fidelity.
    #[test]
    fn single_move_node_emits_nothing_but_is_checked() {
        let header: Vec<&str> = TELEMETRY_TSV_HEADER.split('\t').collect();
        let mut tsv = String::from(TELEMETRY_TSV_HEADER);
        tsv.push('\n');
        tsv.push_str(&row(&header, 0, 1, 0, false, true, true, 200, 0, 100));
        tsv.push('\n');
        let mut out = Vec::new();
        let summary = gen_mutated_labels(tsv.as_bytes(), &mut out, 4, 1).expect("mutate");
        assert_eq!(summary.nodes, 1);
        assert_eq!(summary.mutable_nodes, 0);
        assert_eq!(summary.emitted_rows, 0);
        assert_eq!(summary.faithful_nodes, 1);
        // Output is header-only.
        assert_eq!(String::from_utf8(out).unwrap().lines().count(), 1);
    }
}
