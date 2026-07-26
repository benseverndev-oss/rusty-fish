//! Phase 1 search telemetry: a per-move-decision dataset emitted from the
//! alpha-beta move loop.
//!
//! This module is pure data plumbing. It carries no search logic and, crucially,
//! never influences a search decision. When a [`Searcher`](crate::Searcher) has
//! no collector installed the record site is a single cheap `Option` branch that
//! allocates nothing; with a collector installed the *only* observable
//! difference is that [`MoveDecision`] records accumulate. The engine's
//! byte-identical-search test enforces this invariant (telemetry off vs on must
//! produce the same best move, score, and final node count).
//!
//! One [`MoveDecision`] is recorded per move *considered* in a node's move loop,
//! including a move skipped by late-move pruning (its outcome fields are zeroed
//! and it is not searched — counterfactual verification is a later phase).

/// A single move-decision observation from one node's alpha-beta move loop.
///
/// All fields are fixed-width primitives so a `Vec<MoveDecision>` is a flat,
/// cache-friendly buffer and the TSV projection is lossless. Booleans serialize
/// as `0`/`1`; scores and counts serialize as decimal integers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MoveDecision {
    // --- Context: what the node knew before deciding how to search this move ---
    /// Remaining search depth at this node.
    pub depth: u8,
    /// Distance from the root (`ply` in the search).
    pub ply: u16,
    /// 0-based index of this move in the node's ordered move list.
    pub move_index: u16,
    /// The move is quiet (non-capture, non-promotion) per `is_quiet_move`.
    pub is_quiet: bool,
    /// The move is a TT move, a killer, or the counter-move (ordered first).
    pub is_priority: bool,
    /// The node is a PV node: a full-width window remained (`beta - alpha > 1`).
    pub pv_node: bool,
    /// The move gives check (the side-to-move is in check *after* the move).
    pub gives_check: bool,
    /// The node's static eval used by razoring / reverse-futility pruning, or `0`
    /// when the node did not compute one (static pruning did not apply).
    pub static_eval: i32,
    /// Search extension applied to this move (plies).
    pub extension: u8,
    /// Late-move reduction applied to this move (plies).
    pub reduction: u8,

    // --- Decision / outcome ---
    /// The move was skipped by late-move pruning; it was *not* searched and all
    /// outcome fields below are zero.
    pub lmp_pruned: bool,
    /// The searched move raised alpha (`score > alpha` before the alpha update).
    pub raised_alpha: bool,
    /// The searched move caused a beta cutoff (`alpha >= beta` after the update).
    pub caused_cutoff: bool,
    /// The late-move-reduction re-search fired for this move.
    pub needed_lmr_research: bool,
    /// The PVS full-window re-search fired for this move.
    pub needed_pvs_research: bool,
    /// Nodes spent searching this move's subtree (delta of the searcher's node
    /// counter across this move's child searches and any re-searches).
    pub subtree_nodes: u64,

    // --- v2 context (APPENDED, never inserted) ---
    // Measured finding: the learned-LMR model saturated at val AUC ~0.94 across both
    // more data and more capacity — it is *feature*-limited. These are the cheap,
    // high-signal fields available at the same hook. They are appended at the end so
    // every existing column index (notably the target and the lmp_pruned filter) is
    // unchanged and older readers keep working.
    /// The move's history-heuristic score at this node.
    pub history_score: i32,
    /// The move is the transposition-table move (`is_priority` split into its parts,
    /// which carry more signal separately than as one OR'd flag).
    pub is_tt_move: bool,
    /// The move is a killer at this ply.
    pub is_killer: bool,
    /// The move is the counter-move for the previous move.
    pub is_counter: bool,
    /// The move captures (including en passant).
    pub is_capture: bool,
    /// The move promotes.
    pub is_promotion: bool,
    /// The *node* is in check (distinct from `gives_check`, which is the child).
    pub node_in_check: bool,
    /// Depth of the transposition-table entry at this node, `0` when there is none.
    pub tt_depth: u8,

    // --- v3 context (APPENDED, never inserted) ---
    // Move-ordering signal for the Phase 4 policy. The v1/v2 fields describe how the
    // node *searched* a move; these describe the move *as the orderer saw it*, which is
    // what a learned move-ordering policy has to reproduce and improve on. Captured at
    // the decision point (pre-move board, history snapshotted before the subtree) so
    // they never leak the outcome, and appended so every existing column index — and
    // the `train_lmr.py` FEATURE_COLS that reference them positionally — is unchanged.
    /// The classical move-ordering score (`move_order_score`): TT/capture-SEE/promotion/
    /// counter/killer priorities plus the history score. The residual baseline a learned
    /// policy corrects (`order = classical + learned_correction`), mirroring learned LMR.
    pub order_score: i32,
    /// Static exchange evaluation of the move, in centipawns; `0` for a non-capture. The
    /// single most discriminative capture-ordering signal, and absent from the v1/v2 set.
    pub see: i32,
    /// Kind of the moving piece, `1..=6` (pawn..king); `0` only if the square is somehow
    /// empty (never, for a legal move — kept as a total encoding).
    pub mover_piece: u8,
    /// Kind of the captured piece, `1..=6` (pawn..king), `0` for a non-capture. En
    /// passant records the captured pawn as `1`, matching `is_capture`.
    pub captured_piece: u8,

    // --- v4 context (APPENDED, never inserted) ---
    // Replay support for decision-mutation label augmentation. `caused_cutoff` is
    // order-dependent — only the *first* move (in search order) whose score reaches beta
    // is credited, because the loop then breaks. These four columns let an offline pass
    // replay a node's cutoff logic under a *different* move order and re-derive the
    // labels, synthesizing counterfactual "this move would have cut first" examples with
    // no extra search. Appended, so every existing column index is unchanged.
    /// Groups the move-decisions of one search node. Unique within a single searched
    /// position (it is the node counter at node entry); combine with `pos_id` across
    /// positions. A node's rows are *not* contiguous in the stream — child subtrees emit
    /// between them — so replay groups by `(pos_id, node_id)`.
    pub node_id: u64,
    /// The score the search assigned this move (this node's perspective), the value
    /// compared against `node_beta`/the running alpha. `0` for an unsearched move
    /// (`lmp_pruned`). Order-dependent windows (PVS null-window, LMR) make it a bound for
    /// late fail-low moves, but a move whose true value reaches beta fails high and is
    /// re-searched to an accurate score, so the `>= node_beta` cutoff test stays faithful.
    pub move_score: i32,
    /// The node's alpha at the start of its move loop (after any TT raise). The running
    /// alpha a replay starts from.
    pub node_alpha: i32,
    /// The node's beta (constant across the move loop). A move cuts iff `move_score >=
    /// node_beta`.
    pub node_beta: i32,

    // --- v5 context (APPENDED, never inserted) ---
    // Move-quality feature widening for the Phase 4 policy. `gives_check` already existed
    // (this only promotes it to a model feature); `psqt_delta` is new: the tapered-midgame
    // piece-square gain of the move (destination minus origin for the moving piece), a
    // cheap positional signal computed from the pre-move board so it never leaks the
    // outcome. Appended, so every existing column index is unchanged.
    /// Tapered-midgame piece-square-table gain of the move for the moving piece
    /// (`bonus(to) - bonus(from)`), `0` if the from-square is somehow empty.
    pub psqt_delta: i32,
}

/// TSV header row for the v1 schema, including the leading `pos_id` column that
/// the dataset generator prepends. Kept adjacent to [`MoveDecision::to_tsv_row`]
/// so the column order stays single-sourced.
pub const TELEMETRY_TSV_HEADER: &str = "pos_id\tdepth\tply\tmove_index\tis_quiet\tis_priority\tpv_node\tgives_check\tstatic_eval\textension\treduction\tlmp_pruned\traised_alpha\tcaused_cutoff\tneeded_lmr_research\tneeded_pvs_research\tsubtree_nodes\thistory_score\tis_tt_move\tis_killer\tis_counter\tis_capture\tis_promotion\tnode_in_check\ttt_depth\torder_score\tsee\tmover_piece\tcaptured_piece\tnode_id\tmove_score\tnode_alpha\tnode_beta\tpsqt_delta";

/// The learned-LMR model's input columns, **by header name**, in the exact order the
/// model expects them (matching `train_lmr.py`'s `FEATURE_COLS` and the feature
/// vector the search builds in `negamax`).
///
/// Named rather than positional so the dataset exporter resolves them against the
/// TSV's own header. A future schema change that inserts or reorders a column then
/// re-maps automatically instead of silently shifting every feature by one.
pub const LMR_FEATURE_COLUMNS: [&str; crate::lmr_model::LMR_FEATURES] = [
    "depth",
    "ply",
    "move_index",
    "is_quiet",
    "is_priority",
    "pv_node",
    "gives_check",
    "static_eval",
    "extension",
    "reduction",
    "history_score",
    "is_tt_move",
    "is_killer",
    "is_counter",
    "is_capture",
    "is_promotion",
    "node_in_check",
    "tt_depth",
];

/// Label column: did the searched move raise alpha.
pub const LMR_TARGET_COLUMN: &str = "raised_alpha";

/// Rows with this column set are dropped from the dataset — a late-move-pruned move
/// was never searched, so its outcome columns are zeros rather than observations.
pub const LMR_FILTER_COLUMN: &str = "lmp_pruned";

/// Inclusive clamp bounds applied to unbounded feature columns before they reach the
/// model, keyed by header name. Training and inference must agree on these: the
/// trainer fits its standardization on clamped values, so an unclamped mate score or
/// runaway history entry at inference lands outside the distribution the stored
/// mean/scale describe. Mirrored by `lmr_model`'s `STATIC_EVAL_CLAMP`/`HISTORY_CLAMP`
/// and by `train_lmr.py`.
pub const LMR_FEATURE_CLAMPS: [(&str, f32); 2] =
    [("static_eval", 2000.0), ("history_score", 20000.0)];

/// The move-ordering policy's input columns (Phase 4), **by header name**, in the exact
/// order the model expects them (matching `policy_model::POLICY_FEATURES`'s documented
/// order and the trainer's `POLICY_FEATURE_COLUMNS`).
///
/// These are the signals the move orderer has *before* a move is searched, resolved by
/// name against the TSV header for the same reason the LMR set is. Deliberately absent:
/// `move_index` and `order_score` (both encode the classical rank the policy is trying to
/// improve on — the classical score still enters the search as the residual base, so the
/// model does not need it as an input), and `reduction` / `extension` / `gives_check`
/// (decided or costly only after ordering). `order_score` remains a *telemetry column* —
/// it is just no longer selected as a model feature.
pub const POLICY_FEATURE_COLUMNS: [&str; crate::policy_model::POLICY_FEATURES] = [
    "depth",
    "ply",
    "pv_node",
    "node_in_check",
    "static_eval",
    "is_quiet",
    "is_capture",
    "is_promotion",
    "is_tt_move",
    "is_killer",
    "is_counter",
    "history_score",
    "tt_depth",
    "see",
    "mover_piece",
    "captured_piece",
    "gives_check",
    "psqt_delta",
];

/// Policy label column: did the searched move cause a beta cutoff — i.e. was it the move
/// the node should have ordered first. Denser targets like `raised_alpha` are available;
/// `caused_cutoff` is the sharper "search this first" signal for ordering.
pub const POLICY_TARGET_COLUMN: &str = "caused_cutoff";

/// Inclusive clamp bounds for the policy's unbounded feature columns, keyed by header
/// name. Mirrored by `policy_model`'s `POLICY_CLAMP_INDICES` and applied identically by
/// the trainer, for the same standardization reason as [`LMR_FEATURE_CLAMPS`].
pub const POLICY_FEATURE_CLAMPS: [(&str, f32); 4] = [
    ("static_eval", 2000.0),
    ("history_score", 20000.0),
    ("see", 2000.0),
    ("psqt_delta", 100.0),
];

impl MoveDecision {
    /// Serializes this record as one TSV row, prefixed with `pos_id`. The column
    /// order matches [`TELEMETRY_TSV_HEADER`]. Booleans render as `0`/`1`.
    pub fn to_tsv_row(&self, pos_id: u64) -> String {
        let b = |flag: bool| u8::from(flag);
        format!(
            "{pos_id}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\
             \t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.depth,
            self.ply,
            self.move_index,
            b(self.is_quiet),
            b(self.is_priority),
            b(self.pv_node),
            b(self.gives_check),
            self.static_eval,
            self.extension,
            self.reduction,
            b(self.lmp_pruned),
            b(self.raised_alpha),
            b(self.caused_cutoff),
            b(self.needed_lmr_research),
            b(self.needed_pvs_research),
            self.subtree_nodes,
            self.history_score,
            b(self.is_tt_move),
            b(self.is_killer),
            b(self.is_counter),
            b(self.is_capture),
            b(self.is_promotion),
            b(self.node_in_check),
            self.tt_depth,
            self.order_score,
            self.see,
            self.mover_piece,
            self.captured_piece,
            self.node_id,
            self.move_score,
            self.node_alpha,
            self.node_beta,
            self.psqt_delta,
        )
    }
}

/// Bounded, append-only sink for [`MoveDecision`] records collected during a
/// search. The cap bounds memory on deep searches: once `cap` records are held,
/// further pushes are dropped (the search is never affected either way).
#[derive(Clone, Debug, Default)]
pub struct TelemetryCollector {
    records: Vec<MoveDecision>,
    cap: usize,
    /// Records dropped because the cap was reached — surfaced so a caller can
    /// tell a truncated dataset from a complete one.
    dropped: u64,
}

impl TelemetryCollector {
    /// Creates a collector that holds at most `cap` records.
    pub fn new(cap: usize) -> Self {
        Self {
            records: Vec::new(),
            cap,
            dropped: 0,
        }
    }

    /// Appends a record if the cap has not been reached; otherwise counts a drop.
    #[inline]
    pub fn push(&mut self, record: MoveDecision) {
        if self.records.len() < self.cap {
            self.records.push(record);
        } else {
            self.dropped += 1;
        }
    }

    /// Number of records currently held.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether no records are held.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Records dropped so far because the cap was reached.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Drains and returns the collected records, leaving the collector empty and
    /// ready to collect again (the cap and drop counter are preserved).
    pub fn take(&mut self) -> Vec<MoveDecision> {
        std::mem::take(&mut self.records)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MoveDecision, TelemetryCollector, LMR_FEATURE_CLAMPS, LMR_FEATURE_COLUMNS,
        LMR_FILTER_COLUMN, LMR_TARGET_COLUMN, POLICY_FEATURE_CLAMPS, POLICY_FEATURE_COLUMNS,
        POLICY_TARGET_COLUMN, TELEMETRY_TSV_HEADER,
    };
    use crate::policy_model::POLICY_CLAMP_INDICES;

    /// Every name the dataset exporter resolves must actually exist in the header,
    /// and no feature may be listed twice. Without this a typo would surface only as
    /// a runtime failure inside a Modal container mid-export.
    #[test]
    fn lmr_column_names_all_resolve_against_the_header() {
        let header: Vec<&str> = TELEMETRY_TSV_HEADER.split('\t').collect();
        for name in LMR_FEATURE_COLUMNS
            .iter()
            .chain([&LMR_TARGET_COLUMN, &LMR_FILTER_COLUMN])
            .chain(LMR_FEATURE_CLAMPS.iter().map(|(name, _)| name))
        {
            assert!(
                header.contains(name),
                "{name} is not a column of TELEMETRY_TSV_HEADER"
            );
        }
        let mut seen = LMR_FEATURE_COLUMNS.to_vec();
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(before, seen.len(), "duplicate column in LMR_FEATURE_COLUMNS");
        // The label must not also be an input — that would be pure leakage.
        assert!(!LMR_FEATURE_COLUMNS.contains(&LMR_TARGET_COLUMN));
    }

    /// Same guardrail as the LMR test, for the policy feature set: every name the
    /// trainer resolves must exist in the header exactly once, the label must not leak
    /// in as a feature, and the by-index clamps the inference path uses must line up
    /// with the by-name clamps — a mismatch would clamp the wrong column at inference.
    #[test]
    fn policy_column_names_all_resolve_against_the_header() {
        let header: Vec<&str> = TELEMETRY_TSV_HEADER.split('\t').collect();
        for name in POLICY_FEATURE_COLUMNS
            .iter()
            .chain([&POLICY_TARGET_COLUMN, &LMR_FILTER_COLUMN])
            .chain(POLICY_FEATURE_CLAMPS.iter().map(|(name, _)| name))
        {
            assert!(header.contains(name), "{name} is not a column of TELEMETRY_TSV_HEADER");
        }
        let mut seen = POLICY_FEATURE_COLUMNS.to_vec();
        seen.sort_unstable();
        let before = seen.len();
        seen.dedup();
        assert_eq!(before, seen.len(), "duplicate column in POLICY_FEATURE_COLUMNS");
        assert!(!POLICY_FEATURE_COLUMNS.contains(&POLICY_TARGET_COLUMN));

        // Each by-name clamp must map to the by-index clamp at the same feature slot.
        assert_eq!(POLICY_FEATURE_CLAMPS.len(), POLICY_CLAMP_INDICES.len());
        for (name, bound) in POLICY_FEATURE_CLAMPS {
            let slot = POLICY_FEATURE_COLUMNS
                .iter()
                .position(|column| *column == name)
                .expect("clamped column is a policy feature");
            let (index, index_bound) = POLICY_CLAMP_INDICES
                .iter()
                .copied()
                .find(|(index, _)| *index == slot)
                .unwrap_or_else(|| panic!("no by-index clamp for `{name}` at slot {slot}"));
            assert_eq!(index, slot);
            assert_eq!(bound, index_bound, "clamp bound mismatch for `{name}`");
        }
    }

    #[test]
    fn collector_respects_cap_and_counts_drops() {
        let mut collector = TelemetryCollector::new(2);
        collector.push(MoveDecision::default());
        collector.push(MoveDecision::default());
        collector.push(MoveDecision::default());
        assert_eq!(collector.len(), 2);
        assert_eq!(collector.dropped(), 1);
        let drained = collector.take();
        assert_eq!(drained.len(), 2);
        assert!(collector.is_empty());
        // The cap and drop counter survive a drain.
        collector.push(MoveDecision::default());
        collector.push(MoveDecision::default());
        collector.push(MoveDecision::default());
        assert_eq!(collector.len(), 2);
        assert_eq!(collector.dropped(), 2);
    }

    #[test]
    fn tsv_row_matches_header_arity_and_encodes_bools() {
        let record = MoveDecision {
            depth: 7,
            ply: 3,
            move_index: 5,
            is_quiet: true,
            is_priority: false,
            pv_node: true,
            gives_check: false,
            static_eval: -42,
            extension: 1,
            reduction: 2,
            lmp_pruned: false,
            raised_alpha: true,
            caused_cutoff: true,
            needed_lmr_research: false,
            needed_pvs_research: true,
            subtree_nodes: 1234,
            history_score: -55,
            is_tt_move: true,
            is_killer: false,
            is_counter: true,
            is_capture: false,
            is_promotion: true,
            node_in_check: false,
            tt_depth: 4,
            order_score: 1_000_042,
            see: -17,
            mover_piece: 3,
            captured_piece: 5,
            node_id: 4242,
            move_score: -128,
            node_alpha: -30,
            node_beta: 25,
            psqt_delta: 12,
        };
        let row = record.to_tsv_row(9);
        let header_cols = TELEMETRY_TSV_HEADER.split('\t').count();
        let row_cols = row.split('\t').count();
        assert_eq!(header_cols, row_cols, "row arity must match the header");
        assert_eq!(
            row,
            "9\t7\t3\t5\t1\t0\t1\t0\t-42\t1\t2\t0\t1\t1\t0\t1\t1234\t-55\t1\t0\t1\t0\t1\t0\t4\
             \t1000042\t-17\t3\t5\t4242\t-128\t-30\t25\t12"
        );
    }
}
