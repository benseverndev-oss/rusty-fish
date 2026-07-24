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
}

/// TSV header row for the v1 schema, including the leading `pos_id` column that
/// the dataset generator prepends. Kept adjacent to [`MoveDecision::to_tsv_row`]
/// so the column order stays single-sourced.
pub const TELEMETRY_TSV_HEADER: &str = "pos_id\tdepth\tply\tmove_index\tis_quiet\tis_priority\tpv_node\tgives_check\tstatic_eval\textension\treduction\tlmp_pruned\traised_alpha\tcaused_cutoff\tneeded_lmr_research\tneeded_pvs_research\tsubtree_nodes";

impl MoveDecision {
    /// Serializes this record as one TSV row, prefixed with `pos_id`. The column
    /// order matches [`TELEMETRY_TSV_HEADER`]. Booleans render as `0`/`1`.
    pub fn to_tsv_row(&self, pos_id: u64) -> String {
        let b = |flag: bool| u8::from(flag);
        format!(
            "{pos_id}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
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
    use super::{MoveDecision, TelemetryCollector, TELEMETRY_TSV_HEADER};

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
        };
        let row = record.to_tsv_row(9);
        let header_cols = TELEMETRY_TSV_HEADER.split('\t').count();
        let row_cols = row.split('\t').count();
        assert_eq!(header_cols, row_cols, "row arity must match the header");
        assert_eq!(
            row,
            "9\t7\t3\t5\t1\t0\t1\t0\t-42\t1\t2\t0\t1\t1\t0\t1\t1234"
        );
    }
}
