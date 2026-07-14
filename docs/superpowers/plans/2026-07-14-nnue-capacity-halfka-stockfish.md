# NNUE Capacity, HalfKA, and Stockfish Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce reproducible Stockfish 18 labels, train and compare capacity and HalfKA NNUE candidates on Modal GPUs, and promote only statistically supported candidates to the existing full SPRT gate.

**Architecture:** Add a Rust dataset/labeling boundary that produces immutable manifest-backed TSV shards, using a bounded external UCI Stockfish process. Extend the Python Modal path to store and consume those artifacts. Refactor NNUE features behind a schema so RFNN v1 stays loadable while RFNN v2 can encode 64-bucket HalfKA and maintain correct incremental accumulators.

**Tech Stack:** Rust workspace (`engine-core`, `engine-search`, `engine-bench`), Python 3, PyTorch, Modal, GitHub Actions, pinned Stockfish 18.

## Global Constraints

- Use Modal as the primary path: CPU containers label and gate; an A10G GPU trains.
- Corpus: exactly 1,000,000 legal nonterminal positions: 400,000 random-walk, 400,000 opening-derived, 200,000 quiet-walk positions; deduplicate canonical FENs before splitting.
- Stockfish: pinned Stockfish 18 SHA-256, one process/thread, fixed hash, `go nodes N`, per-position timeout. Calibrate on 1,000 FENs at 25k/100k/400k nodes; use the lowest 25k/100k budget with <=20 cp 95th-percentile delta to the 400k reference and no timeouts, otherwise 400k.
- Split canonical FEN hash into 90% train, 5% validation, 5% test. Store counts and SHA-256 values in immutable manifests.
- Keep WDL target `sigmoid(cp / 400)`. Current v1 features remain a 768-input control at widths 128, 256, and 512.
- Capacity promotion: >=2% validation WDL-loss reduction versus v1-128, <=32 cp maximum float-to-quantized difference on sealed test, >=1% sealed-test WDL-loss reduction, and >=50% in a 384-game deterministic screen.
- HalfKA uses 64 king buckets; king moves and castling refresh affected accumulators; other move classes update incrementally.
- RFNN v1 must continue loading. RFNN v2 must carry schema, input dimension, and bucket count, and reject malformed/inconsistent headers.
- Full promotion gate is the existing 12-shard/2,304-game depth-4 campaign with the PR #38 100 ms gate move bound. Only `AcceptH1` permits a separate adoption proposal.

---

## File structure

- Create: `engine-bench/src/dataset.rs` — canonical FEN records, deterministic source generation, deduplication, split assignment, manifest serialization, and TSV shard validation.
- Create: `engine-bench/src/stockfish.rs` — bounded UCI evaluation protocol, score parsing, calibration, and label records.
- Modify: `engine-bench/src/lib.rs` — export `dataset`/`stockfish`; move reusable UCI process primitives out of the external-match-only implementation.
- Modify: `engine-bench/src/main.rs` — `dataset-build`, `stockfish-label`, `dataset-merge`, and manifest-driven `gen-data` commands.
- Modify: `engine-search/src/nnue.rs` — schema-owned feature extraction, RFNN v1/v2 parser/writer, and HalfKA accumulator behavior.
- Modify: `engine-search/src/lib.rs` — schema-aware NNUE make/unmake hooks and randomized refresh-equivalence tests.
- Modify: `engine-bench/src/train.rs` — schema-bearing samples and v1/v2 TSV export/import validation.
- Modify: `modal/app.py` — idempotent Modal corpus, label, training, screen, and full-gate stages.
- Modify: `modal/train_nnue.py` — manifest-backed v1/v2 input dimension and HalfKA training/export.
- Modify: `modal/README.md`, `docs/HANDOFF.md` — Modal invocation, artifacts, promotion policy, and completed experiment records.

## Task 1: Deterministic corpus and manifest primitives

**Files:**
- Create: `engine-bench/src/dataset.rs`
- Modify: `engine-bench/src/lib.rs:1-20`
- Modify: `engine-bench/src/main.rs:1-260`
- Test: `engine-bench/src/dataset.rs` (`#[cfg(test)]` module)

**Interfaces:**

```rust
pub const TRAIN_SPLIT: &str = "train";
pub const VALIDATION_SPLIT: &str = "validation";
pub const TEST_SPLIT: &str = "test";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PositionRecord { pub fen: String, pub source: String }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetManifest {
    pub run_id: String,
    pub source_counts: BTreeMap<String, usize>,
    pub split_counts: BTreeMap<String, usize>,
    pub shard_sha256: Vec<String>,
    pub dataset_sha256: String,
    pub stockfish_config_sha256: Option<String>,
}

pub fn canonical_fen(fen: &str) -> Result<String, String>;
pub fn split_for_fen(fen: &str) -> &'static str;
pub fn deduplicate_and_split(records: Vec<PositionRecord>) -> Result<BTreeMap<String, Vec<PositionRecord>>, String>;
pub fn write_manifest(path: &Path, manifest: &DatasetManifest) -> Result<(), String>;
pub fn read_manifest(path: &Path) -> Result<DatasetManifest, String>;
```

- [ ] **Step 1: Write failing determinism tests**

```rust
#[test]
fn duplicate_fens_collapse_and_keep_a_stable_split() {
    let records = vec![record(STARTPOS), record(STARTPOS), record(KIWIPETE)];
    let splits = deduplicate_and_split(records).unwrap();
    assert_eq!(splits.values().map(Vec::len).sum::<usize>(), 2);
    assert_eq!(split_for_fen(STARTPOS), split_for_fen(STARTPOS));
}

#[test]
fn manifest_round_trip_preserves_hashes_and_counts() {
    let manifest = sample_manifest();
    write_manifest(&path, &manifest).unwrap();
    assert_eq!(read_manifest(&path).unwrap(), manifest);
}
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run: `cargo test -p engine-bench dataset::tests -- --nocapture`

Expected: compile failure because `dataset` and its interfaces do not exist.

- [ ] **Step 3: Implement canonical records, hash split, and manifest I/O**

```rust
pub fn split_for_fen(fen: &str) -> &'static str {
    match stable_u64(fen.as_bytes()) % 100 {
        0..=89 => TRAIN_SPLIT,
        90..=94 => VALIDATION_SPLIT,
        _ => TEST_SPLIT,
    }
}

pub fn canonical_fen(fen: &str) -> Result<String, String> {
    let board = Board::from_fen(fen)?;
    if board.generate_legal_move_list().is_empty() { return Err("terminal position".into()); }
    Ok(board.to_fen())
}
```

Use a deterministic in-repository hash implementation (not `DefaultHasher`), sort canonical FENs bytewise before writing, and encode the manifest as stable line-oriented TSV so no new serialization dependency is required.

- [ ] **Step 4: Add the `dataset-build` CLI smoke command**

```text
engine-bench dataset-build <run_id> <out_dir> <random_count> <opening_count> <quiet_count> <seed>
```

Reject counts other than `400000 400000 200000` unless `--smoke` is supplied; `--smoke` accepts counts totaling at most 1,000 and writes the same manifest layout.

- [ ] **Step 5: Run verification**

Run: `cargo test -p engine-bench dataset::tests && cargo run -p engine-bench -- dataset-build smoke artifacts/smoke 400 400 200 1 --smoke`

Expected: all dataset tests pass and `artifacts/smoke/manifest.tsv` lists three splits and SHA-256 values.

- [ ] **Step 6: Commit**

```bash
git add engine-bench/src/dataset.rs engine-bench/src/lib.rs engine-bench/src/main.rs
git commit -m "feat: add deterministic NNUE dataset manifests"
```

## Task 2: Bounded Stockfish labeler and calibration

**Files:**
- Create: `engine-bench/src/stockfish.rs`
- Modify: `engine-bench/src/lib.rs:1-20,633-724`
- Modify: `engine-bench/src/main.rs:145-260`
- Test: `engine-bench/src/stockfish.rs` (`#[cfg(test)]` module)

**Interfaces:**

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StockfishConfig {
    pub binary: PathBuf,
    pub binary_sha256: String,
    pub hash_mb: u32,
    pub node_budget: u64,
    pub response_timeout: Duration,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StockfishLabel { pub fen: String, pub score_cp: i32, pub nodes: u64 }
pub fn parse_info_score(line: &str) -> Option<i32>;
pub fn label_positions(config: &StockfishConfig, fens: &[String]) -> Result<Vec<StockfishLabel>, String>;
pub fn calibrate_node_budget(config: &StockfishConfig, fens: &[String]) -> Result<u64, String>;
```

- [ ] **Step 1: Write parser and calibration failure tests**

```rust
#[test]
fn parses_cp_and_converts_mate_to_documented_clamp() {
    assert_eq!(parse_info_score("info depth 12 score cp -37 nodes 25000"), Some(-37));
    assert_eq!(parse_info_score("info depth 18 score mate 3 nodes 25000"), Some(MATE_LABEL_CP));
}

#[test]
fn calibration_chooses_lowest_budget_within_twenty_cp_p95() {
    assert_eq!(choose_budget(&[(25_000, 18), (100_000, 9)]), 25_000);
    assert_eq!(choose_budget(&[(25_000, 24), (100_000, 14)]), 100_000);
}
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `cargo test -p engine-bench stockfish::tests`

Expected: compile failure because `stockfish` does not exist.

- [ ] **Step 3: Extract reusable UCI transport and implement labels**

```rust
fn evaluate_one(process: &mut UciProcess, fen: &str, nodes: u64) -> Result<StockfishLabel, String> {
    process.send("ucinewgame")?;
    process.wait_for("readyok")?;
    process.send(&format!("position fen {fen}"))?;
    process.send(&format!("go nodes {nodes}"))?;
    process.wait_for_score_and_bestmove(fen)
}
```

Start exactly one process per shard, send `uci`, `setoption name Threads value 1`, `setoption name Hash value <hash_mb>`, and `isready`; verify the binary digest before start. Return an error on timeout, malformed output, `bestmove 0000`, nonzero child exit, or a reported node count below the requested budget. Do not fall back to internal evaluation.

- [ ] **Step 4: Add a portable fake-UCI integration test**

Create a test-only `UciTransport` trait and fake implementation returning scripted `info`/`bestmove` lines. Cover command order, timeout, malformed score, mate conversion, and child-error propagation without requiring Stockfish in unit tests.

- [ ] **Step 5: Add CLI commands**

```text
engine-bench stockfish-calibrate <manifest> <stockfish> <sha256> <out_config>
engine-bench stockfish-label <manifest> <split> <stockfish_config> <out_tsv>
```

`stockfish-label` writes `score_cp<TAB>own_features<TAB>opp_features<TAB>canonical_fen<TAB>reported_nodes`; it rejects a split whose FEN hash does not match the manifest.

- [ ] **Step 6: Run verification**

Run: `cargo test -p engine-bench stockfish::tests && cargo test -p engine-bench`

Expected: label parser, fake transport, and existing external-match tests pass.

- [ ] **Step 7: Commit**

```bash
git add engine-bench/src/stockfish.rs engine-bench/src/lib.rs engine-bench/src/main.rs
git commit -m "feat: add bounded Stockfish NNUE labels"
```

## Task 3: Manifest-backed Modal corpus and v1 capacity ladder

**Files:**
- Modify: `modal/app.py:1-151`
- Modify: `modal/train_nnue.py:1-185`
- Modify: `modal/README.md:1-80`
- Test: `modal/train_nnue.py` (Python `unittest` block)

**Interfaces:**

```python
def build_corpus(run_id: str, smoke: bool) -> tuple[str, str]: ...
def label_manifest(run_id: str, manifest_text: str, stockfish_config_text: str) -> dict[str, str]: ...
def train_net(data_text: str, schema: str, input_dimension: int, hidden: int,
              epochs: int, run_id: str) -> bytes: ...
def run_screen(net_bytes: bytes, openings_per_shard: int = 16) -> tuple[int, int, int]: ...
```

- [ ] **Step 1: Write failing Python tests for schema-aware data loading**

```python
def test_load_samples_rejects_mixed_schema_or_feature_dimension(tmp_path):
    path = tmp_path / "mixed.tsv"
    path.write_text("v1\t1\t0\t\nHalfKA\t1\t0\t\n")
    with pytest.raises(ValueError, match="schema"):
        _load_samples(path, expected_schema="v1", input_dimension=768)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest modal -q`

Expected: failure because loader has no manifest/schema checks.

- [ ] **Step 3: Make Modal stages artifact-addressed and idempotent**

```python
@app.function(image=rust_image, timeout=60 * 60)
def build_corpus(run_id: str, smoke: bool) -> tuple[str, str]:
    args = [BIN, "dataset-build", run_id, "/tmp/corpus", "400000", "400000", "200000", "1"]
    if smoke: args.append("--smoke")
    subprocess.run(args, check=True)
    return read_text("/tmp/corpus/manifest.tsv"), read_text("/tmp/corpus/positions.tsv")
```

Persist each artifact at `runs/<run_id>/<stage>-<sha256>` and refuse to overwrite a path with a different input hash. Pass the explicit manifest/config into every remote function; do not retain hidden module-level run state.

- [ ] **Step 4: Implement GPU v1 capacity candidates**

Use `EmbeddingBag(input_dimension, hidden, mode="sum")`, retain the existing WDL loss, and emit one `report.json` per width with train/validation loss, test loss, model checksum, input dimension, schema, epochs, learning rate, and quantization maximum error. Run widths 128, 256, and 512 from the same manifest and seed sequence.

- [ ] **Step 5: Add the screen and promotion evaluator**

```python
def promotes(report: dict, control: dict, screen: tuple[int, int, int]) -> bool:
    return (
        report["validation_wdl_loss"] <= control["validation_wdl_loss"] * 0.98
        and report["test_wdl_loss"] <= control["test_wdl_loss"] * 0.99
        and report["quantization_max_error_cp"] <= 32
        and screen[0] + 0.5 * screen[1] >= 192
    )
```

Make `run_screen` generate 12 deterministic opening shards with 16 openings each and call bounded `gate-file` with `100` milliseconds explicitly.

- [ ] **Step 6: Run verification**

Run: `python -m pytest modal -q && modal run modal/app.py --run-id smoke-v1 --smoke --schema v1 --widths 128,256,512`

Expected: Python tests pass; Modal prints manifest SHA, three candidate reports, and 384-game screen W/D/L only for candidates satisfying offline rules.

- [ ] **Step 7: Commit**

```bash
git add modal/app.py modal/train_nnue.py modal/README.md
git commit -m "feat: run manifest-backed Modal NNUE capacity experiments"
```

## Task 4: RFNN v2 schema and HalfKA feature extraction

**Files:**
- Modify: `engine-search/src/nnue.rs:1-450`
- Modify: `engine-search/src/lib.rs:430-520,2580-2640`
- Modify: `engine-bench/src/train.rs:1-430`
- Test: `engine-search/src/nnue.rs` and `engine-search/src/lib.rs` test modules

**Interfaces:**

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureSchema { RelativePieceSquareV1, HalfKaV2 { buckets: u8 } }

impl FeatureSchema {
    pub fn input_dimension(self) -> usize;
    pub fn active_features(self, board: &Board, perspective: Color) -> Vec<usize>;
    pub fn requires_refresh_after(self, mv: Move, moved: Piece) -> bool;
}

pub struct Nnue { schema: FeatureSchema, hidden: usize, /* parameters */ }
pub fn halfka_feature_index(perspective: Color, king_square: Square, piece: Piece, square: Square) -> Option<usize>;
```

- [ ] **Step 1: Write failing v2 and HalfKA tests**

```rust
#[test]
fn v1_round_trips_and_v2_rejects_wrong_dimension() { /* build both schemas; mutate header */ }

#[test]
fn halfka_king_move_refresh_matches_full_refresh() {
    let net = Nnue::from_seed_with_schema(7, 32, FeatureSchema::HalfKaV2 { buckets: 64 });
    assert_incremental_matches_refresh(&net, CASTLING_FEN, "e1g1");
}

#[test]
fn halfka_handles_capture_promotion_and_en_passant_incrementally() { /* three legal FEN/move pairs */ }
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `cargo test -p engine-search nnue -- --nocapture`

Expected: compile failure because `FeatureSchema` and RFNN v2 do not exist.

- [ ] **Step 3: Implement schema-owned dimensions and v1/v2 parsing**

```rust
match version {
    1 => FeatureSchema::RelativePieceSquareV1,
    2 => FeatureSchema::HalfKaV2 { buckets: cursor.read_u8()? },
    other => return Err(format!("unsupported RFNN version {other}")),
}
```

For v2, serialize schema tag, bucket count, and explicit `u32` input dimension before weights. Require `input_dimension == schema.input_dimension()`. Keep the exact existing v1 byte layout; do not add fields to v1.

- [ ] **Step 4: Implement HalfKA accumulator updates**

```rust
if net.schema.requires_refresh_after(mv, moved_piece) {
    *accumulator = Accumulator::refresh(net, board_after_move);
} else {
    accumulator.remove_feature(net, perspective, moved_piece, from);
    accumulator.add_feature(net, perspective, placed_piece, to);
}
```

Implement both perspectives, castling rook movement, promotion replacement, captured-piece removal, and en-passant capture square. Retain the existing per-node refresh equality assertion.

- [ ] **Step 5: Propagate schema through training samples**

```rust
pub struct TrainingSample {
    pub schema: FeatureSchema,
    pub own: Vec<usize>, pub opp: Vec<usize>, pub target: f32,
}
```

Reject mixed-schema batches and write schema plus input dimension in external TSV headers. Update v1 tests to assert their byte output is unchanged.

- [ ] **Step 6: Run verification**

Run: `cargo test -p engine-search && cargo test -p engine-bench train::tests`

Expected: v1 compatibility, v2 rejection cases, every special move class, randomized make/unmake equality, and training sample tests pass.

- [ ] **Step 7: Commit**

```bash
git add engine-search/src/nnue.rs engine-search/src/lib.rs engine-bench/src/train.rs
git commit -m "feat: add RFNN v2 HalfKA feature schema"
```

## Task 5: HalfKA GPU training and full promotion gate

**Files:**
- Modify: `modal/train_nnue.py:1-185`
- Modify: `modal/app.py:1-151`
- Modify: `modal/README.md:1-80`
- Modify: `.github/workflows/nnue-campaign.yml:1-130`
- Modify: `docs/HANDOFF.md:135-195`
- Test: `modal/train_nnue.py` test module; `engine-bench` smoke commands

**Interfaces:**

```python
def train(data_path: str, schema: str, input_dimension: int, hidden: int,
          epochs: int, batch_size: int, lr: float, device: str): ...
def quantize_and_write(model, schema: str, input_dimension: int, buckets: int,
                       hidden: int, out_path: str) -> None: ...
def selected_halfka_widths(capacity_selection_report: dict[str, int]) -> list[int]: ...
def run_full_gate(net_bytes: bytes, run_id: str) -> dict[str, int | str]: ...
```

- [ ] **Step 1: Write failing v2 Python export tests**

```python
def test_halfka_export_header_matches_rust_v2_contract(tmp_path):
    model = tiny_model(schema="halfka-v2", input_dimension=64 * 2 * 5 * 64, hidden=4)
    quantize_and_write(model, "halfka-v2", 40960, 64, 4, tmp_path / "net.rfnn")
    assert (tmp_path / "net.rfnn").read_bytes()[4:8] == struct.pack("<I", 2)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest modal -q`

Expected: failure because exporter writes RFNN v1 only.

- [ ] **Step 3: Implement schema-aware trainer/exporter and Rust parity check**

Pass `schema`, `input_dimension`, and `buckets` explicitly to the PyTorch model and header writer. Add a Modal Rust-container stage that invokes `engine-bench gate-file` on one opening after loading the exported file; failure blocks the candidate before screen or full gate.

- [ ] **Step 4: Run HalfKA experiments**

Implement `selected_halfka_widths` as `{128: [128, 256], 256: [256, 512],
512: [512]}`. Train the returned widths in order and stop after the first one
that fails promotion. Compare each against the same v1-128 control manifest and
record an immutable `report.json`.

- [ ] **Step 5: Dispatch the full gate only for a promoted candidate**

```python
if promotes(candidate_report, control_report, screen):
    verdict = run_full_gate(net_bytes, run_id)
    assert verdict["games"] == 2304
```

The GitHub workflow receives the net artifact and an explicit manifest checksum; it must print `23W ...`-style aggregate, Elo, LLR, and decision. Keep 12 shards, 96 openings per shard, depth 4, and `gate-file ... 100`.

- [ ] **Step 6: Record outcome and update docs**

Append the run identifier, corpus/config hashes, architecture, widths, quantization report, screen, W/D/L, Elo, SPRT decision, and campaign URL to `docs/HANDOFF.md`. Record `AcceptH0` as a failed branch; for `AcceptH1`, create a new adoption design rather than embedding the net in this task.

- [ ] **Step 7: Run verification**

Run: `python -m pytest modal -q && cargo test --workspace && modal run modal/app.py --run-id smoke-halfka --smoke --schema halfka-v2 --capacity-selection runs/smoke-v1/capacity-selection.json`

Expected: export contract passes, Rust loads the v2 smoke net, bounded smoke gate completes, and only eligible candidates request the full gate.

- [ ] **Step 8: Commit**

```bash
git add modal/train_nnue.py modal/app.py modal/README.md .github/workflows/nnue-campaign.yml docs/HANDOFF.md
git commit -m "feat: gate HalfKA NNUE candidates by reproducible SPRT"
```

## Plan self-review

- Spec coverage: Tasks 1-2 deliver the exact corpus, immutable manifests, Stockfish pinning, fixed-node calibration, timeout behavior, and split guarantees. Task 3 implements Modal GPU capacity runs and 384-game screening. Task 4 implements RFNN v2, 64-bucket HalfKA, v1 compatibility, and accumulator proofs. Task 5 implements HalfKA GPU training, full 2,304-game promotion gate, and evidence recording.
- Placeholder scan: the plan contains no undecided architecture, label, split, threshold, promotion value, or execution command. The deterministic `selected_halfka_widths` function maps the capacity report to the HalfKA width sequence.
- Type consistency: `FeatureSchema`, `StockfishConfig`, `DatasetManifest`, schema-bearing TSV, `train_net`, and `promotes` are defined before later tasks consume them.
