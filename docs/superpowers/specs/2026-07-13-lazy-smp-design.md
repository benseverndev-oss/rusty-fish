# Lazy SMP Multithreading Design

## Goal

Let the engine use multiple search threads that cooperate through a shared
transposition table (Lazy SMP), so wall-clock strength scales with available
cores while the single-threaded search path stays byte-for-byte unchanged.

## Scope

This slice adds parallel search to `engine-search` and exposes it through UCI:

- A `Threads` search option (default `1`) on `SearchOptions`.
- A `SharedTranspositionTable` that many threads may probe and store into
  concurrently, wrapping the existing `TranspositionTable` behind per-shard
  locks.
- Helper search threads that run their own iterative deepening over a shared
  board, sharing only the transposition table and the stop signal.
- A `Threads` UCI option advertised in the `uci` header and parsed by
  `setoption`.

When `Threads <= 1` the engine takes exactly the current code path: no threads
are spawned and no shared-table contention is possible.

## Alternatives considered

1. **Root splitting / YBWC.** Explicitly distributes sibling moves across
   threads. Higher potential efficiency but far more invasive to the existing
   PVS loop and much harder to keep correct. Rejected for this slice.
2. **Lockless (XOR-key) shared table.** The eventual optimum: pack each entry
   into two atomics and validate with the Stockfish XOR trick. Best scaling but
   a full rewrite of the tested `TranspositionTable`. Deferred; noted as
   follow-up.
3. **Sharded, lock-based shared table + independent helper searches
   (chosen).** Reuses the existing, tested `TranspositionTable` unchanged as
   the per-shard store, routes probes to a shard by the key's high bits, and
   lets helper threads deepen independently. Bounded, independently testable,
   and the standard first implementation of Lazy SMP.

## Architecture

- `SharedTranspositionTable` owns `Vec<Mutex<TranspositionTable>>`. The shard
  is selected from the key's high bits while each inner table indexes clusters
  from the low bits, so the two selections stay decorrelated. `get` returns an
  owned (`Copy`) entry so no borrow is held across the lock; `store`,
  `begin_search`, and `resize` lock only the shard(s) they touch.
- `Searcher` holds `Arc<SharedTranspositionTable>`. Helper threads clone the
  `Arc` (shared table) and the `Arc<AtomicBool>` stop signal; each builds its
  own `Searcher` with private killers/history/counter-move tables, so only the
  transposition table and stop flag cross the thread boundary.
- The primary thread runs the existing iterative-deepening loop (it alone bumps
  the shared table generation, probes Syzygy, consults the opening book, and
  emits `info` callbacks). Helper threads run a stripped loop that only deepens
  and fills the shared table; odd-indexed helpers begin one ply deeper to
  desynchronise the fleet.
- When the primary loop ends (depth reached, mate found, time out, or external
  stop), it sets the shared stop flag and joins all helpers before returning.
  The reported result is always the primary thread's.

## Safety rules

- `Threads <= 1` spawns no threads and preserves the exact current behaviour.
- Helper threads never probe the opening book or Syzygy tablebases and never
  emit `info`; the primary thread remains the single source of reported output.
- Only the primary thread calls `begin_search`, so the shared generation is
  bumped once per search.
- No thread holds more than one shard lock at a time and no lock is held across
  recursion, so the search cannot deadlock.
- The reported best move/score/PV come solely from the primary thread.

## Verification

Remote `Rusty Fish Tests / workspace` must pass, including the new shared-table
concurrency test, the multi-threaded search-quality test, and the UCI option
tests. The tactical suite and fixed-opponent gauntlet must not regress from
their committed baselines. A `Threads`-varied external Stockfish SPRT run is
the acceptance evidence that parallel search preserves or improves strength.
