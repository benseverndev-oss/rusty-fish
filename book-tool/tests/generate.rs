use std::fs;
use std::io::Write;
use std::process::Command;
use std::process::Stdio;
use std::time::{SystemTime, UNIX_EPOCH};

const FIXTURE: &str = include_str!("../../assets/opening-book/lichess-cc0-fixture.pgn");
const EXPECTED_BOOK: &str = include_str!("../../assets/opening-book/fixture-book-v2.txt");
const EXPECTED_METRICS: &str = include_str!("../../assets/opening-book/fixture-metrics.tsv");

fn test_directory() -> std::path::PathBuf {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("rusty-fish-book-tool-{suffix}"))
}

#[test]
fn generate_writes_deterministic_book_and_metrics() {
    let root = test_directory();
    fs::create_dir_all(&root).expect("create temporary directory");
    let input = root.join("fixture.pgn");
    let book = root.join("book.txt");
    let metrics = root.join("metrics.tsv");
    fs::write(&input, FIXTURE).expect("write fixture");

    let status = Command::new(env!("CARGO_BIN_EXE_book-tool"))
        .args([
            "generate",
            input.to_str().unwrap(),
            book.to_str().unwrap(),
            metrics.to_str().unwrap(),
        ])
        .status()
        .expect("run book generator");

    assert!(status.success());
    assert_eq!(
        fs::read_to_string(&book).expect("generated book"),
        EXPECTED_BOOK
    );
    assert_eq!(
        fs::read_to_string(&metrics).expect("generated metrics"),
        EXPECTED_METRICS
    );

    fs::remove_dir_all(root).expect("remove temporary directory");
}

const HITRATE_BOOK: &str = concat!(
    "rusty-fish-book v2\n",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -\te2e4:9\n",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3\te7e5:5\n",
);

// Line one checks three positions (start, after e4, after e4 e5); the book has
// the first two, so 2/3 with an in-book depth of 2 and no full coverage. Line
// two checks one position (start), a hit, fully covered. Totals: 2 lines, 4
// checked, 3 in book, hit_rate 3/4, mean_book_depth (2+1)/2, 1 fully covered.
const HITRATE_SUITE: &str = concat!(
    "[Event \"line one\"]\n\n1. e4 e5 2. Nf3 *\n\n",
    "[Event \"line two\"]\n\n1. e4 *\n",
);

const HITRATE_EXPECTED: &str = concat!(
    "metric\tvalue\n",
    "lines\t2\n",
    "plies_checked\t4\n",
    "plies_in_book\t3\n",
    "hit_rate\t0.750000\n",
    "mean_book_depth\t1.500000\n",
    "fully_covered_lines\t1\n",
);

#[test]
fn hitrate_reports_coverage_over_a_suite() {
    let root = test_directory();
    fs::create_dir_all(&root).expect("create temporary directory");
    let book = root.join("book.txt");
    let suite = root.join("suite.pgn");
    fs::write(&book, HITRATE_BOOK).expect("write book");
    fs::write(&suite, HITRATE_SUITE).expect("write suite");

    let output = Command::new(env!("CARGO_BIN_EXE_book-tool"))
        .args(["hitrate", book.to_str().unwrap(), suite.to_str().unwrap()])
        .output()
        .expect("run book generator");

    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(String::from_utf8(output.stdout).expect("utf8 stdout"), HITRATE_EXPECTED);

    fs::remove_dir_all(root).expect("remove temporary directory");
}

// A minimal, valid v2 book: correct header plus the start position mapped to a
// single legal move. Reused for the illegal-suite-move case so that the only
// thing wrong is the suite, not the book.
const VALID_BOOK: &str = concat!(
    "rusty-fish-book v2\n",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -\te2e4:9\n",
);

// Header is not `rusty-fish-book v2`, so signature loading must reject it.
const MALFORMED_BOOK: &str = concat!(
    "rusty-fish-book v1\n",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -\te2e4:9\n",
);

// `e5` is illegal as White's first move from the start position, so the suite
// replay must record an error and the command must fail.
const ILLEGAL_SUITE: &str = "[Event \"illegal\"]\n\n1. e5 *\n";

#[test]
fn hitrate_fails_loudly_on_bad_input() {
    let root = test_directory();
    fs::create_dir_all(&root).expect("create temporary directory");

    // Case 1: a malformed book header paired with an otherwise valid suite.
    let malformed_book = root.join("malformed-book.txt");
    let valid_suite = root.join("valid-suite.pgn");
    fs::write(&malformed_book, MALFORMED_BOOK).expect("write malformed book");
    fs::write(&valid_suite, HITRATE_SUITE).expect("write valid suite");

    let output = Command::new(env!("CARGO_BIN_EXE_book-tool"))
        .args([
            "hitrate",
            malformed_book.to_str().unwrap(),
            valid_suite.to_str().unwrap(),
        ])
        .output()
        .expect("run book generator");
    assert!(
        !output.status.success(),
        "a malformed book must fail loudly; stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );

    // Case 2: a valid book paired with a suite containing an illegal SAN move.
    let valid_book = root.join("valid-book.txt");
    let illegal_suite = root.join("illegal-suite.pgn");
    fs::write(&valid_book, VALID_BOOK).expect("write valid book");
    fs::write(&illegal_suite, ILLEGAL_SUITE).expect("write illegal suite");

    let output = Command::new(env!("CARGO_BIN_EXE_book-tool"))
        .args([
            "hitrate",
            valid_book.to_str().unwrap(),
            illegal_suite.to_str().unwrap(),
        ])
        .output()
        .expect("run book generator");
    assert!(
        !output.status.success(),
        "an illegal suite move must fail loudly; stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );

    fs::remove_dir_all(root).expect("remove temporary directory");
}

const SINGLE_WIN: &str = "[Event \"Rated fixture\"]\n[WhiteElo \"2300\"]\n[BlackElo \"2300\"]\n[Result \"1-0\"]\n\n1. e4 e5 1-0\n";

#[test]
fn a_single_decisive_game_does_not_satisfy_the_minimum_three_observations() {
    let root = test_directory();
    fs::create_dir_all(&root).expect("create temporary directory");
    let input = root.join("single.pgn");
    let book = root.join("book.txt");
    let metrics = root.join("metrics.tsv");
    fs::write(&input, SINGLE_WIN).expect("write fixture");

    let status = Command::new(env!("CARGO_BIN_EXE_book-tool"))
        .args([
            "generate",
            input.to_str().unwrap(),
            book.to_str().unwrap(),
            metrics.to_str().unwrap(),
        ])
        .status()
        .expect("run book generator");

    assert!(status.success());
    // White's e2e4 scores three points in a won game but has one observation,
    // so no move survives the minimum-three filter and the book is header-only.
    assert_eq!(
        fs::read_to_string(&book).expect("generated book"),
        "rusty-fish-book v2\n"
    );

    fs::remove_dir_all(root).expect("remove temporary directory");
}

const THREE_POSITION_CORPUS: &str = concat!(
    "[Event \"Rated fixture\"]\n[WhiteElo \"2300\"]\n[BlackElo \"2300\"]\n[Result \"1-0\"]\n\n1. e4 e5 1-0\n\n",
    "[Event \"Rated fixture\"]\n[WhiteElo \"2300\"]\n[BlackElo \"2300\"]\n[Result \"1-0\"]\n\n1. e4 e5 1-0\n\n",
    "[Event \"Rated fixture\"]\n[WhiteElo \"2300\"]\n[BlackElo \"2300\"]\n[Result \"1-0\"]\n\n1. e4 e5 1-0\n\n",
    "[Event \"Rated fixture\"]\n[WhiteElo \"2300\"]\n[BlackElo \"2300\"]\n[Result \"0-1\"]\n\n1. d4 d5 0-1\n\n",
    "[Event \"Rated fixture\"]\n[WhiteElo \"2300\"]\n[BlackElo \"2300\"]\n[Result \"0-1\"]\n\n1. d4 d5 0-1\n\n",
    "[Event \"Rated fixture\"]\n[WhiteElo \"2300\"]\n[BlackElo \"2300\"]\n[Result \"0-1\"]\n\n1. d4 d5 0-1\n",
);

#[test]
fn max_positions_keeps_the_most_observed_positions() {
    let root = test_directory();
    fs::create_dir_all(&root).expect("create temporary directory");
    let input = root.join("corpus.pgn");
    let book = root.join("book.txt");
    let metrics = root.join("metrics.tsv");
    fs::write(&input, THREE_POSITION_CORPUS).expect("write fixture");

    let status = Command::new(env!("CARGO_BIN_EXE_book-tool"))
        .args([
            "generate",
            input.to_str().unwrap(),
            book.to_str().unwrap(),
            metrics.to_str().unwrap(),
            "--max-positions",
            "1",
        ])
        .status()
        .expect("run book generator");

    assert!(status.success());
    // The start position retains six observations across its two alternatives;
    // the positions after e4 and after d4 retain three each, so the bound of
    // one keeps only the start position.
    assert_eq!(
        fs::read_to_string(&book).expect("generated book"),
        "rusty-fish-book v2\nrnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -\te2e4:9 d2d4:3\n"
    );
    // entries and alternatives are counted after the cap; positions is the
    // pre-filter signature count and is unaffected by it.
    assert_eq!(
        fs::read_to_string(&metrics).expect("generated metrics"),
        "metric\tvalue\nsource_games\t6\naccepted_games\t6\npositions\t3\nentries\t1\nalternatives\t2\n"
    );

    fs::remove_dir_all(root).expect("remove temporary directory");
}

#[test]
fn a_dash_input_path_reads_the_pgn_from_stdin() {
    let root = test_directory();
    fs::create_dir_all(&root).expect("create temporary directory");
    let book = root.join("book.txt");
    let metrics = root.join("metrics.tsv");

    let mut child = Command::new(env!("CARGO_BIN_EXE_book-tool"))
        .args([
            "generate",
            "-",
            book.to_str().unwrap(),
            metrics.to_str().unwrap(),
        ])
        .stdin(Stdio::piped())
        .spawn()
        .expect("start book generator");
    child
        .stdin
        .take()
        .expect("generator stdin")
        .write_all(FIXTURE.as_bytes())
        .expect("stream fixture to stdin");
    let status = child.wait().expect("run book generator");

    assert!(status.success());
    assert_eq!(
        fs::read_to_string(&book).expect("generated book"),
        EXPECTED_BOOK
    );

    fs::remove_dir_all(root).expect("remove temporary directory");
}
