use std::fs;
use std::process::Command;
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
