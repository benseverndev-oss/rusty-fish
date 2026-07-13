#[cfg(test)]
mod tests {
    #[test]
    fn aggregates_transpositions_and_side_relative_results() {
        let report = super::build_book(
            "[Event \"fixture\"]\n\n1. e4 e5 1-0\n",
            super::BookFilter {
                min_rating: 2200,
                max_plies: 16,
            },
        )
        .unwrap();
        assert!(report.book.contains("e2e4:"));
    }
}
