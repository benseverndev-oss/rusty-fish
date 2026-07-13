# rusty-fish

`rusty-fish` is a Rust chess engine and desktop GUI project aimed at a long-term grandmaster-strength roadmap.

## Workspace

- `engine-core`: board state, FEN, legal move generation, game state, and perft.
- `engine-search`: iterative deepening alpha-beta search with quiescence and basic evaluation.
- `engine-uci`: UCI-compatible command-line engine binary.
- `app-desktop`: `egui` desktop app for local play and analysis.

## Notes

- GitHub is initialized on `main`.
- Stripe is intentionally disabled for this project.
- Vercel CLI is not installed in this environment. If you later use Vercel for docs or a web surface, install it with `npm i -g vercel`.
