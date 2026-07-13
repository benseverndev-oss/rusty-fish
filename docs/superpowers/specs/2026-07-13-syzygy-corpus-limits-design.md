# Syzygy Corpus and Probe Limits Design

Complete tablebase deployment with a checksummed real KQvK corpus, exact
GitHub Actions validation, and UCI-configurable probe limits. `SyzygyProbeDepth`
sets the minimum interior WDL depth (default 1); `SyzygyProbeLimit` caps total
pieces (default 7). Root DTZ respects the piece limit and absent/failed probes
fall back to search.

The corpus workflow verifies Lichess KQvK WDL SHA-256
`517667dff787162dbb1ed9d5d6484d30ee854e686ee0675c08d99ecf045d2d50` and
DTZ SHA-256 `71ea9444fa5bd42897d781a0c356975ea6f23e0f65a4254e470897031c161c8c`.
