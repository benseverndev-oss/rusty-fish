#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        deduplicate_and_split, read_manifest, split_for_fen, write_manifest, DatasetManifest,
        PositionRecord,
    };

    const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    fn record(fen: &str) -> PositionRecord {
        PositionRecord {
            fen: fen.to_string(),
            source: "test".to_string(),
        }
    }

    fn sample_manifest() -> DatasetManifest {
        DatasetManifest {
            run_id: "sample".to_string(),
            source_counts: BTreeMap::from([("random".to_string(), 2)]),
            split_counts: BTreeMap::from([("train".to_string(), 2)]),
            shard_sha256: vec!["a".repeat(64)],
            dataset_sha256: "b".repeat(64),
            stockfish_config_sha256: Some("c".repeat(64)),
        }
    }

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
        let path = std::env::temp_dir().join(format!(
            "rusty-fish-manifest-{}-{}.tsv",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        write_manifest(&path, &manifest).unwrap();
        assert_eq!(read_manifest(&path).unwrap(), manifest);
        std::fs::remove_file(path).unwrap();
    }
}
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use engine_core::Board;

pub const TRAIN_SPLIT: &str = "train";
pub const VALIDATION_SPLIT: &str = "validation";
pub const TEST_SPLIT: &str = "test";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PositionRecord {
    pub fen: String,
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetManifest {
    pub run_id: String,
    pub source_counts: BTreeMap<String, usize>,
    pub split_counts: BTreeMap<String, usize>,
    pub shard_sha256: Vec<String>,
    pub dataset_sha256: String,
    pub stockfish_config_sha256: Option<String>,
}

pub fn canonical_fen(fen: &str) -> Result<String, String> {
    let mut board = Board::from_fen(fen)?;
    if board.generate_legal_move_list().is_empty() {
        return Err("terminal position".into());
    }
    Ok(board.to_fen())
}

pub fn split_for_fen(fen: &str) -> &'static str {
    match stable_u64(fen.as_bytes()) % 100 {
        0..=89 => TRAIN_SPLIT,
        90..=94 => VALIDATION_SPLIT,
        _ => TEST_SPLIT,
    }
}

pub fn deduplicate_and_split(
    records: Vec<PositionRecord>,
) -> Result<BTreeMap<String, Vec<PositionRecord>>, String> {
    let mut unique = BTreeMap::<String, String>::new();
    for record in records {
        let fen = canonical_fen(&record.fen)?;
        unique
            .entry(fen)
            .and_modify(|source| {
                if record.source < *source {
                    *source = record.source.clone();
                }
            })
            .or_insert(record.source);
    }

    let mut splits = BTreeMap::from([
        (TRAIN_SPLIT.to_string(), Vec::new()),
        (VALIDATION_SPLIT.to_string(), Vec::new()),
        (TEST_SPLIT.to_string(), Vec::new()),
    ]);
    for (fen, source) in unique {
        splits
            .get_mut(split_for_fen(&fen))
            .expect("all split names are initialized")
            .push(PositionRecord { fen, source });
    }
    Ok(splits)
}

pub fn write_manifest(path: &Path, manifest: &DatasetManifest) -> Result<(), String> {
    validate_manifest_field("run_id", &manifest.run_id)?;
    let mut output = String::from("dataset_manifest\t1\n");
    push_pair(&mut output, "run_id", &manifest.run_id)?;
    for (source, count) in &manifest.source_counts {
        validate_manifest_field("source", source)?;
        output.push_str(&format!("source_count\t{source}\t{count}\n"));
    }
    for (split, count) in &manifest.split_counts {
        validate_manifest_field("split", split)?;
        output.push_str(&format!("split_count\t{split}\t{count}\n"));
    }
    for hash in &manifest.shard_sha256 {
        push_pair(&mut output, "shard_sha256", hash)?;
    }
    push_pair(&mut output, "dataset_sha256", &manifest.dataset_sha256)?;
    if let Some(hash) = &manifest.stockfish_config_sha256 {
        push_pair(&mut output, "stockfish_config_sha256", hash)?;
    }
    fs::write(path, output).map_err(|error| format!("failed to write {}: {error}", path.display()))
}

pub fn read_manifest(path: &Path) -> Result<DatasetManifest, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let mut lines = contents.lines();
    if lines.next() != Some("dataset_manifest\t1") {
        return Err("unsupported dataset manifest format".to_string());
    }
    let mut run_id = None;
    let mut source_counts = BTreeMap::new();
    let mut split_counts = BTreeMap::new();
    let mut shard_sha256 = Vec::new();
    let mut dataset_sha256 = None;
    let mut stockfish_config_sha256 = None;
    for line in lines {
        let fields: Vec<_> = line.split('\t').collect();
        match fields.as_slice() {
            ["run_id", value] => run_id = Some((*value).to_string()),
            ["source_count", source, count] => {
                source_counts.insert((*source).to_string(), parse_count(count)?);
            }
            ["split_count", split, count] => {
                split_counts.insert((*split).to_string(), parse_count(count)?);
            }
            ["shard_sha256", hash] => shard_sha256.push((*hash).to_string()),
            ["dataset_sha256", hash] => dataset_sha256 = Some((*hash).to_string()),
            ["stockfish_config_sha256", hash] => stockfish_config_sha256 = Some((*hash).to_string()),
            _ => return Err(format!("invalid dataset manifest line: {line}")),
        }
    }
    Ok(DatasetManifest {
        run_id: run_id.ok_or_else(|| "manifest is missing run_id".to_string())?,
        source_counts,
        split_counts,
        shard_sha256,
        dataset_sha256: dataset_sha256
            .ok_or_else(|| "manifest is missing dataset_sha256".to_string())?,
        stockfish_config_sha256,
    })
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hash = [
        0x6a09e667_u32, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
        0x1f83d9ab, 0x5be0cd19,
    ];
    let bit_length = (bytes.len() as u64).wrapping_mul(8);
    let mut padded = bytes.to_vec();
    padded.push(0x80);
    while (padded.len() + 8) % 64 != 0 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_length.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut words = [0_u32; 64];
        for (index, word) in words.iter_mut().take(16).enumerate() {
            *word = u32::from_be_bytes(chunk[index * 4..index * 4 + 4].try_into().unwrap());
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let mut state = hash;
        for index in 0..64 {
            let s1 = state[4].rotate_right(6) ^ state[4].rotate_right(11) ^ state[4].rotate_right(25);
            let choice = (state[4] & state[5]) ^ (!state[4] & state[6]);
            let temp1 = state[7]
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(SHA256_CONSTANTS[index])
                .wrapping_add(words[index]);
            let s0 = state[0].rotate_right(2) ^ state[0].rotate_right(13) ^ state[0].rotate_right(22);
            let majority = (state[0] & state[1]) ^ (state[0] & state[2]) ^ (state[1] & state[2]);
            let temp2 = s0.wrapping_add(majority);
            state = [
                temp1.wrapping_add(temp2), state[0], state[1], state[2], state[3].wrapping_add(temp1),
                state[4], state[5], state[6],
            ];
        }
        for (target, value) in hash.iter_mut().zip(state) {
            *target = target.wrapping_add(value);
        }
    }
    hash.iter().map(|word| format!("{word:08x}")).collect()
}

fn stable_u64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn push_pair(output: &mut String, key: &str, value: &str) -> Result<(), String> {
    validate_manifest_field(key, value)?;
    output.push_str(key);
    output.push('\t');
    output.push_str(value);
    output.push('\n');
    Ok(())
}

fn validate_manifest_field(name: &str, value: &str) -> Result<(), String> {
    if value.contains(['\t', '\n', '\r']) {
        return Err(format!("{name} cannot contain tabs or newlines"));
    }
    Ok(())
}

fn parse_count(value: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("invalid manifest count: {value}"))
}

const SHA256_CONSTANTS: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];
