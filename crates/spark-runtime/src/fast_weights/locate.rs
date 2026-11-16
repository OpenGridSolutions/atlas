// SPDX-License-Identifier: AGPL-3.0-only

//! Locating checkpoint tensors on disk without loading them.

use anyhow::{Context, Result};
use std::fs::File;
use std::path::Path;

use super::header::{parse_header, resolve_shards};
use crate::weights::DiskTensor;

/// Finds every tensor of the checkpoint in `model_dir` whose name `keep`
/// accepts and returns where its data starts. Only the shard headers are
/// read.
pub fn locate_tensors(
    model_dir: &Path,
    keep: fn(&str) -> bool,
) -> Result<Vec<(String, DiskTensor)>> {
    let (shards, _) = resolve_shards(model_dir)?;
    let mut found = Vec::new();
    for path in shards {
        let mut file =
            File::open(&path).with_context(|| format!("Failed to open {}", path.display()))?;
        for t in parse_header(&mut file)? {
            if keep(&t.name) {
                found.push((
                    t.name,
                    DiskTensor {
                        path: path.clone(),
                        data_offset: t.abs_offset,
                        shape: t.shape,
                        dtype: t.dtype,
                    },
                ));
            }
        }
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::WeightDtype;
    use std::io::Write;

    /// A one-file checkpoint: `a` (U8 [2, 3]), `table.shard_0` (U8 [4, 2]).
    fn write_checkpoint(dir: &Path) -> Vec<u8> {
        let header = br#"{"a":{"dtype":"U8","shape":[2,3],"data_offsets":[0,6]},"table.shard_0":{"dtype":"U8","shape":[4,2],"data_offsets":[6,14]}}"#;
        let data: Vec<u8> = (100u8..114).collect();
        let mut f = File::create(dir.join("model.safetensors")).expect("create");
        f.write_all(&(header.len() as u64).to_le_bytes())
            .expect("len");
        f.write_all(header).expect("header");
        f.write_all(&data).expect("data");
        data
    }

    #[test]
    fn finds_the_selected_tensors_with_absolute_offsets() {
        let dir = tempfile::tempdir().expect("tempdir");
        let data = write_checkpoint(dir.path());
        let found = locate_tensors(dir.path(), |n| n.starts_with("table.")).expect("locate");
        assert_eq!(found.len(), 1);
        let (name, t) = &found[0];
        assert_eq!(name, "table.shard_0");
        assert_eq!((t.shape.clone(), t.dtype), (vec![4, 2], WeightDtype::UInt8));
        let bytes = std::fs::read(&t.path).expect("read");
        let start = t.data_offset as usize;
        assert_eq!(&bytes[start..start + 8], &data[6..14]);
    }

    #[test]
    fn selecting_nothing_returns_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_checkpoint(dir.path());
        assert!(
            locate_tensors(dir.path(), |_| false)
                .expect("locate")
                .is_empty()
        );
    }

    #[test]
    fn a_directory_without_checkpoint_is_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(locate_tensors(dir.path(), |_| true).is_err());
    }
}
