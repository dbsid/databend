// Copyright 2021 Datafuse Labs
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;
use std::time::Instant;

use databend_common_base::runtime::profile::Profile;
use databend_common_base::runtime::profile::ProfileStatisticsName;
use databend_common_exception::ErrorCode;
use databend_common_exception::Result;
use databend_storages_common_cache::CacheAccessor;
use databend_storages_common_cache::CacheManager;
use databend_storages_common_index::BTREE_INDEX_FOOTER_TAIL_SIZE;
use databend_storages_common_index::BtreeIndexFile;
use databend_storages_common_index::BtreeIndexFileMeta;
use databend_storages_common_index::BtreeIndexFileView;
use databend_storages_common_index::btree_footer_range;
use databend_storages_common_index::decode_btree_footer_bytes;
use opendal::Operator;

#[fastrace::trace]
pub async fn load_btree_index_file(
    operator: Operator,
    location: &str,
    len_hint: Option<u64>,
) -> Result<Arc<BtreeIndexFileView>> {
    let start = Instant::now();
    let cache = CacheManager::instance().get_btree_index_file_cache();
    if let Some(file) = match len_hint {
        Some(len) => cache.get_sized(location, len),
        None => cache.get(location),
    } {
        return open_btree_index_view(file);
    }

    let data = if let Some(len) = len_hint {
        operator
            .read_with(location)
            .range(0..len)
            .await
            .map_err(|err| {
                ErrorCode::StorageOther(format!(
                    "read btree index file failed, {}, {:?}",
                    location, err
                ))
            })?
            .to_bytes()
    } else {
        operator
            .read(location)
            .await
            .map_err(|err| {
                ErrorCode::StorageOther(format!(
                    "read btree index file failed, {}, {:?}",
                    location, err
                ))
            })?
            .to_bytes()
    };
    let file = BtreeIndexFile::create(location.to_string(), data);
    let file = cache.insert(location.to_string(), file);
    let view = open_btree_index_view(file)?;
    log::debug!(
        "loaded btree index file {}, elapsed {} ms",
        location,
        start.elapsed().as_millis()
    );
    Ok(view)
}

fn open_btree_index_view(file: Arc<BtreeIndexFile>) -> Result<Arc<BtreeIndexFileView>> {
    BtreeIndexFileView::open(file.data.clone()).map(Arc::new)
}

#[fastrace::trace]
pub async fn load_btree_index_meta(
    operator: Operator,
    location: &str,
    len_hint: Option<u64>,
) -> Result<Arc<BtreeIndexFileMeta>> {
    let start = Instant::now();
    let meta_cache = CacheManager::instance().get_btree_index_meta_cache();
    if let Some(meta) = match len_hint {
        Some(len) => meta_cache.get_sized(location, len),
        None => meta_cache.get(location),
    } {
        record_elapsed(ProfileStatisticsName::BtreeIndexMetaLoadTime, start);
        return Ok(meta);
    }

    if let Some(file_cache) = CacheManager::instance().get_btree_index_file_cache()
        && let Some(file) = match len_hint {
            Some(len) => file_cache.get_sized(location, len),
            None => file_cache.get(location),
        }
    {
        let view = BtreeIndexFileView::open(file.data.clone())?;
        let meta = BtreeIndexFileMeta {
            footer: view.footer().clone(),
            index_block: view.index_block().clone(),
            filter_block: view.filter_block().clone(),
        };
        let meta = meta_cache.insert(location.to_string(), meta);
        record_elapsed(ProfileStatisticsName::BtreeIndexMetaLoadTime, start);
        return Ok(meta);
    }

    let file_len = if let Some(len) = len_hint {
        len
    } else {
        operator
            .stat(location)
            .await
            .map_err(|err| {
                ErrorCode::StorageOther(format!(
                    "stat btree index file failed, {}, {:?}",
                    location, err
                ))
            })?
            .content_length()
    };
    if file_len < BTREE_INDEX_FOOTER_TAIL_SIZE as u64 {
        return Err(ErrorCode::StorageOther(format!(
            "invalid btree index file length {}, too small",
            file_len
        )));
    }

    let tail_start = file_len - BTREE_INDEX_FOOTER_TAIL_SIZE as u64;
    let tail = read_range(
        operator.clone(),
        location,
        tail_start..file_len,
        "btree index footer tail",
    )
    .await?;
    let footer_range = btree_footer_range(file_len, &tail)?;
    let footer_bytes = read_range(
        operator.clone(),
        location,
        footer_range,
        "btree index footer",
    )
    .await?;
    let footer = decode_btree_footer_bytes(&footer_bytes)?;

    let index_section = databend_storages_common_index::btree_index_section(
        &footer,
        databend_storages_common_index::BtreeIndexSectionKind::Index,
    )?;
    let filter_section = databend_storages_common_index::btree_index_section(
        &footer,
        databend_storages_common_index::BtreeIndexSectionKind::Filter,
    )?;
    let index_block = read_range(
        operator.clone(),
        location,
        index_section.offset..index_section.offset + index_section.length,
        "btree index block",
    )
    .await?;
    let filter_block = read_range(
        operator,
        location,
        filter_section.offset..filter_section.offset + filter_section.length,
        "btree filter block",
    )
    .await?;
    let meta = BtreeIndexFileMeta::from_sections(footer, index_block, filter_block)?;
    let meta = meta_cache.insert(location.to_string(), meta);
    record_elapsed(ProfileStatisticsName::BtreeIndexMetaLoadTime, start);
    Ok(meta)
}

pub async fn load_btree_index_data_block(
    operator: Operator,
    location: &str,
    meta: &BtreeIndexFileMeta,
    block_meta: &databend_storages_common_index::BtreeIndexDataBlockMeta,
) -> Result<Vec<databend_storages_common_index::BtreeIndexRow>> {
    let cache_key = format!("{}#{}+{}", location, block_meta.offset, block_meta.length);
    let cache = CacheManager::instance().get_btree_index_file_cache();
    if let Some(block) = cache.get_sized(&cache_key, block_meta.length) {
        let start = Instant::now();
        let rows = meta.decode_data_block(block_meta, block.data.as_ref())?;
        record_elapsed(ProfileStatisticsName::BtreeIndexDataBlockDecodeTime, start);
        return Ok(rows);
    }

    let read_start = Instant::now();
    let bytes = read_range(
        operator,
        location,
        block_meta.offset..block_meta.offset + block_meta.length,
        "btree data block",
    )
    .await?;
    record_elapsed(
        ProfileStatisticsName::BtreeIndexDataBlockReadTime,
        read_start,
    );
    let block = BtreeIndexFile::create(cache_key.clone(), bytes);
    let block = cache.insert(cache_key, block);
    let decode_start = Instant::now();
    let rows = meta.decode_data_block(block_meta, block.data.as_ref())?;
    record_elapsed(
        ProfileStatisticsName::BtreeIndexDataBlockDecodeTime,
        decode_start,
    );
    Ok(rows)
}

async fn read_range(
    operator: Operator,
    location: &str,
    range: std::ops::Range<u64>,
    label: &str,
) -> Result<bytes::Bytes> {
    operator
        .read_with(location)
        .range(range)
        .await
        .map_err(|err| {
            ErrorCode::StorageOther(format!("read {label} failed, {}, {:?}", location, err))
        })
        .map(|data| data.to_bytes())
}

fn record_elapsed(name: ProfileStatisticsName, start: Instant) {
    Profile::record_usize_profile(name, start.elapsed().as_nanos() as usize);
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use bytes::Bytes;
    use databend_storages_common_index::BtreeIndexMeta;
    use databend_storages_common_index::BtreeIndexWriter;

    use super::*;

    fn row(key: &str, value: &str) -> (Bytes, Bytes) {
        (
            Bytes::copy_from_slice(key.as_bytes()),
            Bytes::copy_from_slice(value.as_bytes()),
        )
    }

    #[tokio::test]
    async fn test_load_btree_index_ranges_from_opendal() -> Result<()> {
        crate::test_utils::init_test_globals()?;
        let operator = Operator::new(opendal::services::Memory::default())
            .unwrap()
            .finish();
        let location = "btree-index.sst";
        let meta = BtreeIndexMeta {
            columns: vec![],
            metadata: BTreeMap::new(),
        };
        let mut writer = BtreeIndexWriter::new(meta, "schema", "wallet ASC,score DESC", "none")
            .with_data_block_size(1);
        writer.add_equality_prefix(Bytes::from_static(b"wallet-1|"));
        for score in 0..4 {
            let key = format!("wallet-1|{score}");
            let value = format!("payload-{score}");
            let (key, value) = row(&key, &value);
            writer.add_row(key, value);
        }
        let data = writer.finish()?;
        operator
            .write(location, data.to_vec())
            .await
            .map_err(|err| {
                ErrorCode::StorageOther(format!("write btree index test file failed: {err:?}"))
            })?;

        let meta = load_btree_index_meta(operator.clone(), location, None).await?;
        assert!(meta.may_contain_equality_prefix(b"wallet-1|"));
        let block_metas = meta.blocks_for_prefix(b"wallet-1|");
        assert!(block_metas.len() > 1);

        let mut rows = Vec::new();
        for block_meta in &block_metas {
            rows.extend(
                load_btree_index_data_block(operator.clone(), location, &meta, block_meta).await?,
            );
        }
        rows.retain(|row| row.encoded_key.starts_with(b"wallet-1|"));
        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].encoded_row_payload.as_slice(), b"payload-0");
        Ok(())
    }
}
