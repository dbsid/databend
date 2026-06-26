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

use std::collections::BTreeMap;
use std::ops::Range;

use bytes::Bytes;
use crc32fast::Hasher;
use databend_common_exception::ErrorCode;
use databend_common_exception::Result;
use databend_common_expression::Scalar;
use databend_common_expression::ScalarRef;
use databend_common_expression::row_encoding::FixedLengthEncoding;
use databend_common_native::CommonCompression;
use databend_storages_common_table_meta::meta::SingleColumnMeta;
use databend_storages_common_table_meta::table::TableCompression;
use serde::Deserialize;
use serde::Serialize;

use crate::IndexFile;
use crate::filters::BloomBuilder;
use crate::filters::BloomFilter;
use crate::filters::Filter;
use crate::filters::FilterBuilder;

const BTREE_INDEX_MAGIC: &[u8; 8] = b"DBBTREE1";
pub const BTREE_INDEX_FILE_VERSION: u32 = 2;
pub const DEFAULT_BTREE_INDEX_DATA_BLOCK_SIZE: usize = 64 * 1024;
pub const DEFAULT_BTREE_INDEX_BLOOM_BITS_PER_KEY: u64 = 10;
const BTREE_INDEX_BLOOM_SEED: u64 = 0;
const FOOTER_LEN_SIZE: usize = 4;
const MAGIC_SIZE: usize = BTREE_INDEX_MAGIC.len();
pub const BTREE_INDEX_FOOTER_TAIL_SIZE: usize = FOOTER_LEN_SIZE + MAGIC_SIZE;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BtreeIndexMeta {
    pub columns: Vec<(String, SingleColumnMeta)>,
    pub metadata: BTreeMap<String, String>,
}

pub type BtreeIndexFile = IndexFile;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BtreeIndexRow {
    pub encoded_key: Vec<u8>,
    pub encoded_row_payload: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BtreeIndexKeyOrder {
    Asc,
    Desc,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BtreeIndexSectionKind {
    Data = 0,
    Filter = 1,
    Index = 2,
    Footer = 3,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BtreeIndexSection {
    pub kind: BtreeIndexSectionKind,
    pub offset: u64,
    pub length: u64,
    pub checksum: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BtreeIndexFooter {
    pub version: u32,
    pub schema: String,
    pub key_order: String,
    pub compression: String,
    pub meta: BtreeIndexMeta,
    pub checksum: u32,
    pub sections: Vec<BtreeIndexSection>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BtreeIndexDataBlockMeta {
    pub first_key: Vec<u8>,
    pub last_key: Vec<u8>,
    pub offset: u64,
    pub length: u64,
    pub uncompressed_length: u64,
    pub row_count: u32,
    pub checksum: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BtreeIndexFilterBlock {
    pub equality_prefix_bloom: Vec<u8>,
    pub equality_prefix_count: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BtreeIndexIndexBlock {
    pub data_blocks: Vec<BtreeIndexDataBlockMeta>,
}

#[derive(Clone, Debug)]
pub struct BtreeIndexFileView {
    data: Bytes,
    footer: BtreeIndexFooter,
    index_block: BtreeIndexIndexBlock,
    filter_block: BtreeIndexFilterBlock,
    compression: CommonCompression,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BtreeIndexFileMeta {
    pub footer: BtreeIndexFooter,
    pub index_block: BtreeIndexIndexBlock,
    pub filter_block: BtreeIndexFilterBlock,
}

impl BtreeIndexFileMeta {
    pub fn from_sections(
        footer: BtreeIndexFooter,
        index_block: Bytes,
        filter_block: Bytes,
    ) -> Result<Self> {
        let index_section = btree_index_section(&footer, BtreeIndexSectionKind::Index)?;
        validate_section_checksum(index_section, &index_block, "btree index block")?;
        let filter_section = btree_index_section(&footer, BtreeIndexSectionKind::Filter)?;
        validate_section_checksum(filter_section, &filter_block, "btree filter block")?;
        let index_block: BtreeIndexIndexBlock =
            decode_from_slice(&index_block, "btree index block")?;
        let filter_block: BtreeIndexFilterBlock =
            decode_from_slice(&filter_block, "btree filter block")?;
        Ok(Self {
            footer,
            index_block,
            filter_block,
        })
    }

    pub fn footer(&self) -> &BtreeIndexFooter {
        &self.footer
    }

    pub fn index_block(&self) -> &BtreeIndexIndexBlock {
        &self.index_block
    }

    pub fn filter_block(&self) -> &BtreeIndexFilterBlock {
        &self.filter_block
    }

    pub fn may_contain_equality_prefix(&self, prefix: &[u8]) -> bool {
        let Ok((filter, _)) = BloomFilter::from_bytes(&self.filter_block.equality_prefix_bloom)
        else {
            return true;
        };
        filter.contains(prefix)
    }

    pub fn blocks_for_prefix(&self, prefix: &[u8]) -> Vec<BtreeIndexDataBlockMeta> {
        self.index_block
            .data_blocks
            .iter()
            .filter(|block| block_matches_prefix(block, prefix))
            .cloned()
            .collect()
    }

    pub fn compression(&self) -> Result<CommonCompression> {
        parse_compression(&self.footer.compression)
    }

    pub fn decode_data_block(
        &self,
        meta: &BtreeIndexDataBlockMeta,
        bytes: &[u8],
    ) -> Result<Vec<BtreeIndexRow>> {
        let compression = self.compression()?;
        decode_btree_data_block(meta, bytes, compression).map(|block| block.rows)
    }
}

impl BtreeIndexFileView {
    pub fn open(data: Bytes) -> Result<Self> {
        let footer = decode_footer(&data)?;
        validate_checksum(&data, &footer)?;
        let compression = parse_compression(&footer.compression)?;

        let index_section = btree_index_section(&footer, BtreeIndexSectionKind::Index)?;
        let index_block: BtreeIndexIndexBlock =
            decode_section(&data, index_section, "btree index block")?;

        let filter_section = btree_index_section(&footer, BtreeIndexSectionKind::Filter)?;
        let filter_block: BtreeIndexFilterBlock =
            decode_section(&data, filter_section, "btree filter block")?;

        Ok(Self {
            data,
            footer,
            index_block,
            filter_block,
            compression,
        })
    }

    pub fn footer(&self) -> &BtreeIndexFooter {
        &self.footer
    }

    pub fn index_block(&self) -> &BtreeIndexIndexBlock {
        &self.index_block
    }

    pub fn filter_block(&self) -> &BtreeIndexFilterBlock {
        &self.filter_block
    }

    pub fn may_contain_equality_prefix(&self, prefix: &[u8]) -> bool {
        let Ok((filter, _)) = BloomFilter::from_bytes(&self.filter_block.equality_prefix_bloom)
        else {
            return true;
        };
        filter.contains(prefix)
    }

    pub fn lookup_exact(&self, key: &[u8]) -> Result<Vec<BtreeIndexRow>> {
        let mut rows = self.lookup_range(key..key)?;
        rows.retain(|row| row.encoded_key.as_slice() == key);
        Ok(rows)
    }

    pub fn lookup_prefix(&self, prefix: &[u8], limit: Option<usize>) -> Result<Vec<BtreeIndexRow>> {
        let mut result = Vec::new();
        for block_meta in self.blocks_for_prefix(prefix) {
            let block = self.read_data_block(block_meta)?;
            for row in block.rows {
                if row.encoded_key.starts_with(prefix) {
                    result.push(row);
                    if limit.is_some_and(|limit| result.len() >= limit) {
                        return Ok(result);
                    }
                }
            }
        }
        Ok(result)
    }

    pub fn lookup_range(&self, range: Range<&[u8]>) -> Result<Vec<BtreeIndexRow>> {
        let mut result = Vec::new();
        for block_meta in self.blocks_for_range(range.clone()) {
            let block = self.read_data_block(block_meta)?;
            for row in block.rows {
                let key = row.encoded_key.as_slice();
                if key >= range.start && (range.end.is_empty() || key <= range.end) {
                    result.push(row);
                }
            }
        }
        Ok(result)
    }

    fn blocks_for_prefix(&self, prefix: &[u8]) -> impl Iterator<Item = &BtreeIndexDataBlockMeta> {
        self.index_block.data_blocks.iter().filter(move |block| {
            let first = block.first_key.as_slice();
            let last = block.last_key.as_slice();
            first.starts_with(prefix)
                || last.starts_with(prefix)
                || (first < prefix && prefix <= last)
        })
    }

    fn blocks_for_range(
        &self,
        range: Range<&[u8]>,
    ) -> impl Iterator<Item = &BtreeIndexDataBlockMeta> {
        self.index_block.data_blocks.iter().filter(move |block| {
            let first = block.first_key.as_slice();
            let last = block.last_key.as_slice();
            last >= range.start && (range.end.is_empty() || first <= range.end)
        })
    }

    fn read_data_block(&self, meta: &BtreeIndexDataBlockMeta) -> Result<BtreeIndexDataBlock> {
        let start = meta.offset as usize;
        let end = start + meta.length as usize;
        if end > self.data.len() {
            return Err(ErrorCode::StorageOther(format!(
                "invalid btree index data block range {}..{}, file length {}",
                start,
                end,
                self.data.len()
            )));
        }
        decode_btree_data_block(meta, &self.data[start..end], self.compression)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BtreeIndexDataBlock {
    rows: Vec<BtreeIndexRow>,
}

pub struct BtreeIndexWriter {
    meta: BtreeIndexMeta,
    schema: String,
    key_order: String,
    compression: String,
    data_block_size: usize,
    rows: Vec<BtreeIndexRow>,
    equality_prefixes: Vec<Bytes>,
}

impl BtreeIndexWriter {
    pub fn new(
        meta: BtreeIndexMeta,
        schema: impl Into<String>,
        key_order: impl Into<String>,
        compression: impl Into<String>,
    ) -> Self {
        Self {
            meta,
            schema: schema.into(),
            key_order: key_order.into(),
            compression: compression.into(),
            data_block_size: DEFAULT_BTREE_INDEX_DATA_BLOCK_SIZE,
            rows: Vec::new(),
            equality_prefixes: Vec::new(),
        }
    }

    pub fn with_data_block_size(mut self, data_block_size: usize) -> Self {
        self.data_block_size = data_block_size.max(1);
        self
    }

    pub fn add_row(&mut self, encoded_key: Bytes, encoded_row_payload: Bytes) {
        self.rows.push(BtreeIndexRow {
            encoded_key: encoded_key.to_vec(),
            encoded_row_payload: encoded_row_payload.to_vec(),
        });
    }

    pub fn add_equality_prefix(&mut self, prefix: Bytes) {
        self.equality_prefixes.push(prefix);
    }

    pub fn finish(mut self) -> Result<Bytes> {
        let compression = parse_compression(&self.compression)?;

        self.rows
            .sort_by(|left, right| left.encoded_key.cmp(&right.encoded_key));
        self.equality_prefixes.sort();
        self.equality_prefixes.dedup();

        let mut data = Vec::with_capacity(self.rows.len().saturating_mul(64));
        let data_section_offset = data.len() as u64;
        let mut data_block_metas = Vec::new();
        let mut pending_rows = Vec::new();
        let mut pending_size = 0;

        for row in self.rows {
            pending_size += row.encoded_key.len() + row.encoded_row_payload.len();
            pending_rows.push(row);
            if pending_size >= self.data_block_size {
                push_data_block(
                    &mut data,
                    &mut data_block_metas,
                    &mut pending_rows,
                    compression,
                )?;
                pending_size = 0;
            }
        }
        if !pending_rows.is_empty() {
            push_data_block(
                &mut data,
                &mut data_block_metas,
                &mut pending_rows,
                compression,
            )?;
        }
        let data_section_length = data.len() as u64 - data_section_offset;

        let filter_section_offset = data.len() as u64;
        let mut bloom_builder = BloomBuilder::create(
            std::cmp::max(
                self.equality_prefixes.len() as u64 * DEFAULT_BTREE_INDEX_BLOOM_BITS_PER_KEY,
                1,
            ),
            BTREE_INDEX_BLOOM_SEED,
        );
        for prefix in &self.equality_prefixes {
            bloom_builder.add_key(prefix);
        }
        let equality_prefix_bloom = bloom_builder.build()?.to_bytes()?;
        let filter_block = BtreeIndexFilterBlock {
            equality_prefix_bloom,
            equality_prefix_count: self.equality_prefixes.len() as u64,
        };
        let data_section_checksum = crc32fast::hash(&data[data_section_offset as usize..]);
        let filter_bytes = encode_to_vec(&filter_block, "btree filter block")?;
        let filter_section_checksum = crc32fast::hash(&filter_bytes);
        data.extend(filter_bytes);
        let filter_section_length = data.len() as u64 - filter_section_offset;

        let index_section_offset = data.len() as u64;
        let index_block = BtreeIndexIndexBlock {
            data_blocks: data_block_metas,
        };
        let index_bytes = encode_to_vec(&index_block, "btree index block")?;
        let index_section_checksum = crc32fast::hash(&index_bytes);
        data.extend(index_bytes);
        let index_section_length = data.len() as u64 - index_section_offset;

        let footer_section_offset = data.len() as u64;
        let checksum = crc32fast::hash(&data);
        let footer = BtreeIndexFooter {
            version: BTREE_INDEX_FILE_VERSION,
            schema: self.schema,
            key_order: self.key_order,
            compression: self.compression,
            meta: self.meta,
            checksum,
            sections: vec![
                BtreeIndexSection {
                    kind: BtreeIndexSectionKind::Data,
                    offset: data_section_offset,
                    length: data_section_length,
                    checksum: data_section_checksum,
                },
                BtreeIndexSection {
                    kind: BtreeIndexSectionKind::Filter,
                    offset: filter_section_offset,
                    length: filter_section_length,
                    checksum: filter_section_checksum,
                },
                BtreeIndexSection {
                    kind: BtreeIndexSectionKind::Index,
                    offset: index_section_offset,
                    length: index_section_length,
                    checksum: index_section_checksum,
                },
                BtreeIndexSection {
                    kind: BtreeIndexSectionKind::Footer,
                    offset: footer_section_offset,
                    length: 0,
                    checksum: 0,
                },
            ],
        };
        let mut footer_bytes = encode_to_vec(&footer, "btree footer")?;
        let footer_len = footer_bytes.len() as u32;
        data.append(&mut footer_bytes);
        data.extend(footer_len.to_le_bytes());
        data.extend(BTREE_INDEX_MAGIC);

        Ok(data.into())
    }
}

fn push_data_block(
    data: &mut Vec<u8>,
    data_block_metas: &mut Vec<BtreeIndexDataBlockMeta>,
    pending_rows: &mut Vec<BtreeIndexRow>,
    compression: CommonCompression,
) -> Result<()> {
    debug_assert!(!pending_rows.is_empty());

    let first_key = pending_rows.first().unwrap().encoded_key.clone();
    let last_key = pending_rows.last().unwrap().encoded_key.clone();
    let offset = data.len() as u64;
    let rows = std::mem::take(pending_rows);
    let row_count = rows.len() as u32;
    let block = BtreeIndexDataBlock { rows };
    let block_bytes = encode_to_vec(&block, "btree data block")?;
    let uncompressed_length = block_bytes.len() as u64;
    let mut compressed_bytes = Vec::new();
    let compressed_length = compression
        .compress(&block_bytes, &mut compressed_bytes)
        .map_err(|e| {
            ErrorCode::StorageOther(format!("failed to compress btree data block: {e}"))
        })?;
    let checksum = crc32fast::hash(&compressed_bytes);
    data.extend(compressed_bytes);
    data_block_metas.push(BtreeIndexDataBlockMeta {
        first_key,
        last_key,
        offset,
        length: compressed_length as u64,
        uncompressed_length,
        row_count,
        checksum,
    });
    Ok(())
}

fn parse_compression(compression: &str) -> Result<CommonCompression> {
    let table_compression = TableCompression::try_from(compression)?;
    Ok(CommonCompression::from(table_compression))
}

fn block_matches_prefix(block: &BtreeIndexDataBlockMeta, prefix: &[u8]) -> bool {
    let first = block.first_key.as_slice();
    let last = block.last_key.as_slice();
    first.starts_with(prefix) || last.starts_with(prefix) || (first < prefix && prefix <= last)
}

pub fn btree_index_section(
    footer: &BtreeIndexFooter,
    kind: BtreeIndexSectionKind,
) -> Result<&BtreeIndexSection> {
    footer
        .sections
        .iter()
        .find(|s| s.kind == kind)
        .ok_or_else(|| {
            ErrorCode::StorageOther(format!("btree index footer missing section {:?}", kind))
        })
}

fn decode_footer(data: &[u8]) -> Result<BtreeIndexFooter> {
    let min_len = MAGIC_SIZE + FOOTER_LEN_SIZE;
    if data.len() < min_len {
        return Err(ErrorCode::StorageOther(format!(
            "invalid btree index file length {}, too small",
            data.len()
        )));
    }

    let magic_start = data.len() - MAGIC_SIZE;
    if &data[magic_start..] != BTREE_INDEX_MAGIC {
        return Err(ErrorCode::StorageOther(
            "invalid btree index magic".to_string(),
        ));
    }

    let footer_len_start = magic_start - FOOTER_LEN_SIZE;
    let footer_len =
        u32::from_le_bytes(data[footer_len_start..magic_start].try_into().unwrap()) as usize;
    if footer_len > footer_len_start {
        return Err(ErrorCode::StorageOther(format!(
            "invalid btree index footer length {}, file length {}",
            footer_len,
            data.len()
        )));
    }

    let footer_start = footer_len_start - footer_len;
    let footer: BtreeIndexFooter =
        decode_from_slice(&data[footer_start..footer_len_start], "btree footer")?;
    if footer.version != BTREE_INDEX_FILE_VERSION {
        return Err(ErrorCode::StorageOther(format!(
            "unsupported btree index file version {}",
            footer.version
        )));
    }
    Ok(footer)
}

pub fn btree_footer_range(file_len: u64, tail: &[u8]) -> Result<Range<u64>> {
    let min_len = BTREE_INDEX_FOOTER_TAIL_SIZE;
    if tail.len() != min_len {
        return Err(ErrorCode::StorageOther(format!(
            "invalid btree footer tail length {}, expected {}",
            tail.len(),
            min_len
        )));
    }
    let magic_start = tail.len() - MAGIC_SIZE;
    if &tail[magic_start..] != BTREE_INDEX_MAGIC {
        return Err(ErrorCode::StorageOther(
            "invalid btree index magic".to_string(),
        ));
    }
    let footer_len_start = magic_start - FOOTER_LEN_SIZE;
    let footer_len =
        u32::from_le_bytes(tail[footer_len_start..magic_start].try_into().unwrap()) as u64;
    let footer_len_start_in_file = file_len
        .checked_sub(BTREE_INDEX_FOOTER_TAIL_SIZE as u64)
        .ok_or_else(|| {
            ErrorCode::StorageOther(format!("invalid btree index file length {}", file_len))
        })?;
    let footer_start = footer_len_start_in_file
        .checked_sub(footer_len)
        .ok_or_else(|| {
            ErrorCode::StorageOther(format!(
                "invalid btree index footer length {}, file length {}",
                footer_len, file_len
            ))
        })?;
    Ok(footer_start..footer_len_start_in_file)
}

pub fn decode_btree_footer_bytes(footer_bytes: &[u8]) -> Result<BtreeIndexFooter> {
    let footer: BtreeIndexFooter = decode_from_slice(footer_bytes, "btree footer")?;
    if footer.version != BTREE_INDEX_FILE_VERSION {
        return Err(ErrorCode::StorageOther(format!(
            "unsupported btree index file version {}",
            footer.version
        )));
    }
    Ok(footer)
}

fn validate_checksum(data: &[u8], footer: &BtreeIndexFooter) -> Result<()> {
    let footer_section = btree_index_section(footer, BtreeIndexSectionKind::Footer)?;
    let checksum_range_end = footer_section.offset as usize;
    if checksum_range_end > data.len() {
        return Err(ErrorCode::StorageOther(format!(
            "invalid btree index footer offset {}, file length {}",
            checksum_range_end,
            data.len()
        )));
    }
    let mut hasher = Hasher::new();
    hasher.update(&data[..checksum_range_end]);
    let checksum = hasher.finalize();
    if checksum != footer.checksum {
        return Err(ErrorCode::StorageOther(format!(
            "btree index checksum mismatch, expected {}, got {}",
            footer.checksum, checksum
        )));
    }
    Ok(())
}

fn decode_section<T>(data: &[u8], section: &BtreeIndexSection, label: &str) -> Result<T>
where T: for<'de> Deserialize<'de> {
    let start = section.offset as usize;
    let end = start + section.length as usize;
    if end > data.len() {
        return Err(ErrorCode::StorageOther(format!(
            "invalid {} range {}..{}, file length {}",
            label,
            start,
            end,
            data.len()
        )));
    }
    let bytes = &data[start..end];
    validate_section_checksum(section, bytes, label)?;
    decode_from_slice(bytes, label)
}

fn validate_section_checksum(section: &BtreeIndexSection, bytes: &[u8], label: &str) -> Result<()> {
    if section.checksum == 0 {
        return Ok(());
    }
    let checksum = crc32fast::hash(bytes);
    if checksum != section.checksum {
        return Err(ErrorCode::StorageOther(format!(
            "{} checksum mismatch, expected {}, got {}",
            label, section.checksum, checksum
        )));
    }
    Ok(())
}

fn decode_btree_data_block(
    meta: &BtreeIndexDataBlockMeta,
    bytes: &[u8],
    compression: CommonCompression,
) -> Result<BtreeIndexDataBlock> {
    let checksum = crc32fast::hash(bytes);
    if checksum != meta.checksum {
        return Err(ErrorCode::StorageOther(format!(
            "btree index data block checksum mismatch, expected {}, got {}",
            meta.checksum, checksum
        )));
    }
    let mut decompressed = vec![0; meta.uncompressed_length as usize];
    compression
        .decompress(bytes, &mut decompressed)
        .map_err(|e| {
            ErrorCode::StorageOther(format!("failed to decompress btree data block: {e}"))
        })?;
    decode_from_slice(&decompressed, "btree data block")
}

fn encode_to_vec<T: Serialize + ?Sized>(value: &T, label: &str) -> Result<Vec<u8>> {
    bincode::serde::encode_to_vec(value, bincode::config::standard())
        .map_err(|e| ErrorCode::StorageOther(format!("failed to encode {label}: {e:?}")))
}

fn decode_from_slice<T>(value: &[u8], label: &str) -> Result<T>
where T: for<'de> Deserialize<'de> {
    bincode::serde::decode_from_slice(value, bincode::config::standard())
        .map(|(v, len)| {
            assert_eq!(len, value.len());
            v
        })
        .map_err(|e| ErrorCode::StorageOther(format!("failed to decode {label}: {e:?}")))
}

impl TryFrom<&BtreeIndexMeta> for Vec<u8> {
    type Error = ErrorCode;

    fn try_from(value: &BtreeIndexMeta) -> std::result::Result<Self, Self::Error> {
        encode_to_vec(value, "btree index meta")
    }
}

impl TryFrom<Bytes> for BtreeIndexMeta {
    type Error = ErrorCode;

    fn try_from(value: Bytes) -> std::result::Result<Self, Self::Error> {
        decode_from_slice(value.as_ref(), "btree index meta")
    }
}

impl TryFrom<&BtreeIndexFileMeta> for Vec<u8> {
    type Error = ErrorCode;

    fn try_from(value: &BtreeIndexFileMeta) -> std::result::Result<Self, Self::Error> {
        encode_to_vec(value, "btree index file meta")
    }
}

impl TryFrom<Bytes> for BtreeIndexFileMeta {
    type Error = ErrorCode;

    fn try_from(value: Bytes) -> std::result::Result<Self, Self::Error> {
        decode_from_slice(value.as_ref(), "btree index file meta")
    }
}

pub fn encode_btree_payload(payload: &[Scalar]) -> Result<Vec<u8>> {
    encode_to_vec(payload, "btree row payload")
}

pub fn decode_btree_payload(payload: &[u8]) -> Result<Vec<Scalar>> {
    decode_from_slice(payload, "btree row payload")
}

pub fn btree_equality_prefix(key: &[u8], component_count: usize) -> Result<Vec<u8>> {
    let mut offset = 0;
    for _ in 0..component_count {
        let Some(len_bytes) = key.get(offset..offset + 4) else {
            return Err(ErrorCode::StorageOther(
                "invalid btree key encoding, missing component length".to_string(),
            ));
        };
        let len = u32::from_be_bytes(len_bytes.try_into().unwrap()) as usize;
        offset += 4 + len;
        if key.get(offset) != Some(&0xff) {
            return Err(ErrorCode::StorageOther(
                "invalid btree key encoding, missing component separator".to_string(),
            ));
        }
        offset += 1;
    }
    Ok(key[..offset].to_vec())
}

pub fn encode_btree_key_component(
    buf: &mut Vec<u8>,
    scalar: ScalarRef<'_>,
    order: BtreeIndexKeyOrder,
) -> Result<()> {
    let mut bytes = Vec::new();
    match scalar {
        ScalarRef::Null => bytes.push(0),
        ScalarRef::EmptyArray => bytes.push(1),
        ScalarRef::EmptyMap => bytes.push(2),
        ScalarRef::Boolean(v) => bytes.extend_from_slice(v.encode().as_ref()),
        ScalarRef::Number(v) => encode_number(&mut bytes, v),
        ScalarRef::Decimal(v) => encode_decimal(&mut bytes, v),
        ScalarRef::Timestamp(v) => bytes.extend_from_slice(v.encode().as_ref()),
        ScalarRef::TimestampTz(v) => bytes.extend_from_slice(v.encode().as_ref()),
        ScalarRef::Date(v) => bytes.extend_from_slice(v.encode().as_ref()),
        ScalarRef::Interval(v) => bytes.extend_from_slice(v.encode().as_ref()),
        ScalarRef::String(v) => encode_variable(&mut bytes, v.as_bytes()),
        ScalarRef::Binary(v)
        | ScalarRef::Bitmap(v)
        | ScalarRef::Variant(v)
        | ScalarRef::Geometry(v) => encode_variable(&mut bytes, v),
        ScalarRef::Tuple(fields) => {
            for field in fields {
                encode_btree_key_component(&mut bytes, field, BtreeIndexKeyOrder::Asc)?;
            }
        }
        ScalarRef::Array(_)
        | ScalarRef::Map(_)
        | ScalarRef::Geography(_)
        | ScalarRef::Vector(_)
        | ScalarRef::Opaque(_) => {
            let payload = encode_btree_payload(&[scalar.to_owned()])?;
            encode_variable(&mut bytes, &payload);
        }
    }
    if matches!(order, BtreeIndexKeyOrder::Desc) {
        for byte in &mut bytes {
            *byte = !*byte;
        }
    }
    buf.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    buf.extend_from_slice(&bytes);
    buf.push(0xff);
    Ok(())
}

fn encode_variable(buf: &mut Vec<u8>, bytes: &[u8]) {
    buf.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    buf.extend_from_slice(bytes);
}

fn encode_number(buf: &mut Vec<u8>, number: databend_common_expression::types::NumberScalar) {
    use databend_common_expression::types::NumberScalar;
    match number {
        NumberScalar::UInt8(v) => buf.extend_from_slice(v.encode().as_ref()),
        NumberScalar::UInt16(v) => buf.extend_from_slice(v.encode().as_ref()),
        NumberScalar::UInt32(v) => buf.extend_from_slice(v.encode().as_ref()),
        NumberScalar::UInt64(v) => buf.extend_from_slice(v.encode().as_ref()),
        NumberScalar::Int8(v) => buf.extend_from_slice(v.encode().as_ref()),
        NumberScalar::Int16(v) => buf.extend_from_slice(v.encode().as_ref()),
        NumberScalar::Int32(v) => buf.extend_from_slice(v.encode().as_ref()),
        NumberScalar::Int64(v) => buf.extend_from_slice(v.encode().as_ref()),
        NumberScalar::Float32(v) => buf.extend_from_slice(v.encode().as_ref()),
        NumberScalar::Float64(v) => buf.extend_from_slice(v.encode().as_ref()),
    }
}

fn encode_decimal(buf: &mut Vec<u8>, decimal: databend_common_expression::types::DecimalScalar) {
    use databend_common_expression::types::DecimalScalar;
    match decimal {
        DecimalScalar::Decimal64(v, size) => {
            buf.push(size.precision());
            buf.push(size.scale());
            buf.extend_from_slice(v.encode().as_ref());
        }
        DecimalScalar::Decimal128(v, size) => {
            buf.push(size.precision());
            buf.push(size.scale());
            buf.extend_from_slice(v.encode().as_ref());
        }
        DecimalScalar::Decimal256(v, size) => {
            buf.push(size.precision());
            buf.push(size.scale());
            buf.extend_from_slice(v.encode().as_ref());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(key: &str, value: &str) -> (Bytes, Bytes) {
        (
            Bytes::copy_from_slice(key.as_bytes()),
            Bytes::copy_from_slice(value.as_bytes()),
        )
    }

    #[test]
    fn test_btree_index_sst_round_trip() -> Result<()> {
        let meta = BtreeIndexMeta {
            columns: vec![],
            metadata: BTreeMap::new(),
        };
        let mut writer =
            BtreeIndexWriter::new(meta, "schema", "a ASC,b DESC", "none").with_data_block_size(16);
        writer.add_equality_prefix(Bytes::from_static(b"wallet-1"));
        let (key, value) = row("wallet-2|14|1", "payload-2");
        writer.add_row(key, value);
        let (key, value) = row("wallet-1|14|2", "payload-1");
        writer.add_row(key, value);

        let data = writer.finish()?;
        let view = BtreeIndexFileView::open(data)?;

        assert_eq!(view.footer().version, BTREE_INDEX_FILE_VERSION);
        assert!(view.may_contain_equality_prefix(b"wallet-1"));
        assert!(!view.may_contain_equality_prefix(b"wallet-404"));

        let rows = view.lookup_prefix(b"wallet-1", Some(100))?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].encoded_row_payload.as_slice(), b"payload-1");
        Ok(())
    }

    #[test]
    fn test_btree_index_prefix_lookup_across_data_blocks() -> Result<()> {
        let meta = BtreeIndexMeta {
            columns: vec![],
            metadata: BTreeMap::new(),
        };
        let mut writer = BtreeIndexWriter::new(meta, "schema", "wallet ASC,score DESC", "none")
            .with_data_block_size(1);
        writer.add_equality_prefix(Bytes::from_static(b"wallet-1|"));
        for score in (0..10).rev() {
            let key = format!("wallet-1|{:02}", score);
            let value = format!("payload-{score}");
            let (key, value) = row(&key, &value);
            writer.add_row(key, value);
        }

        let data = writer.finish()?;
        let view = BtreeIndexFileView::open(data)?;

        assert!(view.index_block().data_blocks.len() > 1);
        assert!(view.may_contain_equality_prefix(b"wallet-1|"));
        let rows = view.lookup_prefix(b"wallet-1|", Some(100))?;
        assert_eq!(rows.len(), 10);
        assert_eq!(rows[0].encoded_key.as_slice(), b"wallet-1|00");
        assert_eq!(rows[9].encoded_key.as_slice(), b"wallet-1|09");
        Ok(())
    }

    #[test]
    fn test_btree_index_footer_range_from_tail() -> Result<()> {
        let meta = BtreeIndexMeta {
            columns: vec![],
            metadata: BTreeMap::new(),
        };
        let mut writer = BtreeIndexWriter::new(meta, "schema", "wallet ASC", "none");
        writer.add_equality_prefix(Bytes::from_static(b"wallet-1"));
        let (key, value) = row("wallet-1", "payload-1");
        writer.add_row(key, value);

        let data = writer.finish()?;
        let file_len = data.len() as u64;
        let tail = &data[data.len() - BTREE_INDEX_FOOTER_TAIL_SIZE..];
        let footer_range = btree_footer_range(file_len, tail)?;
        let footer = decode_btree_footer_bytes(
            &data[footer_range.start as usize..footer_range.end as usize],
        )?;

        assert_eq!(footer.version, BTREE_INDEX_FILE_VERSION);
        assert_eq!(footer.key_order, "wallet ASC");
        Ok(())
    }
}
