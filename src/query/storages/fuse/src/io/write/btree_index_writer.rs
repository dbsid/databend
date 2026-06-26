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
use std::collections::BTreeSet;

use bytes::Bytes;
use databend_common_exception::ErrorCode;
use databend_common_exception::Result;
use databend_common_expression::DataBlock;
use databend_common_expression::TableField;
use databend_common_expression::TableSchemaRef;
use databend_common_meta_app::schema::TableIndexColumnOrder;
use databend_common_meta_app::schema::TableIndexType;
use databend_common_meta_app::schema::TableMeta;
use databend_storages_common_index::BtreeIndexKeyOrder;
use databend_storages_common_index::BtreeIndexMeta;
use databend_storages_common_index::BtreeIndexWriter;
use databend_storages_common_index::btree_equality_prefix;
use databend_storages_common_index::encode_btree_key_component;
use databend_storages_common_index::encode_btree_payload;
use databend_storages_common_table_meta::meta::Location;
use databend_storages_common_table_meta::meta::SingleColumnMeta;
use databend_storages_common_table_meta::table::TableCompression;
use log::debug;

use crate::io::TableMetaLocationGenerator;

const BTREE_INDEX_OPTION_COMPRESSION: &str = "compression";
const BTREE_INDEX_OPTION_COVERED_TYPE: &str = "index_covered_type";
const BTREE_INDEX_COVERED_ALL_COLUMNS: &str = "covered_all_columns_in_schema";

#[derive(Clone)]
pub struct BtreeIndexBuilder {
    pub(crate) name: String,
    pub(crate) version: String,
    pub(crate) key_fields: Vec<(TableField, TableIndexColumnOrder)>,
    pub(crate) payload_fields: Vec<TableField>,
    pub(crate) options: BTreeMap<String, String>,
}

impl BtreeIndexBuilder {
    pub fn gen_btree_index_location(&self, block_location: &Location) -> String {
        TableMetaLocationGenerator::gen_btree_index_location_from_block_location(
            &block_location.0,
            &self.name,
            &self.version,
        )
    }
}

pub fn create_btree_index_builders(table_meta: &TableMeta) -> Vec<BtreeIndexBuilder> {
    let mut btree_index_builders = Vec::with_capacity(table_meta.indexes.len());
    for index in table_meta.indexes.values() {
        if !matches!(index.index_type, TableIndexType::Btree) {
            continue;
        }
        if !index.sync_creation {
            continue;
        }

        let key_columns = if index.key_columns.is_empty() {
            index
                .column_ids
                .iter()
                .map(
                    |column_id| databend_common_meta_app::schema::TableIndexColumn {
                        column_id: *column_id,
                        order: TableIndexColumnOrder::Asc,
                    },
                )
                .collect::<Vec<_>>()
        } else {
            index.key_columns.clone()
        };
        let mut key_fields = Vec::with_capacity(key_columns.len());
        for key_column in &key_columns {
            match table_meta.schema.field_of_column_id(key_column.column_id) {
                Ok(field) => key_fields.push((field.clone(), key_column.order.clone())),
                Err(_) => {
                    debug!(
                        "Ignoring invalid btree index: {}, missing key column id {}",
                        index.name, key_column.column_id
                    );
                    key_fields.clear();
                    break;
                }
            }
        }
        if key_fields.len() != key_columns.len() {
            continue;
        }

        let payload_fields = if is_covered_all_columns(&index.options) {
            table_meta
                .schema
                .remove_virtual_computed_fields()
                .fields
                .clone()
        } else {
            let mut payload_fields =
                Vec::with_capacity(key_columns.len() + index.include_column_ids.len());
            let mut seen_column_ids = BTreeSet::new();
            for key_column in &key_columns {
                if !seen_column_ids.insert(key_column.column_id) {
                    continue;
                }
                let Ok(field) = table_meta.schema.field_of_column_id(key_column.column_id) else {
                    payload_fields.clear();
                    break;
                };
                payload_fields.push(field.clone());
            }
            if payload_fields.len() != seen_column_ids.len() {
                continue;
            }
            for column_id in &index.include_column_ids {
                if !seen_column_ids.insert(*column_id) {
                    continue;
                }
                match table_meta.schema.field_of_column_id(*column_id) {
                    Ok(field) => payload_fields.push(field.clone()),
                    Err(_) => {
                        debug!(
                            "Ignoring invalid btree index: {}, missing include column id {}",
                            index.name, column_id
                        );
                        payload_fields.clear();
                        break;
                    }
                }
            }
            if payload_fields.len() != seen_column_ids.len() {
                continue;
            }
            payload_fields
        };

        btree_index_builders.push(BtreeIndexBuilder {
            name: index.name.clone(),
            version: index.version.clone(),
            key_fields,
            payload_fields,
            options: index.options.clone(),
        });
    }
    btree_index_builders
}

#[derive(Debug)]
pub struct BtreeIndexState {
    pub(crate) data: Vec<u8>,
    pub(crate) size: u64,
    pub(crate) location: Location,
}

impl BtreeIndexState {
    pub fn try_create(data: Vec<u8>, location: String) -> Result<Self> {
        let size = data.len() as u64;
        Ok(Self {
            data,
            size,
            location: (location, 0),
        })
    }

    pub fn from_data_block(
        source_schema: &TableSchemaRef,
        block: &DataBlock,
        block_location: &Location,
        btree_index_builder: &BtreeIndexBuilder,
    ) -> Result<Self> {
        let location = btree_index_builder.gen_btree_index_location(block_location);
        let data = build_btree_index(source_schema, block, btree_index_builder)?;
        Self::try_create(data, location)
    }
}

pub fn build_btree_index(
    source_schema: &TableSchemaRef,
    block: &DataBlock,
    btree_index_builder: &BtreeIndexBuilder,
) -> Result<Vec<u8>> {
    let compression = btree_index_builder
        .options
        .get(BTREE_INDEX_OPTION_COMPRESSION)
        .cloned()
        .unwrap_or_else(|| "zstd".to_string());
    // Validate early so invalid DDL options fail before producing index bytes.
    let _ = TableCompression::try_from(compression.as_str())?;

    let full_block = block.convert_to_full();
    let key_field_indexes = key_field_indexes(source_schema, &btree_index_builder.key_fields)?;
    let payload_field_indexes = field_indexes(source_schema, &btree_index_builder.payload_fields)?;
    let schema = serde_json::to_string(&btree_index_builder.payload_fields).map_err(|e| {
        ErrorCode::StorageOther(format!("failed to encode btree payload schema: {e:?}"))
    })?;
    let key_order = btree_index_builder
        .key_fields
        .iter()
        .map(|(field, order)| match order {
            TableIndexColumnOrder::Asc => format!("{} ASC", field.name()),
            TableIndexColumnOrder::Desc => format!("{} DESC", field.name()),
        })
        .collect::<Vec<_>>()
        .join(",");
    let meta = BtreeIndexMeta {
        columns: btree_index_builder
            .payload_fields
            .iter()
            .map(|field| {
                (
                    field.name().to_string(),
                    SingleColumnMeta::new(0, 0, block.num_rows() as u64),
                )
            })
            .collect(),
        metadata: BTreeMap::from([
            ("index_name".to_string(), btree_index_builder.name.clone()),
            (
                "index_version".to_string(),
                btree_index_builder.version.clone(),
            ),
            ("encoding".to_string(), "databend-btree-sst-v1".to_string()),
        ]),
    };
    let mut writer = BtreeIndexWriter::new(meta, schema, key_order, compression);
    let key_column_count = key_field_indexes.len();
    for row in 0..full_block.num_rows() {
        let mut key = Vec::new();
        for (field_index, order) in &key_field_indexes {
            let scalar = unsafe { full_block.get_by_offset(*field_index).index_unchecked(row) };
            encode_btree_key_component(&mut key, scalar, *order)?;
        }

        for prefix_column_count in 1..=key_column_count {
            writer.add_equality_prefix(Bytes::copy_from_slice(
                btree_equality_prefix(&key, prefix_column_count)?.as_slice(),
            ));
        }

        let mut payload = Vec::with_capacity(payload_field_indexes.len());
        for field_index in &payload_field_indexes {
            let scalar = unsafe { full_block.get_by_offset(*field_index).index_unchecked(row) };
            payload.push(scalar.to_owned());
        }
        let payload = encode_btree_payload(&payload)?;
        writer.add_row(Bytes::from(key), Bytes::from(payload));
    }
    writer.finish().map(|bytes| bytes.to_vec())
}

fn is_covered_all_columns(options: &BTreeMap<String, String>) -> bool {
    options
        .get(BTREE_INDEX_OPTION_COVERED_TYPE)
        .is_some_and(|value| value.eq_ignore_ascii_case(BTREE_INDEX_COVERED_ALL_COLUMNS))
}

fn field_indexes(schema: &TableSchemaRef, fields: &[TableField]) -> Result<Vec<usize>> {
    fields
        .iter()
        .map(|field| schema.index_of(field.name()))
        .collect()
}

fn key_field_indexes(
    schema: &TableSchemaRef,
    fields: &[(TableField, TableIndexColumnOrder)],
) -> Result<Vec<(usize, BtreeIndexKeyOrder)>> {
    fields
        .iter()
        .map(|(field, order)| Ok((schema.index_of(field.name())?, to_btree_key_order(order))))
        .collect()
}

fn to_btree_key_order(order: &TableIndexColumnOrder) -> BtreeIndexKeyOrder {
    match order {
        TableIndexColumnOrder::Asc => BtreeIndexKeyOrder::Asc,
        TableIndexColumnOrder::Desc => BtreeIndexKeyOrder::Desc,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use databend_common_expression::DataBlock;
    use databend_common_expression::FromData;
    use databend_common_expression::Scalar;
    use databend_common_expression::ScalarRef;
    use databend_common_expression::TableDataType;
    use databend_common_expression::TableField;
    use databend_common_expression::TableSchemaRefExt;
    use databend_common_expression::types::NumberDataType;
    use databend_common_expression::types::StringType;
    use databend_common_expression::types::number::UInt64Type;
    use databend_storages_common_index::BtreeIndexFileView;

    use super::*;

    #[test]
    fn test_build_btree_index_sst_from_block() -> Result<()> {
        let wallet = TableField::new_from_column_id("wallet_address", TableDataType::String, 0);
        let platform = TableField::new_from_column_id(
            "platform_id",
            TableDataType::Number(NumberDataType::UInt64),
            1,
        );
        let balance = TableField::new_from_column_id(
            "balance",
            TableDataType::Number(NumberDataType::UInt64),
            2,
        );
        let schema =
            TableSchemaRefExt::create(vec![wallet.clone(), platform.clone(), balance.clone()]);
        let block = DataBlock::new_from_columns(vec![
            StringType::from_data(vec!["wallet-b", "wallet-a", "wallet-a"]),
            UInt64Type::from_data(vec![14, 14, 14]),
            UInt64Type::from_data(vec![20, 30, 40]),
        ]);
        let builder = BtreeIndexBuilder {
            name: "idx_wallet".to_string(),
            version: "123456789".to_string(),
            key_fields: vec![
                (wallet.clone(), TableIndexColumnOrder::Asc),
                (platform.clone(), TableIndexColumnOrder::Asc),
                (balance.clone(), TableIndexColumnOrder::Desc),
            ],
            payload_fields: vec![wallet.clone(), platform.clone(), balance.clone()],
            options: BTreeMap::from([("compression".to_string(), "none".to_string())]),
        };

        let data = build_btree_index(&schema, &block, &builder)?;
        let view = Arc::new(BtreeIndexFileView::open(data.into())?);

        let mut prefix = Vec::new();
        encode_btree_key_component(
            &mut prefix,
            ScalarRef::String("wallet-a"),
            BtreeIndexKeyOrder::Asc,
        )?;
        encode_btree_key_component(
            &mut prefix,
            ScalarRef::Number(databend_common_expression::types::NumberScalar::UInt64(14)),
            BtreeIndexKeyOrder::Asc,
        )?;

        assert!(view.may_contain_equality_prefix(&prefix));
        let rows = view.lookup_prefix(&prefix, Some(100))?;
        assert_eq!(rows.len(), 2);
        assert_eq!(
            view.footer().key_order,
            "wallet_address ASC,platform_id ASC,balance DESC"
        );
        assert!(view.footer().schema.contains("\"balance\""));
        assert!(view.footer().schema.contains("\"platform_id\""));

        let first_payload = decode_payload_for_test(&rows[0].encoded_row_payload)?;
        let second_payload = decode_payload_for_test(&rows[1].encoded_row_payload)?;
        assert_eq!(first_payload, vec![
            Scalar::String("wallet-a".to_string()),
            Scalar::Number(databend_common_expression::types::NumberScalar::UInt64(14)),
            Scalar::Number(databend_common_expression::types::NumberScalar::UInt64(40)),
        ]);
        assert_eq!(second_payload, vec![
            Scalar::String("wallet-a".to_string()),
            Scalar::Number(databend_common_expression::types::NumberScalar::UInt64(14)),
            Scalar::Number(databend_common_expression::types::NumberScalar::UInt64(30)),
        ]);
        Ok(())
    }

    fn decode_payload_for_test(payload: &[u8]) -> Result<Vec<Scalar>> {
        databend_storages_common_index::decode_btree_payload(payload)
    }
}
