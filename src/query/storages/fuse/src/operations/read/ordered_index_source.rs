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

use std::cmp::Ordering;
use std::sync::Arc;
use std::sync::LazyLock;
use std::time::Instant;

use async_channel::Receiver;
use databend_common_base::runtime::profile::Profile;
use databend_common_base::runtime::profile::ProfileStatisticsName;
use databend_common_catalog::plan::DataSourcePlan;
use databend_common_catalog::plan::Filters;
use databend_common_catalog::plan::OrderedIndexColumnOrder;
use databend_common_catalog::plan::OrderedIndexInfo;
use databend_common_catalog::plan::PartInfoPtr;
use databend_common_catalog::plan::StealablePartitions;
use databend_common_catalog::table_context::TableContext;
use databend_common_exception::ErrorCode;
use databend_common_exception::Result;
use databend_common_expression::ColumnBuilder;
use databend_common_expression::DataBlock;
use databend_common_expression::DataSchema;
use databend_common_expression::Evaluator;
use databend_common_expression::Expr;
use databend_common_expression::FunctionContext;
use databend_common_expression::Scalar;
use databend_common_expression::TableDataType;
use databend_common_expression::TableField;
use databend_common_expression::types::BooleanType;
use databend_common_expression::types::DataType;
use databend_common_expression::types::Decimal;
use databend_common_expression::types::DecimalDataType;
use databend_common_expression::types::DecimalScalar;
use databend_common_expression::types::NumberDataType;
use databend_common_expression::types::NumberScalar;
use databend_common_expression::types::i256;
use databend_common_functions::BUILTIN_FUNCTIONS;
use databend_common_pipeline::core::OutputPort;
use databend_common_pipeline::core::Pipeline;
use databend_common_pipeline::core::SourcePipeBuilder;
use databend_common_pipeline::sources::AsyncSource;
use databend_common_pipeline::sources::AsyncSourcer;
use databend_storages_common_cache::CacheAccessor;
use databend_storages_common_cache::CacheValue;
use databend_storages_common_cache::InMemoryLruCache;
use databend_storages_common_index::OrderedIndexDataBlockMeta;
use databend_storages_common_index::OrderedIndexFileMeta;
use databend_storages_common_index::OrderedIndexKeyOrder;
use databend_storages_common_index::OrderedIndexRow;
use databend_storages_common_index::decode_ordered_payload;
use databend_storages_common_index::decode_ordered_payload_projection;
use databend_storages_common_index::encode_ordered_key_component;
use databend_storages_common_index::filters::BloomFilter;
use databend_storages_common_index::filters::Filter as _;
use databend_storages_common_index::ordered_equality_prefix;
use futures::future;
use opendal::Operator;
use sha2::Digest;

use crate::fuse_part::FuseBlockPartInfo;
use crate::io::TableMetaLocationGenerator;
use crate::io::load_ordered_index_data_block;
use crate::io::load_ordered_index_meta;

const ORDERED_INDEX_DATA_BLOCK_READ_BATCH_SIZE: usize = 32;
const ORDERED_DECODED_PAYLOAD_ROW_CACHE_ITEMS: usize = 65536;
const ORDERED_DECODED_EQUALITY_PREFIX_BLOOM_CACHE_ITEMS: usize = 8192;
const ORDERED_RESOLVED_INDEX_LOCATION_CACHE_ITEMS: usize = 65536;

static ORDERED_DECODED_PAYLOAD_ROW_CACHE: LazyLock<
    InMemoryLruCache<OrderedIndexDecodedPayloadRow>,
> = LazyLock::new(|| {
    InMemoryLruCache::with_items_capacity(
        "ordered_index_decoded_payload_row".to_string(),
        ORDERED_DECODED_PAYLOAD_ROW_CACHE_ITEMS,
    )
});

static ORDERED_DECODED_EQUALITY_PREFIX_BLOOM_CACHE: LazyLock<
    InMemoryLruCache<OrderedIndexDecodedEqualityPrefixBloom>,
> = LazyLock::new(|| {
    InMemoryLruCache::with_items_capacity(
        "ordered_index_decoded_equality_prefix_bloom".to_string(),
        ORDERED_DECODED_EQUALITY_PREFIX_BLOOM_CACHE_ITEMS,
    )
});

static ORDERED_RESOLVED_INDEX_LOCATION_CACHE: LazyLock<
    InMemoryLruCache<OrderedIndexResolvedLocation>,
> = LazyLock::new(|| {
    InMemoryLruCache::with_items_capacity(
        "ordered_index_resolved_location".to_string(),
        ORDERED_RESOLVED_INDEX_LOCATION_CACHE_ITEMS,
    )
});

#[derive(Clone)]
struct OrderedIndexDecodedPayloadRow {
    row: Vec<Scalar>,
}

impl From<OrderedIndexDecodedPayloadRow> for CacheValue<OrderedIndexDecodedPayloadRow> {
    fn from(value: OrderedIndexDecodedPayloadRow) -> Self {
        let mem_bytes = std::mem::size_of::<OrderedIndexDecodedPayloadRow>()
            + scalar_rows_mem_bytes(&value.row);
        CacheValue::new(value, mem_bytes)
    }
}

#[derive(Clone)]
struct OrderedIndexResolvedLocation {
    location: String,
}

impl From<OrderedIndexResolvedLocation> for CacheValue<OrderedIndexResolvedLocation> {
    fn from(value: OrderedIndexResolvedLocation) -> Self {
        let mem_bytes = std::mem::size_of::<OrderedIndexResolvedLocation>() + value.location.len();
        CacheValue::new(value, mem_bytes)
    }
}

#[derive(Clone)]
struct OrderedIndexDecodedEqualityPrefixBloom {
    filter: Option<Arc<BloomFilter>>,
    mem_bytes: usize,
}

impl From<OrderedIndexDecodedEqualityPrefixBloom>
    for CacheValue<OrderedIndexDecodedEqualityPrefixBloom>
{
    fn from(value: OrderedIndexDecodedEqualityPrefixBloom) -> Self {
        let mem_bytes =
            value.mem_bytes + std::mem::size_of::<OrderedIndexDecodedEqualityPrefixBloom>();
        CacheValue::new(value, mem_bytes)
    }
}

pub fn build_ordered_index_source_pipeline(
    ctx: Arc<dyn TableContext>,
    operator: Operator,
    pipeline: &mut Pipeline,
    plan: &DataSourcePlan,
    ordered_index: OrderedIndexInfo,
    max_threads: usize,
    receiver: Option<Receiver<Result<PartInfoPtr>>>,
) -> Result<()> {
    let _ = max_threads;
    // ORDERED scans merge candidates globally to preserve ordered limit semantics.
    let max_threads = 1;
    let partitions =
        crate::operations::read::fuse_source::dispatch_partitions(ctx.clone(), plan, max_threads);
    let partitions = StealablePartitions::new(partitions, ctx.clone());
    let output_schema: DataSchema = plan.schema().as_ref().into();
    let payload_schema: DataSchema = DataSchema::new(
        ordered_index
            .payload_fields
            .iter()
            .map(|field| {
                databend_common_expression::DataField::new(
                    field.name(),
                    DataType::from(field.data_type()),
                )
            })
            .collect(),
    );
    let func_ctx = ctx.get_function_context()?;

    let mut source_builder = SourcePipeBuilder::create();
    for worker_id in 0..max_threads {
        let output = OutputPort::create();
        source_builder.add_source(
            output.clone(),
            OrderedIndexSource::create(
                ctx.clone(),
                output,
                operator.clone(),
                output_schema.clone(),
                payload_schema.clone(),
                func_ctx.clone(),
                Some(partitions.clone()),
                receiver.clone(),
                ordered_index.clone(),
                worker_id,
            )?,
        );
    }
    pipeline.add_pipe(source_builder.finalize());
    Ok(())
}

struct OrderedIndexSource {
    operator: Operator,
    output_schema: DataSchema,
    payload_schema: DataSchema,
    func_ctx: FunctionContext,
    partitions: Option<StealablePartitions>,
    receiver: Option<Receiver<Result<PartInfoPtr>>>,
    ordered_index: OrderedIndexInfo,
    worker_id: usize,
    is_finished: bool,
    location_preference: OrderedIndexLocationPreference,
}

impl OrderedIndexSource {
    #[allow(clippy::too_many_arguments)]
    fn create(
        ctx: Arc<dyn TableContext>,
        output: Arc<OutputPort>,
        operator: Operator,
        output_schema: DataSchema,
        payload_schema: DataSchema,
        func_ctx: FunctionContext,
        partitions: Option<StealablePartitions>,
        receiver: Option<Receiver<Result<PartInfoPtr>>>,
        ordered_index: OrderedIndexInfo,
        worker_id: usize,
    ) -> Result<databend_common_pipeline::core::ProcessorPtr> {
        AsyncSourcer::create(ctx.get_scan_progress(), output, Self {
            operator,
            output_schema,
            payload_schema,
            func_ctx,
            partitions,
            receiver,
            ordered_index,
            worker_id,
            is_finished: false,
            location_preference: OrderedIndexLocationPreference::Unknown,
        })
    }
}

#[async_trait::async_trait]
impl AsyncSource for OrderedIndexSource {
    const NAME: &'static str = "OrderedIndexSource";

    #[async_backtrace::framed]
    async fn generate(&mut self) -> Result<Option<DataBlock>> {
        if self.is_finished {
            return Ok(None);
        }
        self.is_finished = true;

        let prefix = encode_prefix(&self.ordered_index)?;
        let filter = self.build_filter_expr(self.ordered_index.filters.as_ref())?;
        let candidate_read_start = Instant::now();
        let mut candidates = Vec::new();
        while let Some(parts) = self.fetch_parts().await? {
            if parts.is_empty() {
                continue;
            }

            let candidate_collect_start = Instant::now();
            candidates.append(
                &mut self
                    .load_candidate_blocks(parts, &prefix, filter.as_ref())
                    .await?,
            );
            record_elapsed(
                ProfileStatisticsName::OrderedIndexCandidateCollectTime,
                candidate_collect_start,
            );
        }

        let rows = self
            .read_candidate_rows(candidates, &prefix, filter.as_ref())
            .await?;
        record_elapsed(
            ProfileStatisticsName::OrderedIndexCandidateReadTime,
            candidate_read_start,
        );
        if rows.is_empty() {
            return Ok(None);
        }
        let decode_start = Instant::now();
        let mut decoded_rows = Vec::with_capacity(rows.len());
        for row in rows {
            decoded_rows.push(decode_ordered_payload_cached(&row.encoded_row_payload)?);
        }
        record_elapsed(
            ProfileStatisticsName::OrderedIndexPayloadDecodeTime,
            decode_start,
        );

        let output_build_start = Instant::now();
        let mut block = payload_rows_to_block(&self.ordered_index.payload_fields, decoded_rows)?;
        block = block.resort(&self.payload_schema, &self.output_schema)?;
        record_elapsed(
            ProfileStatisticsName::OrderedIndexOutputBuildTime,
            output_build_start,
        );
        Ok(Some(block))
    }
}

struct OrderedIndexCandidateBlock {
    index_location: String,
    meta: Arc<OrderedIndexFileMeta>,
    block_meta: OrderedIndexDataBlockMeta,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrderedIndexLocationPreference {
    Unknown,
    CurrentFirst,
    LegacyFirst,
}

struct OrderedIndexFilter {
    expr: Option<Expr<usize>>,
    payload_field_indexes: Vec<usize>,
    payload_fields: Vec<TableField>,
    key_predicates: Vec<OrderedIndexKeyPredicate>,
    fast_predicates: Option<Vec<OrderedIndexFastPredicate>>,
}

#[derive(Clone, Debug)]
struct OrderedIndexKeyPredicate {
    component_index: usize,
    order: OrderedIndexKeyOrder,
    kind: OrderedIndexKeyPredicateKind,
}

#[derive(Clone, Debug)]
enum OrderedIndexKeyPredicateKind {
    IsNull {
        null_component: Vec<u8>,
    },
    IsNotNull {
        null_component: Vec<u8>,
    },
    Compare {
        op: OrderedIndexFastCompareOp,
        constant_component: Vec<u8>,
        null_component: Vec<u8>,
    },
}

#[derive(Clone, Debug)]
struct OrderedIndexKeyField {
    component_index: usize,
    order: OrderedIndexKeyOrder,
    data_type: TableDataType,
}

#[derive(Clone, Debug)]
enum OrderedIndexFastPredicate {
    IsNull {
        column: usize,
    },
    IsNotNull {
        column: usize,
    },
    Compare {
        column: usize,
        op: OrderedIndexFastCompareOp,
        constant: Scalar,
    },
}

#[derive(Clone, Copy, Debug)]
enum OrderedIndexFastCompareOp {
    Eq,
    NotEq,
    Gt,
    Gte,
    Lt,
    Lte,
}

impl OrderedIndexSource {
    async fn fetch_parts(&self) -> Result<Option<Vec<PartInfoPtr>>> {
        if let Some(receiver) = &self.receiver {
            return match receiver.recv().await {
                Ok(Ok(part)) => Ok(Some(vec![part])),
                Ok(Err(err)) => Err(err),
                Err(_) => Ok(None),
            };
        }
        Ok(self
            .partitions
            .as_ref()
            .and_then(|partitions| partitions.steal(self.worker_id, 8)))
    }

    fn build_filter_expr(&self, filters: Option<&Filters>) -> Result<Option<OrderedIndexFilter>> {
        let Some(filters) = filters else {
            return Ok(None);
        };
        let expr = filters
            .filter
            .as_expr(&BUILTIN_FUNCTIONS)
            .project_column_ref(|name| self.payload_schema.index_of(name))?;
        let prefix_equalities = self.ordered_prefix_payload_equalities()?;
        let fast_predicates = compile_ordered_fast_predicates(&expr, &prefix_equalities);
        let key_fields = self.ordered_key_fields_by_payload_index()?;
        let (mut key_predicates, fast_predicates) = if let Some(predicates) = fast_predicates {
            split_ordered_fast_predicates(predicates, &key_fields)?
        } else {
            (Vec::new(), None)
        };
        key_predicates.sort_by_key(|predicate| predicate.component_index);
        let mut payload_field_indexes = if let Some(predicates) = &fast_predicates {
            predicates
                .iter()
                .map(|predicate| predicate.column())
                .collect()
        } else {
            expr.column_refs().keys().cloned().collect::<Vec<_>>()
        };
        payload_field_indexes.sort_unstable();
        payload_field_indexes.dedup();
        let payload_fields = payload_field_indexes
            .iter()
            .map(|index| {
                self.ordered_index
                    .payload_fields
                    .get(*index)
                    .cloned()
                    .ok_or_else(|| {
                        ErrorCode::StorageOther(format!(
                            "ordered filter references missing payload column {}, width {}",
                            index,
                            self.ordered_index.payload_fields.len()
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let fast_predicates = fast_predicates
            .map(|predicates| {
                predicates
                    .into_iter()
                    .map(|predicate| predicate.remap(&payload_field_indexes))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let expr = if fast_predicates.is_none() {
            Some(expr.project_column_ref(|index| {
                Ok(payload_field_indexes
                    .iter()
                    .position(|payload_index| payload_index == index)
                    .unwrap())
            })?)
        } else {
            None
        };
        Ok(Some(OrderedIndexFilter {
            expr,
            payload_field_indexes,
            payload_fields,
            key_predicates,
            fast_predicates,
        }))
    }

    fn ordered_prefix_payload_equalities(&self) -> Result<Vec<(usize, Scalar)>> {
        let mut equalities = Vec::with_capacity(self.ordered_index.equality_prefix.len());
        for (key_column, scalar) in self
            .ordered_index
            .key_columns
            .iter()
            .zip(self.ordered_index.equality_prefix.iter())
        {
            let payload_index = self.payload_schema.index_of(key_column.field.name())?;
            equalities.push((payload_index, scalar.clone()));
        }
        Ok(equalities)
    }

    fn ordered_key_fields_by_payload_index(&self) -> Result<Vec<Option<OrderedIndexKeyField>>> {
        let mut key_fields = vec![None; self.ordered_index.payload_fields.len()];
        for (component_index, key_column) in self.ordered_index.key_columns.iter().enumerate() {
            let payload_index = self.payload_schema.index_of(key_column.field.name())?;
            let order = match key_column.order {
                OrderedIndexColumnOrder::Asc => OrderedIndexKeyOrder::Asc,
                OrderedIndexColumnOrder::Desc => OrderedIndexKeyOrder::Desc,
            };
            key_fields[payload_index] = Some(OrderedIndexKeyField {
                component_index,
                order,
                data_type: key_column.field.data_type().clone(),
            });
        }
        Ok(key_fields)
    }

    fn filter_index_row_refs(
        &self,
        rows: &[&OrderedIndexRow],
        filter: &OrderedIndexFilter,
        limit: Option<usize>,
        key_predicate_start_component: usize,
        key_predicate_start_offset: usize,
    ) -> Result<Vec<OrderedIndexRow>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        let start = Instant::now();
        if let Some(predicates) = &filter.fast_predicates {
            let filtered_rows = if predicates.is_empty() {
                clone_matching_row_refs(rows, limit, |row| {
                    evaluate_ordered_key_predicates_from(
                        &filter.key_predicates,
                        &row.encoded_key,
                        key_predicate_start_component,
                        key_predicate_start_offset,
                    )
                })
            } else {
                let mut filtered_rows = Vec::new();
                for row in rows {
                    if !evaluate_ordered_key_predicates_from(
                        &filter.key_predicates,
                        &row.encoded_key,
                        key_predicate_start_component,
                        key_predicate_start_offset,
                    ) {
                        continue;
                    }
                    let payload_row = decode_ordered_payload_projection(
                        &row.encoded_row_payload,
                        &filter.payload_field_indexes,
                    )?;
                    if evaluate_ordered_fast_predicates(predicates, &payload_row) {
                        filtered_rows.push((*row).clone());
                        if limit.is_some_and(|limit| filtered_rows.len() >= limit) {
                            break;
                        }
                    }
                }
                filtered_rows
            };
            record_elapsed(ProfileStatisticsName::OrderedIndexFilterTime, start);
            return Ok(filtered_rows);
        }

        let payload_rows = rows
            .iter()
            .map(|row| {
                decode_ordered_payload_projection(
                    &row.encoded_row_payload,
                    &filter.payload_field_indexes,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let block = payload_rows_to_block(&filter.payload_fields, payload_rows)?;
        let evaluator = Evaluator::new(&block, &self.func_ctx, &BUILTIN_FUNCTIONS);
        let filter = evaluator
            .run(filter.expr.as_ref().ok_or_else(|| {
                ErrorCode::StorageOther("missing ordered fallback filter expression".to_string())
            })?)?
            .try_downcast::<BooleanType>()
            .unwrap();

        let filtered_rows = match filter {
            databend_common_expression::Value::Scalar(true) => clone_row_refs(rows, limit),
            databend_common_expression::Value::Scalar(false) => Vec::new(),
            databend_common_expression::Value::Column(bitmap) => {
                let mut filtered_rows = Vec::new();
                for (idx, row) in rows.iter().enumerate() {
                    if bitmap.get_bit(idx) {
                        filtered_rows.push((*row).clone());
                        if limit.is_some_and(|limit| filtered_rows.len() >= limit) {
                            break;
                        }
                    }
                }
                filtered_rows
            }
        };
        record_elapsed(ProfileStatisticsName::OrderedIndexFilterTime, start);
        Ok(filtered_rows)
    }

    async fn load_candidate_blocks(
        &mut self,
        parts: Vec<PartInfoPtr>,
        prefix: &[u8],
        filter: Option<&OrderedIndexFilter>,
    ) -> Result<Vec<OrderedIndexCandidateBlock>> {
        Profile::record_usize_profile(ProfileStatisticsName::OrderedIndexPartsScanned, parts.len());
        let key_predicates = filter
            .map(|filter| filter.key_predicates.clone())
            .unwrap_or_default();
        let prefix_component_count = self.ordered_index.equality_prefix.len();
        let mut candidates = Vec::new();
        let mut parts = parts.into_iter();
        if self.location_preference == OrderedIndexLocationPreference::Unknown
            && let Some(part) = parts.next()
        {
            let (mut part_candidates, preference) = Self::load_candidate_blocks_for_part(
                self.operator.clone(),
                part,
                self.ordered_index.index_name.clone(),
                self.ordered_index.index_version.clone(),
                self.ordered_index.use_block_ordered_index_size_hint,
                self.location_preference,
                prefix.to_vec(),
                prefix_component_count,
                key_predicates.clone(),
            )
            .await?;
            if self.location_preference != preference {
                self.location_preference = preference;
            }
            candidates.append(&mut part_candidates);
        }
        let futures = parts.map(|part| {
            Self::load_candidate_blocks_for_part(
                self.operator.clone(),
                part,
                self.ordered_index.index_name.clone(),
                self.ordered_index.index_version.clone(),
                self.ordered_index.use_block_ordered_index_size_hint,
                self.location_preference,
                prefix.to_vec(),
                prefix_component_count,
                key_predicates.clone(),
            )
        });
        let candidate_batches = future::try_join_all(futures).await?;
        for (mut part_candidates, preference) in candidate_batches {
            if self.location_preference != preference {
                self.location_preference = preference;
            }
            candidates.append(&mut part_candidates);
        }
        Profile::record_usize_profile(
            ProfileStatisticsName::OrderedIndexCandidateBlocks,
            candidates.len(),
        );
        Ok(candidates)
    }

    async fn load_candidate_blocks_for_part(
        operator: Operator,
        part: PartInfoPtr,
        index_name: String,
        index_version: String,
        use_block_ordered_index_size_hint: bool,
        location_preference: OrderedIndexLocationPreference,
        prefix: Vec<u8>,
        prefix_component_count: usize,
        key_predicates: Vec<OrderedIndexKeyPredicate>,
    ) -> Result<(
        Vec<OrderedIndexCandidateBlock>,
        OrderedIndexLocationPreference,
    )> {
        let fuse_part = FuseBlockPartInfo::from_part(&part)?;
        let index_location =
            TableMetaLocationGenerator::gen_ordered_index_location_from_block_location(
                &fuse_part.location,
                &index_name,
                &index_version,
            );
        let len_hint = use_block_ordered_index_size_hint
            .then_some(fuse_part.ordered_index_size)
            .flatten();
        let legacy_location =
            TableMetaLocationGenerator::gen_legacy_ordered_index_location_from_block_location(
                &fuse_part.location,
                &index_name,
                &index_version,
            );
        let (index_location, meta) = load_ordered_index_meta_with_legacy_fallback(
            operator.clone(),
            &index_location,
            &legacy_location,
            len_hint,
            location_preference,
        )
        .await?;
        let preference = if index_location == legacy_location {
            OrderedIndexLocationPreference::LegacyFirst
        } else {
            OrderedIndexLocationPreference::CurrentFirst
        };
        if !may_contain_equality_prefix_cached(&index_location, &meta, &prefix) {
            return Ok((Vec::new(), preference));
        }
        let candidates = meta
            .blocks_for_prefix(&prefix)
            .into_iter()
            .filter(|block_meta| {
                candidate_block_may_match_key_predicates(
                    block_meta,
                    &prefix,
                    prefix_component_count,
                    &key_predicates,
                )
            })
            .map(|block_meta| OrderedIndexCandidateBlock {
                index_location: index_location.clone(),
                meta: meta.clone(),
                block_meta,
            })
            .collect();
        Ok((candidates, preference))
    }

    async fn read_candidate_rows(
        &self,
        mut candidates: Vec<OrderedIndexCandidateBlock>,
        prefix: &[u8],
        filter: Option<&OrderedIndexFilter>,
    ) -> Result<Vec<OrderedIndexRow>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let sort_start = Instant::now();
        candidates.sort_by(|left, right| {
            left.block_meta
                .first_key
                .cmp(&right.block_meta.first_key)
                .then_with(|| left.block_meta.last_key.cmp(&right.block_meta.last_key))
                .then_with(|| left.index_location.cmp(&right.index_location))
                .then_with(|| left.block_meta.offset.cmp(&right.block_meta.offset))
        });
        record_elapsed(
            ProfileStatisticsName::OrderedIndexCandidateSortTime,
            sort_start,
        );

        if candidate_blocks_are_disjoint(&candidates) {
            self.read_candidate_rows_in_order(candidates, prefix, filter)
                .await
        } else {
            self.read_candidate_rows_with_sort(candidates, prefix, filter)
                .await
        }
    }

    async fn read_candidate_rows_in_order(
        &self,
        candidates: Vec<OrderedIndexCandidateBlock>,
        prefix: &[u8],
        filter: Option<&OrderedIndexFilter>,
    ) -> Result<Vec<OrderedIndexRow>> {
        let mut rows: Vec<OrderedIndexRow> = Vec::new();
        for candidate in candidates {
            let remaining = self
                .ordered_index
                .limit
                .map(|limit| limit.saturating_sub(rows.len()));
            if remaining == Some(0) {
                return Ok(rows);
            }
            let mut block_rows = self
                .read_candidate_block(&candidate, prefix, filter, remaining, None)
                .await?;
            rows.append(&mut block_rows);
            if self
                .ordered_index
                .limit
                .is_some_and(|limit| rows.len() >= limit)
            {
                rows.truncate(self.ordered_index.limit.unwrap());
                return Ok(rows);
            }
        }
        Ok(rows)
    }

    async fn read_candidate_rows_with_sort(
        &self,
        candidates: Vec<OrderedIndexCandidateBlock>,
        prefix: &[u8],
        filter: Option<&OrderedIndexFilter>,
    ) -> Result<Vec<OrderedIndexRow>> {
        let Some(limit) = self.ordered_index.limit else {
            let mut rows = Vec::new();
            for batch in candidates.chunks(ORDERED_INDEX_DATA_BLOCK_READ_BATCH_SIZE) {
                for mut block_rows in self
                    .read_candidate_blocks_batch(batch, prefix, filter, None, None)
                    .await?
                {
                    rows.append(&mut block_rows);
                }
            }
            rows.sort_by(|left, right| left.encoded_key.cmp(&right.encoded_key));
            return Ok(rows);
        };

        let mut rows: Vec<OrderedIndexRow> = Vec::new();
        let mut candidate_offset = 0;
        while candidate_offset < candidates.len() {
            let batch_end = candidate_offset + 1;
            let upper_bound_key = if rows.len() == limit {
                Some(rows[limit - 1].encoded_key.as_slice())
            } else {
                None
            };
            for mut block_rows in self
                .read_candidate_blocks_batch(
                    &candidates[candidate_offset..batch_end],
                    prefix,
                    filter,
                    Some(limit),
                    upper_bound_key,
                )
                .await?
            {
                rows.append(&mut block_rows);
            }
            candidate_offset = batch_end;

            if rows.len() >= limit {
                sort_and_truncate_rows(&mut rows, limit);
                if candidates.get(candidate_offset).is_none_or(|next| {
                    next.block_meta.first_key.as_slice()
                        >= rows[rows.len() - 1].encoded_key.as_slice()
                }) {
                    return Ok(rows);
                }
            }
        }
        sort_and_truncate_rows(&mut rows, limit);
        Ok(rows)
    }

    async fn read_candidate_blocks_batch(
        &self,
        candidates: &[OrderedIndexCandidateBlock],
        prefix: &[u8],
        filter: Option<&OrderedIndexFilter>,
        limit: Option<usize>,
        upper_bound_key: Option<&[u8]>,
    ) -> Result<Vec<Vec<OrderedIndexRow>>> {
        let futures = candidates.iter().map(|candidate| {
            self.read_candidate_block(candidate, prefix, filter, limit, upper_bound_key)
        });
        future::try_join_all(futures).await
    }

    async fn read_candidate_block(
        &self,
        candidate: &OrderedIndexCandidateBlock,
        prefix: &[u8],
        filter: Option<&OrderedIndexFilter>,
        limit: Option<usize>,
        upper_bound_key: Option<&[u8]>,
    ) -> Result<Vec<OrderedIndexRow>> {
        let load_start = Instant::now();
        let rows = load_ordered_index_data_block(
            self.operator.clone(),
            &candidate.index_location,
            &candidate.meta,
            &candidate.block_meta,
        )
        .await?;
        record_elapsed(
            ProfileStatisticsName::OrderedIndexDataBlockLoadTime,
            load_start,
        );
        Profile::record_usize_profile(ProfileStatisticsName::OrderedIndexRowsDecoded, rows.len());
        let row_select_start = Instant::now();
        let mut rows = ordered_prefix_row_refs(&rows, prefix);
        truncate_row_refs_before_key(&mut rows, upper_bound_key);
        let rows = if let Some(filter) = filter {
            let (key_predicate_start_component, key_predicate_start_offset) =
                ordered_key_predicate_scan_start(
                    filter,
                    prefix,
                    self.ordered_index.equality_prefix.len(),
                );
            self.filter_index_row_refs(
                &rows,
                filter,
                limit,
                key_predicate_start_component,
                key_predicate_start_offset,
            )?
        } else {
            clone_row_refs(&rows, limit)
        };
        record_elapsed(
            ProfileStatisticsName::OrderedIndexRowSelectTime,
            row_select_start,
        );
        Profile::record_usize_profile(ProfileStatisticsName::OrderedIndexRowsMatched, rows.len());
        Ok(rows)
    }
}

async fn load_ordered_index_meta_with_legacy_fallback(
    operator: Operator,
    index_location: &str,
    legacy_location: &str,
    len_hint: Option<u64>,
    preference: OrderedIndexLocationPreference,
) -> Result<(String, Arc<OrderedIndexFileMeta>)> {
    if let Some(resolved) = ORDERED_RESOLVED_INDEX_LOCATION_CACHE.get(index_location) {
        match load_ordered_index_meta(operator.clone(), &resolved.location, len_hint).await {
            Ok(meta) => return Ok((resolved.location.clone(), meta)),
            Err(err) if err.code() == ErrorCode::STORAGE_NOT_FOUND => {
                ORDERED_RESOLVED_INDEX_LOCATION_CACHE.evict(index_location);
            }
            Err(err) => return Err(err),
        }
    }

    if preference == OrderedIndexLocationPreference::LegacyFirst {
        match load_ordered_index_meta(operator.clone(), legacy_location, len_hint).await {
            Ok(meta) => {
                ORDERED_RESOLVED_INDEX_LOCATION_CACHE.insert(
                    index_location.to_string(),
                    OrderedIndexResolvedLocation {
                        location: legacy_location.to_string(),
                    },
                );
                return Ok((legacy_location.to_string(), meta));
            }
            Err(err) if err.code() == ErrorCode::STORAGE_NOT_FOUND => {}
            Err(err) => return Err(err),
        }
    }

    match load_ordered_index_meta(operator.clone(), index_location, len_hint).await {
        Ok(meta) => {
            ORDERED_RESOLVED_INDEX_LOCATION_CACHE.insert(
                index_location.to_string(),
                OrderedIndexResolvedLocation {
                    location: index_location.to_string(),
                },
            );
            Ok((index_location.to_string(), meta))
        }
        Err(err) if err.code() == ErrorCode::STORAGE_NOT_FOUND => {
            let meta = load_ordered_index_meta(operator, legacy_location, len_hint).await?;
            ORDERED_RESOLVED_INDEX_LOCATION_CACHE.insert(
                index_location.to_string(),
                OrderedIndexResolvedLocation {
                    location: legacy_location.to_string(),
                },
            );
            Ok((legacy_location.to_string(), meta))
        }
        Err(err) => Err(err),
    }
}

impl OrderedIndexFastPredicate {
    fn column(&self) -> usize {
        match self {
            OrderedIndexFastPredicate::IsNull { column }
            | OrderedIndexFastPredicate::IsNotNull { column }
            | OrderedIndexFastPredicate::Compare { column, .. } => *column,
        }
    }

    fn remap(self, payload_field_indexes: &[usize]) -> Result<Self> {
        let remap_column = |column| {
            payload_field_indexes
                .iter()
                .position(|payload_index| *payload_index == column)
                .ok_or_else(|| {
                    ErrorCode::StorageOther(format!(
                        "ordered fast filter references missing payload column {}",
                        column
                    ))
                })
        };
        Ok(match self {
            OrderedIndexFastPredicate::IsNull { column } => OrderedIndexFastPredicate::IsNull {
                column: remap_column(column)?,
            },
            OrderedIndexFastPredicate::IsNotNull { column } => {
                OrderedIndexFastPredicate::IsNotNull {
                    column: remap_column(column)?,
                }
            }
            OrderedIndexFastPredicate::Compare {
                column,
                op,
                constant,
            } => OrderedIndexFastPredicate::Compare {
                column: remap_column(column)?,
                op,
                constant,
            },
        })
    }
}

fn split_ordered_fast_predicates(
    predicates: Vec<OrderedIndexFastPredicate>,
    key_fields: &[Option<OrderedIndexKeyField>],
) -> Result<(
    Vec<OrderedIndexKeyPredicate>,
    Option<Vec<OrderedIndexFastPredicate>>,
)> {
    let mut key_predicates = Vec::new();
    let mut payload_predicates = Vec::new();
    for predicate in predicates {
        let Some(Some(key_field)) = key_fields.get(predicate.column()) else {
            payload_predicates.push(predicate);
            continue;
        };
        if let Some(key_predicate) = predicate.clone().try_into_key_predicate(key_field)? {
            key_predicates.push(key_predicate);
        } else {
            payload_predicates.push(predicate);
        }
    }
    Ok((key_predicates, Some(payload_predicates)))
}

impl OrderedIndexFastPredicate {
    fn try_into_key_predicate(
        self,
        key_field: &OrderedIndexKeyField,
    ) -> Result<Option<OrderedIndexKeyPredicate>> {
        let null_component = encode_key_component_value(&Scalar::Null, key_field.order)?;
        let kind = match self {
            OrderedIndexFastPredicate::IsNull { .. } => {
                OrderedIndexKeyPredicateKind::IsNull { null_component }
            }
            OrderedIndexFastPredicate::IsNotNull { .. } => {
                OrderedIndexKeyPredicateKind::IsNotNull { null_component }
            }
            OrderedIndexFastPredicate::Compare { op, constant, .. } => {
                let Some(constant) = normalize_key_filter_constant(&constant, &key_field.data_type)
                else {
                    return Ok(None);
                };
                let constant_component = encode_key_component_value(&constant, key_field.order)?;
                OrderedIndexKeyPredicateKind::Compare {
                    op,
                    constant_component,
                    null_component,
                }
            }
        };
        Ok(Some(OrderedIndexKeyPredicate {
            component_index: key_field.component_index,
            order: key_field.order,
            kind,
        }))
    }
}

fn normalize_key_filter_constant(constant: &Scalar, data_type: &TableDataType) -> Option<Scalar> {
    if matches!(constant, Scalar::Null) {
        return Some(Scalar::Null);
    }

    match data_type.remove_nullable() {
        TableDataType::Number(number_type) => normalize_number_key_constant(constant, number_type),
        TableDataType::Decimal(decimal_type) => {
            normalize_decimal_key_constant(constant, decimal_type)
        }
        TableDataType::String => match constant {
            Scalar::String(_) => Some(constant.clone()),
            _ => None,
        },
        TableDataType::Boolean => match constant {
            Scalar::Boolean(_) => Some(constant.clone()),
            _ => None,
        },
        TableDataType::Date => match constant {
            Scalar::Date(_) => Some(constant.clone()),
            _ => None,
        },
        TableDataType::Timestamp => match constant {
            Scalar::Timestamp(_) => Some(constant.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn normalize_number_key_constant(constant: &Scalar, number_type: NumberDataType) -> Option<Scalar> {
    let Scalar::Number(number) = constant else {
        return None;
    };
    let value = number.integer_to_i128()?;
    let number = match number_type {
        NumberDataType::UInt8 => NumberScalar::UInt8(u8::try_from(value).ok()?),
        NumberDataType::UInt16 => NumberScalar::UInt16(u16::try_from(value).ok()?),
        NumberDataType::UInt32 => NumberScalar::UInt32(u32::try_from(value).ok()?),
        NumberDataType::UInt64 => NumberScalar::UInt64(u64::try_from(value).ok()?),
        NumberDataType::Int8 => NumberScalar::Int8(i8::try_from(value).ok()?),
        NumberDataType::Int16 => NumberScalar::Int16(i16::try_from(value).ok()?),
        NumberDataType::Int32 => NumberScalar::Int32(i32::try_from(value).ok()?),
        NumberDataType::Int64 => NumberScalar::Int64(i64::try_from(value).ok()?),
        NumberDataType::Float32 | NumberDataType::Float64 => return None,
    };
    Some(Scalar::Number(number))
}

fn normalize_decimal_key_constant(
    constant: &Scalar,
    decimal_type: DecimalDataType,
) -> Option<Scalar> {
    let target_size = decimal_type.size();
    let value = match constant {
        Scalar::Decimal(decimal) => {
            let source_size = decimal.size();
            rescale_decimal_i256(
                decimal.as_decimal::<i256>(),
                source_size.scale(),
                target_size.scale(),
            )?
        }
        Scalar::Number(number) if number.is_integer() => {
            let value = i256::from(number.integer_to_i128()?);
            value.checked_mul(i256::e(target_size.scale()))?
        }
        _ => return None,
    };

    decimal_scalar_from_i256(value, decimal_type).map(Scalar::Decimal)
}

fn rescale_decimal_i256(value: i256, source_scale: u8, target_scale: u8) -> Option<i256> {
    if source_scale == target_scale {
        return Some(value);
    }

    let diff = target_scale.abs_diff(source_scale);
    if target_scale > source_scale {
        value.checked_mul(i256::e(diff))
    } else {
        value.checked_div(i256::e(diff))
    }
}

fn decimal_scalar_from_i256(value: i256, decimal_type: DecimalDataType) -> Option<DecimalScalar> {
    match decimal_type {
        DecimalDataType::Decimal64(size) => {
            if value < i256::from(i64::min_for_precision(size.precision()))
                || value > i256::from(i64::max_for_precision(size.precision()))
            {
                return None;
            }
            Some(DecimalScalar::Decimal64(value.as_i64(), size))
        }
        DecimalDataType::Decimal128(size) => {
            if value < i256::from(i128::min_for_precision(size.precision()))
                || value > i256::from(i128::max_for_precision(size.precision()))
            {
                return None;
            }
            Some(DecimalScalar::Decimal128(value.as_i128(), size))
        }
        DecimalDataType::Decimal256(size) => {
            if value < i256::min_for_precision(size.precision())
                || value > i256::max_for_precision(size.precision())
            {
                return None;
            }
            Some(DecimalScalar::Decimal256(value, size))
        }
    }
}

fn record_elapsed(name: ProfileStatisticsName, start: Instant) {
    Profile::record_usize_profile(name, start.elapsed().as_nanos() as usize);
}

fn may_contain_equality_prefix_cached(
    index_location: &str,
    meta: &OrderedIndexFileMeta,
    prefix: &[u8],
) -> bool {
    let cache_key = format!("{}#equality_prefix_bloom", index_location);
    if let Some(cached) = ORDERED_DECODED_EQUALITY_PREFIX_BLOOM_CACHE.get(&cache_key) {
        return cached
            .filter
            .as_ref()
            .is_none_or(|filter| filter.contains(prefix));
    }

    let filter = BloomFilter::from_bytes(&meta.filter_block.equality_prefix_bloom)
        .ok()
        .map(|(filter, _)| Arc::new(filter));
    let may_contain = filter.as_ref().is_none_or(|filter| filter.contains(prefix));
    ORDERED_DECODED_EQUALITY_PREFIX_BLOOM_CACHE.insert(
        cache_key,
        OrderedIndexDecodedEqualityPrefixBloom {
            filter,
            mem_bytes: meta.filter_block.equality_prefix_bloom.len(),
        },
    );
    may_contain
}

fn ordered_prefix_row_refs<'a>(
    rows: &'a [OrderedIndexRow],
    prefix: &[u8],
) -> Vec<&'a OrderedIndexRow> {
    if prefix.is_empty() {
        return rows.iter().collect();
    }

    let start = rows.partition_point(|row| row.encoded_key.as_slice() < prefix);
    let mut end = start;
    while end < rows.len() && rows[end].encoded_key.starts_with(prefix) {
        end += 1;
    }
    rows[start..end].iter().collect()
}

fn truncate_row_refs_before_key(rows: &mut Vec<&OrderedIndexRow>, upper_bound_key: Option<&[u8]>) {
    let Some(upper_bound_key) = upper_bound_key else {
        return;
    };

    let end = rows.partition_point(|row| row.encoded_key.as_slice() < upper_bound_key);
    rows.truncate(end);
}

fn ordered_key_predicate_scan_start(
    filter: &OrderedIndexFilter,
    prefix: &[u8],
    prefix_component_count: usize,
) -> (usize, usize) {
    if filter
        .key_predicates
        .iter()
        .all(|predicate| predicate.component_index >= prefix_component_count)
    {
        (prefix_component_count, prefix.len())
    } else {
        (0, 0)
    }
}

fn clone_row_refs(rows: &[&OrderedIndexRow], limit: Option<usize>) -> Vec<OrderedIndexRow> {
    rows.iter()
        .take(limit.unwrap_or(usize::MAX))
        .map(|row| (*row).clone())
        .collect()
}

fn clone_matching_row_refs(
    rows: &[&OrderedIndexRow],
    limit: Option<usize>,
    mut predicate: impl FnMut(&OrderedIndexRow) -> bool,
) -> Vec<OrderedIndexRow> {
    let mut result = Vec::new();
    for row in rows {
        if predicate(row) {
            result.push((*row).clone());
            if limit.is_some_and(|limit| result.len() >= limit) {
                break;
            }
        }
    }
    result
}

fn sort_and_truncate_rows(rows: &mut Vec<OrderedIndexRow>, limit: usize) {
    rows.sort_by(|left, right| left.encoded_key.cmp(&right.encoded_key));
    rows.truncate(limit);
}

fn candidate_blocks_are_disjoint(candidates: &[OrderedIndexCandidateBlock]) -> bool {
    candidates
        .windows(2)
        .all(|blocks| blocks[0].block_meta.last_key < blocks[1].block_meta.first_key)
}

fn candidate_block_may_match_key_predicates(
    block_meta: &OrderedIndexDataBlockMeta,
    prefix: &[u8],
    prefix_component_count: usize,
    key_predicates: &[OrderedIndexKeyPredicate],
) -> bool {
    if key_predicates.is_empty()
        || !block_meta.first_key.starts_with(prefix)
        || !block_meta.last_key.starts_with(prefix)
    {
        return true;
    }

    let mut constant_prefix_component_count = prefix_component_count;
    let mut predicate_index = 0;
    while predicate_index < key_predicates.len() {
        let component_index = key_predicates[predicate_index].component_index;
        if component_index > constant_prefix_component_count {
            if !candidate_block_has_constant_prefix_before_component(
                block_meta,
                constant_prefix_component_count,
                component_index,
            ) {
                return true;
            }
            constant_prefix_component_count = component_index;
        }

        let Some(first_component) = encoded_key_component(&block_meta.first_key, component_index)
        else {
            return true;
        };
        let Some(last_component) = encoded_key_component(&block_meta.last_key, component_index)
        else {
            return true;
        };
        while key_predicates
            .get(predicate_index)
            .is_some_and(|predicate| predicate.component_index == component_index)
        {
            if !key_component_range_may_match_predicate(
                &key_predicates[predicate_index],
                first_component,
                last_component,
            ) {
                return false;
            }
            predicate_index += 1;
        }

        if first_component == last_component {
            constant_prefix_component_count =
                constant_prefix_component_count.max(component_index + 1);
        } else if key_predicates
            .get(predicate_index)
            .is_some_and(|predicate| predicate.component_index > component_index)
        {
            return true;
        }
    }
    true
}

fn candidate_block_has_constant_prefix_before_component(
    block_meta: &OrderedIndexDataBlockMeta,
    known_constant_prefix_component_count: usize,
    component_index: usize,
) -> bool {
    if component_index == known_constant_prefix_component_count {
        return true;
    }
    if component_index < known_constant_prefix_component_count {
        return false;
    }

    let Ok(first_prefix) = ordered_equality_prefix(&block_meta.first_key, component_index) else {
        return false;
    };
    let Ok(last_prefix) = ordered_equality_prefix(&block_meta.last_key, component_index) else {
        return false;
    };
    first_prefix == last_prefix
}

fn key_component_range_may_match_predicate(
    predicate: &OrderedIndexKeyPredicate,
    first_component: &[u8],
    last_component: &[u8],
) -> bool {
    if first_component > last_component {
        return true;
    }

    match &predicate.kind {
        OrderedIndexKeyPredicateKind::IsNull { null_component } => {
            component_in_range(null_component, first_component, last_component)
        }
        OrderedIndexKeyPredicateKind::IsNotNull { null_component } => {
            !(first_component == null_component.as_slice()
                && last_component == null_component.as_slice())
        }
        OrderedIndexKeyPredicateKind::Compare {
            op,
            constant_component,
            null_component,
        } => {
            if first_component == null_component.as_slice()
                && last_component == null_component.as_slice()
            {
                return false;
            }
            match op {
                OrderedIndexFastCompareOp::Eq => {
                    component_in_range(constant_component, first_component, last_component)
                }
                OrderedIndexFastCompareOp::NotEq => {
                    first_component != constant_component.as_slice()
                        || last_component != constant_component.as_slice()
                }
                OrderedIndexFastCompareOp::Gt => match predicate.order {
                    OrderedIndexKeyOrder::Asc => last_component > constant_component.as_slice(),
                    OrderedIndexKeyOrder::Desc => first_component < constant_component.as_slice(),
                },
                OrderedIndexFastCompareOp::Gte => match predicate.order {
                    OrderedIndexKeyOrder::Asc => last_component >= constant_component.as_slice(),
                    OrderedIndexKeyOrder::Desc => first_component <= constant_component.as_slice(),
                },
                OrderedIndexFastCompareOp::Lt => match predicate.order {
                    OrderedIndexKeyOrder::Asc => first_component < constant_component.as_slice(),
                    OrderedIndexKeyOrder::Desc => last_component > constant_component.as_slice(),
                },
                OrderedIndexFastCompareOp::Lte => match predicate.order {
                    OrderedIndexKeyOrder::Asc => first_component <= constant_component.as_slice(),
                    OrderedIndexKeyOrder::Desc => last_component >= constant_component.as_slice(),
                },
            }
        }
    }
}

fn component_in_range(component: &[u8], first_component: &[u8], last_component: &[u8]) -> bool {
    first_component <= component && component <= last_component
}

fn decode_ordered_payload_cached(payload: &[u8]) -> Result<Vec<Scalar>> {
    let cache_key = decoded_payload_cache_key(payload);
    if let Some(cached) = ORDERED_DECODED_PAYLOAD_ROW_CACHE.get(&cache_key) {
        return Ok(cached.row.clone());
    }

    let row = decode_ordered_payload(payload)?;
    ORDERED_DECODED_PAYLOAD_ROW_CACHE.insert(cache_key, OrderedIndexDecodedPayloadRow {
        row: row.clone(),
    });
    Ok(row)
}

fn decoded_payload_cache_key(payload: &[u8]) -> String {
    let digest = sha2::Sha256::digest(payload);
    format!("{}:{digest:x}", payload.len())
}

fn scalar_rows_mem_bytes(row: &[Scalar]) -> usize {
    row.iter().map(scalar_mem_bytes).sum::<usize>() + std::mem::size_of_val(row)
}

fn scalar_mem_bytes(scalar: &Scalar) -> usize {
    std::mem::size_of::<Scalar>()
        + match scalar {
            Scalar::String(value) => value.len(),
            Scalar::Binary(value)
            | Scalar::Bitmap(value)
            | Scalar::Variant(value)
            | Scalar::Geometry(value) => value.len(),
            Scalar::Tuple(values) => values.iter().map(scalar_mem_bytes).sum(),
            _ => 0,
        }
}

fn compile_ordered_fast_predicates(
    expr: &Expr<usize>,
    prefix_equalities: &[(usize, Scalar)],
) -> Option<Vec<OrderedIndexFastPredicate>> {
    let mut predicates = Vec::new();
    collect_ordered_fast_predicates(expr, prefix_equalities, &mut predicates)?;
    Some(predicates)
}

fn collect_ordered_fast_predicates(
    expr: &Expr<usize>,
    prefix_equalities: &[(usize, Scalar)],
    predicates: &mut Vec<OrderedIndexFastPredicate>,
) -> Option<()> {
    let Expr::FunctionCall(function) = expr else {
        return match expr {
            Expr::Constant(constant) => match &constant.scalar {
                Scalar::Boolean(true) => Some(()),
                _ => None,
            },
            _ => None,
        };
    };

    let name = function.id.name();
    match name.as_ref() {
        "and" | "and_filters" => {
            for arg in &function.args {
                collect_ordered_fast_predicates(arg, prefix_equalities, predicates)?;
            }
            Some(())
        }
        "not" if function.args.len() == 1 => {
            if let Some(column) = extract_is_not_null_column(&function.args[0]) {
                if prefix_equality_scalar(prefix_equalities, column)
                    .is_some_and(|scalar| matches!(scalar, Scalar::Null))
                {
                    return Some(());
                }
                predicates.push(OrderedIndexFastPredicate::IsNull { column });
                Some(())
            } else {
                None
            }
        }
        "is_not_null" if function.args.len() == 1 => {
            let Expr::ColumnRef(column) = &function.args[0] else {
                return None;
            };
            if prefix_equality_scalar(prefix_equalities, column.id)
                .is_some_and(|scalar| !matches!(scalar, Scalar::Null))
            {
                return Some(());
            }
            predicates.push(OrderedIndexFastPredicate::IsNotNull { column: column.id });
            Some(())
        }
        "eq" | "noteq" | "gt" | "gte" | "lt" | "lte" if function.args.len() == 2 => {
            let (column, constant, op) = extract_fast_compare(&function.args, name.as_ref())?;
            if matches!(op, OrderedIndexFastCompareOp::Eq)
                && prefix_equality_scalar(prefix_equalities, column)
                    .is_some_and(|prefix_scalar| prefix_scalar == &constant)
            {
                return Some(());
            }
            predicates.push(OrderedIndexFastPredicate::Compare {
                column,
                op,
                constant,
            });
            Some(())
        }
        _ => None,
    }
}

fn prefix_equality_scalar(prefix_equalities: &[(usize, Scalar)], column: usize) -> Option<&Scalar> {
    prefix_equalities
        .iter()
        .find_map(|(prefix_column, scalar)| (*prefix_column == column).then_some(scalar))
}

fn extract_is_not_null_column(expr: &Expr<usize>) -> Option<usize> {
    let Expr::FunctionCall(function) = expr else {
        return None;
    };
    if function.id.name().as_ref() != "is_not_null" || function.args.len() != 1 {
        return None;
    }
    let Expr::ColumnRef(column) = &function.args[0] else {
        return None;
    };
    Some(column.id)
}

fn extract_fast_compare(
    args: &[Expr<usize>],
    op_name: &str,
) -> Option<(usize, Scalar, OrderedIndexFastCompareOp)> {
    match (&args[0], &args[1]) {
        (Expr::ColumnRef(column), Expr::Constant(constant)) => Some((
            column.id,
            constant.scalar.clone(),
            fast_compare_op(op_name, false)?,
        )),
        (Expr::Constant(constant), Expr::ColumnRef(column)) => Some((
            column.id,
            constant.scalar.clone(),
            fast_compare_op(op_name, true)?,
        )),
        _ => None,
    }
}

fn fast_compare_op(op_name: &str, reverse: bool) -> Option<OrderedIndexFastCompareOp> {
    let op = match op_name {
        "eq" => OrderedIndexFastCompareOp::Eq,
        "noteq" => OrderedIndexFastCompareOp::NotEq,
        "gt" if reverse => OrderedIndexFastCompareOp::Lt,
        "gt" => OrderedIndexFastCompareOp::Gt,
        "gte" if reverse => OrderedIndexFastCompareOp::Lte,
        "gte" => OrderedIndexFastCompareOp::Gte,
        "lt" if reverse => OrderedIndexFastCompareOp::Gt,
        "lt" => OrderedIndexFastCompareOp::Lt,
        "lte" if reverse => OrderedIndexFastCompareOp::Gte,
        "lte" => OrderedIndexFastCompareOp::Lte,
        _ => return None,
    };
    Some(op)
}

fn evaluate_ordered_fast_predicates(
    predicates: &[OrderedIndexFastPredicate],
    payload_row: &[Scalar],
) -> bool {
    predicates
        .iter()
        .all(|predicate| evaluate_ordered_fast_predicate(predicate, payload_row))
}

#[cfg(test)]
fn evaluate_ordered_key_predicates(
    predicates: &[OrderedIndexKeyPredicate],
    encoded_key: &[u8],
) -> bool {
    evaluate_ordered_key_predicates_from(predicates, encoded_key, 0, 0)
}

fn evaluate_ordered_key_predicates_from(
    predicates: &[OrderedIndexKeyPredicate],
    encoded_key: &[u8],
    mut component_index: usize,
    mut offset: usize,
) -> bool {
    if predicates.is_empty() {
        return true;
    }

    let mut predicate_index = 0usize;
    if predicates[predicate_index].component_index < component_index {
        component_index = 0;
        offset = 0;
    }
    while predicate_index < predicates.len() {
        let len_bytes = match encoded_key.get(offset..offset + 4) {
            Some(len_bytes) => len_bytes,
            None => return false,
        };
        let len = match <[u8; 4]>::try_from(len_bytes) {
            Ok(len_bytes) => u32::from_be_bytes(len_bytes) as usize,
            Err(_) => return false,
        };
        let value_start = offset + 4;
        let Some(value_end) = value_start.checked_add(len) else {
            return false;
        };
        let Some(component) = encoded_key.get(value_start..value_end) else {
            return false;
        };
        let Some(separator) = encoded_key.get(value_end) else {
            return false;
        };
        if *separator != 0xff {
            return false;
        }

        while predicates
            .get(predicate_index)
            .is_some_and(|predicate| predicate.component_index == component_index)
        {
            if !evaluate_ordered_key_predicate_component(&predicates[predicate_index], component) {
                return false;
            }
            predicate_index += 1;
        }

        offset = value_end + 1;
        component_index += 1;
        if predicates
            .get(predicate_index)
            .is_some_and(|predicate| predicate.component_index < component_index)
        {
            return false;
        }
    }
    true
}

fn evaluate_ordered_key_predicate_component(
    predicate: &OrderedIndexKeyPredicate,
    component: &[u8],
) -> bool {
    match &predicate.kind {
        OrderedIndexKeyPredicateKind::IsNull { null_component } => component == null_component,
        OrderedIndexKeyPredicateKind::IsNotNull { null_component } => component != null_component,
        OrderedIndexKeyPredicateKind::Compare {
            op,
            constant_component,
            null_component,
        } => {
            if component == null_component || constant_component == null_component {
                return false;
            }
            let ordering = match predicate.order {
                OrderedIndexKeyOrder::Asc => component.cmp(constant_component.as_slice()),
                OrderedIndexKeyOrder::Desc => {
                    component.cmp(constant_component.as_slice()).reverse()
                }
            };
            evaluate_ordered_ordering(*op, ordering)
        }
    }
}

fn evaluate_ordered_fast_predicate(
    predicate: &OrderedIndexFastPredicate,
    payload_row: &[Scalar],
) -> bool {
    match predicate {
        OrderedIndexFastPredicate::IsNull { column } => payload_row
            .get(*column)
            .is_some_and(|value| matches!(value, Scalar::Null)),
        OrderedIndexFastPredicate::IsNotNull { column } => payload_row
            .get(*column)
            .is_some_and(|value| !matches!(value, Scalar::Null)),
        OrderedIndexFastPredicate::Compare {
            column,
            op,
            constant,
        } => {
            let Some(value) = payload_row.get(*column) else {
                return false;
            };
            if matches!(value, Scalar::Null) || matches!(constant, Scalar::Null) {
                return false;
            }
            let Some(ordering) = compare_ordered_filter_scalars(value, constant) else {
                return false;
            };
            evaluate_ordered_ordering(*op, ordering)
        }
    }
}

fn evaluate_ordered_ordering(op: OrderedIndexFastCompareOp, ordering: Ordering) -> bool {
    match op {
        OrderedIndexFastCompareOp::Eq => ordering == Ordering::Equal,
        OrderedIndexFastCompareOp::NotEq => ordering != Ordering::Equal,
        OrderedIndexFastCompareOp::Gt => ordering == Ordering::Greater,
        OrderedIndexFastCompareOp::Gte => matches!(ordering, Ordering::Greater | Ordering::Equal),
        OrderedIndexFastCompareOp::Lt => ordering == Ordering::Less,
        OrderedIndexFastCompareOp::Lte => matches!(ordering, Ordering::Less | Ordering::Equal),
    }
}

fn compare_ordered_filter_scalars(left: &Scalar, right: &Scalar) -> Option<Ordering> {
    if let Some(ordering) = left.partial_cmp(right) {
        return Some(ordering);
    }

    match (left, right) {
        (Scalar::Number(left), Scalar::Number(right)) => {
            if left.is_integer()
                && right.is_integer()
                && let (Some(left), Some(right)) = (left.integer_to_i128(), right.integer_to_i128())
            {
                return Some(left.cmp(&right));
            }
            Some(left.to_f64().cmp(&right.to_f64()))
        }
        (Scalar::Decimal(left), Scalar::Decimal(right)) => {
            left.to_float64().partial_cmp(&right.to_float64())
        }
        (Scalar::Decimal(left), Scalar::Number(right)) => {
            left.to_float64().partial_cmp(&right.to_f64().into_inner())
        }
        (Scalar::Number(left), Scalar::Decimal(right)) => {
            left.to_f64().into_inner().partial_cmp(&right.to_float64())
        }
        _ => None,
    }
}

fn encode_key_component_value(scalar: &Scalar, order: OrderedIndexKeyOrder) -> Result<Vec<u8>> {
    let mut key = Vec::new();
    encode_ordered_key_component(&mut key, scalar.as_ref(), order)?;
    let Some(component) = encoded_key_component(&key, 0) else {
        return Err(ErrorCode::StorageOther(
            "failed to encode ordered key component".to_string(),
        ));
    };
    Ok(component.to_vec())
}

fn encoded_key_component(encoded_key: &[u8], component_index: usize) -> Option<&[u8]> {
    let mut offset = 0usize;
    for current_index in 0..=component_index {
        let len_bytes = encoded_key.get(offset..offset + 4)?;
        let len = u32::from_be_bytes(len_bytes.try_into().ok()?) as usize;
        let value_start = offset + 4;
        let value_end = value_start.checked_add(len)?;
        let component = encoded_key.get(value_start..value_end)?;
        let separator = encoded_key.get(value_end)?;
        if *separator != 0xff {
            return None;
        }
        if current_index == component_index {
            return Some(component);
        }
        offset = value_end + 1;
    }
    None
}

fn encode_prefix(ordered_index: &OrderedIndexInfo) -> Result<Vec<u8>> {
    let mut key = Vec::new();
    for (scalar, key_column) in ordered_index
        .equality_prefix
        .iter()
        .zip(ordered_index.key_columns.iter())
    {
        encode_ordered_key_component(&mut key, scalar.as_ref(), match key_column.order {
            OrderedIndexColumnOrder::Asc => OrderedIndexKeyOrder::Asc,
            OrderedIndexColumnOrder::Desc => OrderedIndexKeyOrder::Desc,
        })?;
    }
    ordered_equality_prefix(&key, ordered_index.equality_prefix.len())
}

fn payload_rows_to_block(
    payload_fields: &[TableField],
    rows: Vec<Vec<Scalar>>,
) -> Result<DataBlock> {
    let num_rows = rows.len();
    if payload_fields.is_empty() {
        if rows.iter().any(|row| !row.is_empty()) {
            return Err(ErrorCode::StorageOther(
                "invalid ordered payload width for empty projection".to_string(),
            ));
        }
        return Ok(DataBlock::empty_with_rows(num_rows));
    }
    let mut builders = payload_fields
        .iter()
        .map(|field| ColumnBuilder::with_capacity(&DataType::from(field.data_type()), num_rows))
        .collect::<Vec<_>>();
    for row in rows {
        if row.len() != builders.len() {
            return Err(ErrorCode::StorageOther(format!(
                "invalid ordered payload width {}, expected {}",
                row.len(),
                builders.len()
            )));
        }
        for (builder, scalar) in builders.iter_mut().zip(row.iter()) {
            builder.push(scalar.as_ref());
        }
    }
    Ok(DataBlock::new_from_columns(
        builders
            .into_iter()
            .map(|builder| builder.build())
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use databend_common_catalog::plan::OrderedIndexKeyColumn;
    use databend_common_expression::ColumnRef;
    use databend_common_expression::Constant;
    use databend_common_expression::FunctionContext;
    use databend_common_expression::ScalarRef;
    use databend_common_expression::TableDataType;
    use databend_common_expression::type_check::check_function;
    use databend_common_expression::types::DecimalDataType;
    use databend_common_expression::types::DecimalScalar;
    use databend_common_expression::types::DecimalSize;
    use databend_common_expression::types::NumberDataType;
    use databend_common_expression::types::NumberScalar;
    use databend_storages_common_index::OrderedIndexMeta;
    use databend_storages_common_index::OrderedIndexRow;
    use databend_storages_common_index::OrderedIndexWriter;
    use databend_storages_common_index::encode_ordered_payload;
    use opendal::Operator;

    use super::*;

    #[test]
    fn test_payload_rows_to_block() -> Result<()> {
        let wallet = TableField::new_from_column_id("wallet_address", TableDataType::String, 0);
        let balance = TableField::new_from_column_id(
            "balance",
            TableDataType::Number(NumberDataType::UInt64),
            1,
        );
        let block = payload_rows_to_block(&[wallet, balance], vec![vec![
            Scalar::String("wallet-a".to_string()),
            Scalar::Number(NumberScalar::UInt64(42)),
        ]])?;

        assert_eq!(block.num_rows(), 1);
        assert_eq!(
            block.get_by_offset(0).index(0),
            Some(ScalarRef::String("wallet-a"))
        );
        assert_eq!(
            block.get_by_offset(1).index(0),
            Some(ScalarRef::Number(NumberScalar::UInt64(42)))
        );
        Ok(())
    }

    #[test]
    fn test_filter_index_rows_preserves_matching_keys() -> Result<()> {
        let wallet = TableField::new_from_column_id("wallet_address", TableDataType::String, 0);
        let balance = TableField::new_from_column_id(
            "balance",
            TableDataType::Number(NumberDataType::UInt64),
            1,
        );
        let payload_fields = vec![wallet.clone(), balance.clone()];
        let payload_schema = DataSchema::new(vec![
            databend_common_expression::DataField::new(
                wallet.name(),
                DataType::from(wallet.data_type()),
            ),
            databend_common_expression::DataField::new(
                balance.name(),
                DataType::from(balance.data_type()),
            ),
        ]);
        let source = OrderedIndexSource {
            operator: Operator::new(opendal::services::Memory::default())
                .unwrap()
                .finish(),
            output_schema: payload_schema.clone(),
            payload_schema,
            func_ctx: FunctionContext::default(),
            partitions: None,
            receiver: None,
            ordered_index: OrderedIndexInfo {
                index_name: "idx".to_string(),
                index_version: "1".to_string(),
                key_columns: vec![OrderedIndexKeyColumn {
                    field: wallet.clone(),
                    order: OrderedIndexColumnOrder::Asc,
                }],
                payload_fields,
                equality_prefix: vec![Scalar::String("wallet-a".to_string())],
                limit: Some(100),
                filters: None,
                use_block_ordered_index_size_hint: false,
            },
            worker_id: 0,
            is_finished: false,
            location_preference: OrderedIndexLocationPreference::Unknown,
        };
        let rows = vec![
            OrderedIndexRow {
                encoded_key: b"k1".to_vec(),
                encoded_row_payload: encode_ordered_payload(&[
                    Scalar::String("wallet-a".to_string()),
                    Scalar::Number(NumberScalar::UInt64(1)),
                ])?,
            },
            OrderedIndexRow {
                encoded_key: b"k2".to_vec(),
                encoded_row_payload: encode_ordered_payload(&[
                    Scalar::String("wallet-a".to_string()),
                    Scalar::Number(NumberScalar::UInt64(2)),
                ])?,
            },
            OrderedIndexRow {
                encoded_key: b"k3".to_vec(),
                encoded_row_payload: encode_ordered_payload(&[
                    Scalar::String("wallet-a".to_string()),
                    Scalar::Number(NumberScalar::UInt64(3)),
                ])?,
            },
        ];
        let expr = check_function(
            None,
            "eq",
            &[],
            &[
                Expr::ColumnRef(ColumnRef {
                    span: None,
                    id: 0usize,
                    data_type: DataType::Number(NumberDataType::UInt64),
                    display_name: "balance".to_string(),
                }),
                Expr::Constant(Constant {
                    span: None,
                    scalar: Scalar::Number(NumberScalar::UInt64(2)),
                    data_type: DataType::Number(NumberDataType::UInt64),
                }),
            ],
            &BUILTIN_FUNCTIONS,
        )?;
        let filter = OrderedIndexFilter {
            expr: Some(expr),
            payload_field_indexes: vec![1],
            payload_fields: vec![balance],
            key_predicates: vec![],
            fast_predicates: None,
        };

        let row_refs = rows.iter().collect::<Vec<_>>();
        let rows = source.filter_index_row_refs(&row_refs, &filter, None, 0, 0)?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].encoded_key.as_slice(), b"k2");
        Ok(())
    }

    #[test]
    fn test_fast_filter_index_rows() -> Result<()> {
        let wallet = TableField::new_from_column_id("wallet_address", TableDataType::String, 0);
        let balance = TableField::new_from_column_id(
            "balance",
            TableDataType::Number(NumberDataType::UInt64),
            1,
        );
        let tag_black_hole = TableField::new_from_column_id(
            "tag_black_hole",
            TableDataType::Number(NumberDataType::UInt64).wrap_nullable(),
            2,
        );
        let payload_fields = vec![wallet.clone(), balance.clone(), tag_black_hole.clone()];
        let payload_schema = DataSchema::new(vec![
            databend_common_expression::DataField::new(
                wallet.name(),
                DataType::from(wallet.data_type()),
            ),
            databend_common_expression::DataField::new(
                balance.name(),
                DataType::from(balance.data_type()),
            ),
            databend_common_expression::DataField::new(
                tag_black_hole.name(),
                DataType::from(tag_black_hole.data_type()),
            ),
        ]);
        let source = OrderedIndexSource {
            operator: Operator::new(opendal::services::Memory::default())
                .unwrap()
                .finish(),
            output_schema: payload_schema.clone(),
            payload_schema,
            func_ctx: FunctionContext::default(),
            partitions: None,
            receiver: None,
            ordered_index: OrderedIndexInfo {
                index_name: "idx".to_string(),
                index_version: "1".to_string(),
                key_columns: vec![OrderedIndexKeyColumn {
                    field: wallet.clone(),
                    order: OrderedIndexColumnOrder::Asc,
                }],
                payload_fields,
                equality_prefix: vec![Scalar::String("wallet-a".to_string())],
                limit: Some(100),
                filters: None,
                use_block_ordered_index_size_hint: false,
            },
            worker_id: 0,
            is_finished: false,
            location_preference: OrderedIndexLocationPreference::Unknown,
        };
        let rows = vec![
            OrderedIndexRow {
                encoded_key: b"k1".to_vec(),
                encoded_row_payload: encode_ordered_payload(&[
                    Scalar::String("wallet-a".to_string()),
                    Scalar::Number(NumberScalar::UInt64(0)),
                    Scalar::Null,
                ])?,
            },
            OrderedIndexRow {
                encoded_key: b"k2".to_vec(),
                encoded_row_payload: encode_ordered_payload(&[
                    Scalar::String("wallet-a".to_string()),
                    Scalar::Number(NumberScalar::UInt64(2)),
                    Scalar::Number(NumberScalar::UInt64(1)),
                ])?,
            },
            OrderedIndexRow {
                encoded_key: b"k3".to_vec(),
                encoded_row_payload: encode_ordered_payload(&[
                    Scalar::String("wallet-a".to_string()),
                    Scalar::Number(NumberScalar::UInt64(3)),
                    Scalar::Null,
                ])?,
            },
        ];
        let filter = OrderedIndexFilter {
            expr: None,
            payload_field_indexes: vec![1, 2],
            payload_fields: vec![balance, tag_black_hole],
            key_predicates: vec![],
            fast_predicates: Some(vec![
                OrderedIndexFastPredicate::Compare {
                    column: 0,
                    op: OrderedIndexFastCompareOp::Gt,
                    constant: Scalar::Number(NumberScalar::UInt64(0)),
                },
                OrderedIndexFastPredicate::IsNull { column: 1 },
            ]),
        };

        let row_refs = rows.iter().collect::<Vec<_>>();
        let rows = source.filter_index_row_refs(&row_refs, &filter, None, 0, 0)?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].encoded_key.as_slice(), b"k3");
        Ok(())
    }

    #[test]
    fn test_ordered_prefix_row_refs_seek_to_prefix_start() {
        let rows = vec![
            test_row(b"wallet-0|999"),
            test_row(b"wallet-1|001"),
            test_row(b"wallet-1|002"),
            test_row(b"wallet-2|001"),
        ];

        let row_refs = ordered_prefix_row_refs(&rows, b"wallet-1|");

        assert_eq!(row_refs.len(), 2);
        assert_eq!(row_refs[0].encoded_key.as_slice(), b"wallet-1|001");
        assert_eq!(row_refs[1].encoded_key.as_slice(), b"wallet-1|002");
    }

    #[test]
    fn test_ordered_prefix_row_refs_handles_missing_prefix() {
        let rows = vec![
            test_row(b"wallet-0|999"),
            test_row(b"wallet-2|001"),
            test_row(b"wallet-3|001"),
        ];

        let row_refs = ordered_prefix_row_refs(&rows, b"wallet-1|");

        assert!(row_refs.is_empty());
    }

    #[test]
    fn test_truncate_row_refs_before_topk_boundary() {
        let rows = vec![
            test_row(b"wallet-1|001"),
            test_row(b"wallet-1|002"),
            test_row(b"wallet-1|003"),
        ];
        let mut row_refs = rows.iter().collect::<Vec<_>>();

        truncate_row_refs_before_key(&mut row_refs, Some(b"wallet-1|003"));

        assert_eq!(row_refs.len(), 2);
        assert_eq!(row_refs[0].encoded_key.as_slice(), b"wallet-1|001");
        assert_eq!(row_refs[1].encoded_key.as_slice(), b"wallet-1|002");
    }

    #[test]
    fn test_fast_filter_compiler_skips_prefix_equality() -> Result<()> {
        let wallet_eq = check_function(
            None,
            "eq",
            &[],
            &[
                Expr::ColumnRef(ColumnRef {
                    span: None,
                    id: 0usize,
                    data_type: DataType::String,
                    display_name: "wallet_address".to_string(),
                }),
                Expr::Constant(Constant {
                    span: None,
                    scalar: Scalar::String("wallet-a".to_string()),
                    data_type: DataType::String,
                }),
            ],
            &BUILTIN_FUNCTIONS,
        )?;
        let balance_gt = check_function(
            None,
            "gt",
            &[],
            &[
                Expr::ColumnRef(ColumnRef {
                    span: None,
                    id: 1usize,
                    data_type: DataType::Number(NumberDataType::UInt64),
                    display_name: "balance".to_string(),
                }),
                Expr::Constant(Constant {
                    span: None,
                    scalar: Scalar::Number(NumberScalar::UInt64(0)),
                    data_type: DataType::Number(NumberDataType::UInt64),
                }),
            ],
            &BUILTIN_FUNCTIONS,
        )?;
        let expr = check_function(
            None,
            "and_filters",
            &[],
            &[wallet_eq, balance_gt],
            &BUILTIN_FUNCTIONS,
        )?;

        let predicates =
            compile_ordered_fast_predicates(&expr, &[(0, Scalar::String("wallet-a".to_string()))])
                .unwrap();
        assert_eq!(predicates.len(), 1);
        assert!(matches!(
            predicates[0],
            OrderedIndexFastPredicate::Compare {
                column: 1,
                op: OrderedIndexFastCompareOp::Gt,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn test_fast_filter_compiler_skips_prefix_is_null() -> Result<()> {
        let tag_is_not_null = check_function(
            None,
            "is_not_null",
            &[],
            &[Expr::ColumnRef(ColumnRef {
                span: None,
                id: 0usize,
                data_type: DataType::Nullable(Box::new(DataType::Number(NumberDataType::UInt8))),
                display_name: "tag_black_hole".to_string(),
            })],
            &BUILTIN_FUNCTIONS,
        )?;
        let tag_is_null = check_function(None, "not", &[], &[tag_is_not_null], &BUILTIN_FUNCTIONS)?;

        let predicates =
            compile_ordered_fast_predicates(&tag_is_null, &[(0, Scalar::Null)]).unwrap();
        assert!(predicates.is_empty());
        Ok(())
    }

    #[test]
    fn test_fast_filter_compare_cross_numeric_types() {
        assert_eq!(
            compare_ordered_filter_scalars(
                &Scalar::Number(NumberScalar::UInt8(1)),
                &Scalar::Number(NumberScalar::Int64(0)),
            ),
            Some(Ordering::Greater)
        );

        let decimal_size = DecimalSize::new(65, 30).unwrap();
        assert_eq!(
            compare_ordered_filter_scalars(
                &Scalar::Decimal(DecimalScalar::Decimal128(1, decimal_size)),
                &Scalar::Number(NumberScalar::Int64(0)),
            ),
            Some(Ordering::Greater)
        );

        let decimal_zero = DecimalSize::new(1, 0).unwrap();
        assert_eq!(
            compare_ordered_filter_scalars(
                &Scalar::Decimal(DecimalScalar::Decimal128(1, decimal_size)),
                &Scalar::Decimal(DecimalScalar::Decimal64(0, decimal_zero)),
            ),
            Some(Ordering::Greater)
        );
    }

    #[test]
    fn test_key_predicate_compares_tinyint_suffix() -> Result<()> {
        let key_field = OrderedIndexKeyField {
            component_index: 1,
            order: OrderedIndexKeyOrder::Asc,
            data_type: TableDataType::Number(NumberDataType::Int8),
        };
        let (key_predicates, payload_predicates) = split_ordered_fast_predicates(
            vec![OrderedIndexFastPredicate::Compare {
                column: 1,
                op: OrderedIndexFastCompareOp::Gt,
                constant: Scalar::Number(NumberScalar::Int64(0)),
            }],
            &[None, Some(key_field)],
        )?;
        assert_eq!(key_predicates.len(), 1);
        assert_eq!(payload_predicates.unwrap().len(), 0);

        let key = encoded_test_key(&[
            (
                Scalar::String("token-a".to_string()),
                OrderedIndexKeyOrder::Asc,
            ),
            (
                Scalar::Number(NumberScalar::Int8(1)),
                OrderedIndexKeyOrder::Asc,
            ),
        ])?;
        assert!(evaluate_ordered_key_predicates(&key_predicates, &key));

        let key = encoded_test_key(&[
            (
                Scalar::String("token-a".to_string()),
                OrderedIndexKeyOrder::Asc,
            ),
            (
                Scalar::Number(NumberScalar::Int8(0)),
                OrderedIndexKeyOrder::Asc,
            ),
        ])?;
        assert!(!evaluate_ordered_key_predicates(&key_predicates, &key));
        Ok(())
    }

    #[test]
    fn test_key_predicate_evaluates_from_equality_prefix() -> Result<()> {
        let key_field = OrderedIndexKeyField {
            component_index: 1,
            order: OrderedIndexKeyOrder::Asc,
            data_type: TableDataType::Number(NumberDataType::Int8),
        };
        let (key_predicates, payload_predicates) = split_ordered_fast_predicates(
            vec![OrderedIndexFastPredicate::Compare {
                column: 1,
                op: OrderedIndexFastCompareOp::Gt,
                constant: Scalar::Number(NumberScalar::Int64(0)),
            }],
            &[None, Some(key_field)],
        )?;
        assert_eq!(key_predicates.len(), 1);
        assert_eq!(payload_predicates.unwrap().len(), 0);

        let key = encoded_test_key(&[
            (
                Scalar::String("token-a".to_string()),
                OrderedIndexKeyOrder::Asc,
            ),
            (
                Scalar::Number(NumberScalar::Int8(1)),
                OrderedIndexKeyOrder::Asc,
            ),
        ])?;
        let prefix = ordered_equality_prefix(&key, 1)?;
        assert!(evaluate_ordered_key_predicates_from(
            &key_predicates,
            &key,
            1,
            prefix.len()
        ));

        let key = encoded_test_key(&[
            (
                Scalar::String("token-a".to_string()),
                OrderedIndexKeyOrder::Asc,
            ),
            (
                Scalar::Number(NumberScalar::Int8(0)),
                OrderedIndexKeyOrder::Asc,
            ),
        ])?;
        let prefix = ordered_equality_prefix(&key, 1)?;
        assert!(!evaluate_ordered_key_predicates_from(
            &key_predicates,
            &key,
            1,
            prefix.len()
        ));
        Ok(())
    }

    #[test]
    fn test_key_predicate_reverses_desc_compare_order() -> Result<()> {
        let key_field = OrderedIndexKeyField {
            component_index: 0,
            order: OrderedIndexKeyOrder::Desc,
            data_type: TableDataType::Number(NumberDataType::Int64),
        };
        let (key_predicates, _) = split_ordered_fast_predicates(
            vec![OrderedIndexFastPredicate::Compare {
                column: 0,
                op: OrderedIndexFastCompareOp::Gt,
                constant: Scalar::Number(NumberScalar::Int64(0)),
            }],
            &[Some(key_field)],
        )?;

        let key = encoded_test_key(&[(
            Scalar::Number(NumberScalar::Int64(10)),
            OrderedIndexKeyOrder::Desc,
        )])?;
        assert!(evaluate_ordered_key_predicates(&key_predicates, &key));

        let key = encoded_test_key(&[(
            Scalar::Number(NumberScalar::Int64(-1)),
            OrderedIndexKeyOrder::Desc,
        )])?;
        assert!(!evaluate_ordered_key_predicates(&key_predicates, &key));
        Ok(())
    }

    #[test]
    fn test_key_predicate_compares_desc_decimal_from_integer_constant() -> Result<()> {
        let size = DecimalSize::new(20, 2)?;
        let key_field = OrderedIndexKeyField {
            component_index: 0,
            order: OrderedIndexKeyOrder::Desc,
            data_type: TableDataType::Decimal(DecimalDataType::Decimal128(size)),
        };
        let (key_predicates, payload_predicates) = split_ordered_fast_predicates(
            vec![OrderedIndexFastPredicate::Compare {
                column: 0,
                op: OrderedIndexFastCompareOp::Gt,
                constant: Scalar::Number(NumberScalar::Int64(0)),
            }],
            &[Some(key_field)],
        )?;
        assert_eq!(key_predicates.len(), 1);
        assert_eq!(payload_predicates.unwrap().len(), 0);

        let key = encoded_test_key(&[(
            Scalar::Decimal(DecimalScalar::Decimal128(100, size)),
            OrderedIndexKeyOrder::Desc,
        )])?;
        assert!(evaluate_ordered_key_predicates(&key_predicates, &key));

        let key = encoded_test_key(&[(
            Scalar::Decimal(DecimalScalar::Decimal128(0, size)),
            OrderedIndexKeyOrder::Desc,
        )])?;
        assert!(!evaluate_ordered_key_predicates(&key_predicates, &key));
        Ok(())
    }

    #[test]
    fn test_candidate_blocks_disjoint_rejects_overlap() {
        let candidates = vec![candidate_block(b"a", b"m"), candidate_block(b"n", b"z")];
        assert!(candidate_blocks_are_disjoint(&candidates));

        let candidates = vec![candidate_block(b"a", b"n"), candidate_block(b"n", b"z")];
        assert!(!candidate_blocks_are_disjoint(&candidates));

        let candidates = vec![candidate_block(b"a", b"z"), candidate_block(b"m", b"n")];
        assert!(!candidate_blocks_are_disjoint(&candidates));
    }

    #[test]
    fn test_candidate_block_pruning_uses_first_suffix_desc_range() -> Result<()> {
        let key_field = OrderedIndexKeyField {
            component_index: 2,
            order: OrderedIndexKeyOrder::Desc,
            data_type: TableDataType::Number(NumberDataType::Int64),
        };
        let (key_predicates, _) = split_ordered_fast_predicates(
            vec![OrderedIndexFastPredicate::Compare {
                column: 0,
                op: OrderedIndexFastCompareOp::Gt,
                constant: Scalar::Number(NumberScalar::Int64(0)),
            }],
            &[Some(key_field)],
        )?;
        let prefix_key = encoded_test_key(&[
            (
                Scalar::Number(NumberScalar::Int64(14)),
                OrderedIndexKeyOrder::Asc,
            ),
            (
                Scalar::String("token-a".to_string()),
                OrderedIndexKeyOrder::Asc,
            ),
        ])?;
        let prefix = ordered_equality_prefix(&prefix_key, 2)?;

        let positive_block = candidate_block(
            &encoded_test_key(&[
                (
                    Scalar::Number(NumberScalar::Int64(14)),
                    OrderedIndexKeyOrder::Asc,
                ),
                (
                    Scalar::String("token-a".to_string()),
                    OrderedIndexKeyOrder::Asc,
                ),
                (
                    Scalar::Number(NumberScalar::Int64(10)),
                    OrderedIndexKeyOrder::Desc,
                ),
            ])?,
            &encoded_test_key(&[
                (
                    Scalar::Number(NumberScalar::Int64(14)),
                    OrderedIndexKeyOrder::Asc,
                ),
                (
                    Scalar::String("token-a".to_string()),
                    OrderedIndexKeyOrder::Asc,
                ),
                (
                    Scalar::Number(NumberScalar::Int64(1)),
                    OrderedIndexKeyOrder::Desc,
                ),
            ])?,
        );
        assert!(candidate_block_may_match_key_predicates(
            &positive_block.block_meta,
            &prefix,
            2,
            &key_predicates
        ));

        let non_positive_block = candidate_block(
            &encoded_test_key(&[
                (
                    Scalar::Number(NumberScalar::Int64(14)),
                    OrderedIndexKeyOrder::Asc,
                ),
                (
                    Scalar::String("token-a".to_string()),
                    OrderedIndexKeyOrder::Asc,
                ),
                (
                    Scalar::Number(NumberScalar::Int64(0)),
                    OrderedIndexKeyOrder::Desc,
                ),
            ])?,
            &encoded_test_key(&[
                (
                    Scalar::Number(NumberScalar::Int64(14)),
                    OrderedIndexKeyOrder::Asc,
                ),
                (
                    Scalar::String("token-a".to_string()),
                    OrderedIndexKeyOrder::Asc,
                ),
                (
                    Scalar::Number(NumberScalar::Int64(-10)),
                    OrderedIndexKeyOrder::Desc,
                ),
            ])?,
        );
        assert!(!candidate_block_may_match_key_predicates(
            &non_positive_block.block_meta,
            &prefix,
            2,
            &key_predicates
        ));

        let boundary_block = candidate_block(
            b"before-prefix",
            &encoded_test_key(&[
                (
                    Scalar::Number(NumberScalar::Int64(14)),
                    OrderedIndexKeyOrder::Asc,
                ),
                (
                    Scalar::String("token-a".to_string()),
                    OrderedIndexKeyOrder::Asc,
                ),
                (
                    Scalar::Number(NumberScalar::Int64(-10)),
                    OrderedIndexKeyOrder::Desc,
                ),
            ])?,
        );
        assert!(candidate_block_may_match_key_predicates(
            &boundary_block.block_meta,
            &prefix,
            2,
            &key_predicates
        ));
        Ok(())
    }

    #[test]
    fn test_candidate_block_pruning_uses_later_suffix_only_when_prefix_is_constant() -> Result<()> {
        let key_field = OrderedIndexKeyField {
            component_index: 3,
            order: OrderedIndexKeyOrder::Asc,
            data_type: TableDataType::Number(NumberDataType::Int8).wrap_nullable(),
        };
        let (key_predicates, _) = split_ordered_fast_predicates(
            vec![OrderedIndexFastPredicate::IsNull { column: 0 }],
            &[Some(key_field)],
        )?;
        let prefix_key = encoded_test_key(&[
            (
                Scalar::Number(NumberScalar::Int64(14)),
                OrderedIndexKeyOrder::Asc,
            ),
            (
                Scalar::String("token-a".to_string()),
                OrderedIndexKeyOrder::Asc,
            ),
        ])?;
        let prefix = ordered_equality_prefix(&prefix_key, 2)?;

        let same_balance_non_null_tag_block = candidate_block(
            &encoded_test_key(&[
                (
                    Scalar::Number(NumberScalar::Int64(14)),
                    OrderedIndexKeyOrder::Asc,
                ),
                (
                    Scalar::String("token-a".to_string()),
                    OrderedIndexKeyOrder::Asc,
                ),
                (
                    Scalar::Number(NumberScalar::Int64(10)),
                    OrderedIndexKeyOrder::Desc,
                ),
                (
                    Scalar::Number(NumberScalar::Int8(1)),
                    OrderedIndexKeyOrder::Asc,
                ),
            ])?,
            &encoded_test_key(&[
                (
                    Scalar::Number(NumberScalar::Int64(14)),
                    OrderedIndexKeyOrder::Asc,
                ),
                (
                    Scalar::String("token-a".to_string()),
                    OrderedIndexKeyOrder::Asc,
                ),
                (
                    Scalar::Number(NumberScalar::Int64(10)),
                    OrderedIndexKeyOrder::Desc,
                ),
                (
                    Scalar::Number(NumberScalar::Int8(1)),
                    OrderedIndexKeyOrder::Asc,
                ),
            ])?,
        );
        assert!(!candidate_block_may_match_key_predicates(
            &same_balance_non_null_tag_block.block_meta,
            &prefix,
            2,
            &key_predicates
        ));

        let different_balance_block = candidate_block(
            &encoded_test_key(&[
                (
                    Scalar::Number(NumberScalar::Int64(14)),
                    OrderedIndexKeyOrder::Asc,
                ),
                (
                    Scalar::String("token-a".to_string()),
                    OrderedIndexKeyOrder::Asc,
                ),
                (
                    Scalar::Number(NumberScalar::Int64(11)),
                    OrderedIndexKeyOrder::Desc,
                ),
                (
                    Scalar::Number(NumberScalar::Int8(1)),
                    OrderedIndexKeyOrder::Asc,
                ),
            ])?,
            &encoded_test_key(&[
                (
                    Scalar::Number(NumberScalar::Int64(14)),
                    OrderedIndexKeyOrder::Asc,
                ),
                (
                    Scalar::String("token-a".to_string()),
                    OrderedIndexKeyOrder::Asc,
                ),
                (
                    Scalar::Number(NumberScalar::Int64(10)),
                    OrderedIndexKeyOrder::Desc,
                ),
                (Scalar::Null, OrderedIndexKeyOrder::Asc),
            ])?,
        );
        assert!(candidate_block_may_match_key_predicates(
            &different_balance_block.block_meta,
            &prefix,
            2,
            &key_predicates
        ));
        Ok(())
    }

    #[tokio::test]
    async fn test_ordered_candidate_rows_stop_after_limit() -> Result<()> {
        crate::test_utils::init_test_globals()?;
        let operator = Operator::new(opendal::services::Memory::default())
            .unwrap()
            .finish();
        let location = "ordered-limit-first.sst";
        let meta = OrderedIndexMeta {
            columns: vec![],
            metadata: Default::default(),
        };
        let mut writer = OrderedIndexWriter::new(meta, "schema", "wallet ASC", "none");
        writer.add_row(
            bytes::Bytes::from_static(b"wallet-1|001"),
            bytes::Bytes::from(encode_ordered_payload(&[Scalar::String(
                "row-1".to_string(),
            )])?),
        );
        let data = writer.finish()?;
        operator
            .write(location, data.to_vec())
            .await
            .map_err(|err| {
                ErrorCode::StorageOther(format!("write ordered index test file failed: {err:?}"))
            })?;

        let meta = load_ordered_index_meta(operator.clone(), location, None).await?;
        let first_block = meta.index_block().data_blocks[0].clone();
        let mut missing_block = first_block.clone();
        missing_block.first_key = b"wallet-2|001".to_vec();
        missing_block.last_key = b"wallet-2|999".to_vec();

        let field = TableField::new_from_column_id("wallet_address", TableDataType::String, 0);
        let source = test_source(operator, field, Some(1));
        let rows = source
            .read_candidate_rows(
                vec![
                    OrderedIndexCandidateBlock {
                        index_location: location.to_string(),
                        meta: meta.clone(),
                        block_meta: first_block,
                    },
                    OrderedIndexCandidateBlock {
                        index_location: "missing-ordered-limit-second.sst".to_string(),
                        meta,
                        block_meta: missing_block,
                    },
                ],
                b"wallet-",
                None,
            )
            .await?;

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].encoded_key.as_slice(), b"wallet-1|001");
        Ok(())
    }

    #[tokio::test]
    async fn test_ordered_index_location_cache_skips_repeated_legacy_probe() -> Result<()> {
        crate::test_utils::init_test_globals()?;
        let operator = Operator::new(opendal::services::Memory::default())
            .unwrap()
            .finish();
        let block_location = "test/_b/0123456789abcdef0123456789abcdef_v4.parquet";
        let index_name = "idx_location_cache";
        let index_version = "1234567890abcdef";
        let index_location =
            TableMetaLocationGenerator::gen_ordered_index_location_from_block_location(
                block_location,
                index_name,
                index_version,
            );
        let legacy_location =
            TableMetaLocationGenerator::gen_legacy_ordered_index_location_from_block_location(
                block_location,
                index_name,
                index_version,
            );

        let meta = OrderedIndexMeta {
            columns: vec![],
            metadata: Default::default(),
        };
        let mut writer = OrderedIndexWriter::new(meta, "schema", "wallet ASC", "none");
        writer.add_equality_prefix(bytes::Bytes::from_static(b"wallet-1|"));
        operator
            .write(&legacy_location, writer.finish()?.to_vec())
            .await
            .map_err(|err| {
                ErrorCode::StorageOther(format!("write ordered index test file failed: {err:?}"))
            })?;

        let (resolved_location, _) = load_ordered_index_meta_with_legacy_fallback(
            operator.clone(),
            &index_location,
            &legacy_location,
            None,
            OrderedIndexLocationPreference::Unknown,
        )
        .await?;
        assert_eq!(resolved_location, legacy_location);

        operator
            .write(&index_location, b"not an ordered index".to_vec())
            .await
            .map_err(|err| {
                ErrorCode::StorageOther(format!("write ordered index test file failed: {err:?}"))
            })?;
        let (resolved_location, _) = load_ordered_index_meta_with_legacy_fallback(
            operator,
            &index_location,
            &legacy_location,
            None,
            OrderedIndexLocationPreference::LegacyFirst,
        )
        .await?;
        assert_eq!(resolved_location, legacy_location);
        Ok(())
    }

    #[tokio::test]
    async fn test_legacy_first_preference_skips_current_probe_for_batch() -> Result<()> {
        crate::test_utils::init_test_globals()?;
        let operator = Operator::new(opendal::services::Memory::default())
            .unwrap()
            .finish();
        let index_name = "idx_batch_location_cache";
        let index_version = "1234567890abcdef";
        let block_locations = [
            "test/_b/11111111111111111111111111111111_v4.parquet",
            "test/_b/22222222222222222222222222222222_v4.parquet",
        ];
        for (idx, block_location) in block_locations.iter().enumerate() {
            let legacy_location =
                TableMetaLocationGenerator::gen_legacy_ordered_index_location_from_block_location(
                    block_location,
                    index_name,
                    index_version,
                );
            let meta = OrderedIndexMeta {
                columns: vec![],
                metadata: Default::default(),
            };
            let mut writer = OrderedIndexWriter::new(meta, "schema", "wallet ASC", "none");
            let key = encoded_test_key(&[(
                Scalar::String("wallet-1".to_string()),
                OrderedIndexKeyOrder::Asc,
            )])?;
            writer.add_equality_prefix(bytes::Bytes::from(ordered_equality_prefix(&key, 1)?));
            writer.add_row(
                bytes::Bytes::from(key),
                bytes::Bytes::from(encode_ordered_payload(&[Scalar::String(format!(
                    "row-{}",
                    idx + 1
                ))])?),
            );
            operator
                .write(&legacy_location, writer.finish()?.to_vec())
                .await
                .map_err(|err| {
                    ErrorCode::StorageOther(format!(
                        "write ordered index test file failed: {err:?}"
                    ))
                })?;
        }

        let second_current_location =
            TableMetaLocationGenerator::gen_ordered_index_location_from_block_location(
                block_locations[1],
                index_name,
                index_version,
            );
        operator
            .write(&second_current_location, b"not an ordered index".to_vec())
            .await
            .map_err(|err| {
                ErrorCode::StorageOther(format!("write ordered index test file failed: {err:?}"))
            })?;

        let field = TableField::new_from_column_id("wallet_address", TableDataType::String, 0);
        let mut source = test_source(operator, field, Some(10));
        source.ordered_index.index_name = index_name.to_string();
        source.ordered_index.index_version = index_version.to_string();
        source.ordered_index.equality_prefix = vec![Scalar::String("wallet-1".to_string())];
        let prefix = encode_prefix(&source.ordered_index)?;
        let candidates = source
            .load_candidate_blocks(
                block_locations
                    .iter()
                    .map(|location| test_part(location))
                    .collect(),
                &prefix,
                None,
            )
            .await?;

        assert_eq!(
            source.location_preference,
            OrderedIndexLocationPreference::LegacyFirst
        );
        assert_eq!(candidates.len(), 2);
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.index_location.contains("/_i_bt/"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_legacy_first_preference_falls_back_to_current_for_mixed_batch() -> Result<()> {
        crate::test_utils::init_test_globals()?;
        let operator = Operator::new(opendal::services::Memory::default())
            .unwrap()
            .finish();
        let index_name = "idx_mixed_location_cache";
        let index_version = "1234567890abcdef";
        let legacy_block_location = "test/_b/33333333333333333333333333333333_v4.parquet";
        let current_block_location = "test/_b/44444444444444444444444444444444_v4.parquet";
        let key = encoded_test_key(&[(
            Scalar::String("wallet-1".to_string()),
            OrderedIndexKeyOrder::Asc,
        )])?;

        let legacy_location =
            TableMetaLocationGenerator::gen_legacy_ordered_index_location_from_block_location(
                legacy_block_location,
                index_name,
                index_version,
            );
        write_ordered_test_file(operator.clone(), &legacy_location, &key, "legacy-row").await?;

        let current_location =
            TableMetaLocationGenerator::gen_ordered_index_location_from_block_location(
                current_block_location,
                index_name,
                index_version,
            );
        write_ordered_test_file(operator.clone(), &current_location, &key, "current-row").await?;

        let field = TableField::new_from_column_id("wallet_address", TableDataType::String, 0);
        let mut source = test_source(operator, field, Some(10));
        source.ordered_index.index_name = index_name.to_string();
        source.ordered_index.index_version = index_version.to_string();
        source.ordered_index.equality_prefix = vec![Scalar::String("wallet-1".to_string())];
        let prefix = encode_prefix(&source.ordered_index)?;
        let candidates = source
            .load_candidate_blocks(
                vec![
                    test_part(legacy_block_location),
                    test_part(current_block_location),
                ],
                &prefix,
                None,
            )
            .await?;

        assert_eq!(candidates.len(), 2);
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.index_location.contains("/_i_bt/"))
        );
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.index_location.contains("/_i_o/"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_overlapped_candidate_rows_stop_after_topk_boundary() -> Result<()> {
        crate::test_utils::init_test_globals()?;
        let operator = Operator::new(opendal::services::Memory::default())
            .unwrap()
            .finish();
        let location = "ordered-overlap-limit-first.sst";
        let meta = OrderedIndexMeta {
            columns: vec![],
            metadata: Default::default(),
        };
        let mut writer = OrderedIndexWriter::new(meta, "schema", "wallet ASC", "none")
            .with_data_block_size(usize::MAX);
        for key in ["wallet-1|001", "wallet-1|002", "wallet-1|003"] {
            writer.add_row(
                bytes::Bytes::from(key.as_bytes().to_vec()),
                bytes::Bytes::from(encode_ordered_payload(&[Scalar::String(key.to_string())])?),
            );
        }
        let data = writer.finish()?;
        operator
            .write(location, data.to_vec())
            .await
            .map_err(|err| {
                ErrorCode::StorageOther(format!("write ordered index test file failed: {err:?}"))
            })?;

        let meta = load_ordered_index_meta(operator.clone(), location, None).await?;
        let first_block = meta.index_block().data_blocks[0].clone();
        let mut overlapping_missing_block = first_block.clone();
        overlapping_missing_block.first_key = b"wallet-1|003".to_vec();
        overlapping_missing_block.last_key = b"wallet-1|004".to_vec();

        let field = TableField::new_from_column_id("wallet_address", TableDataType::String, 0);
        let source = test_source(operator, field, Some(2));
        let rows = source
            .read_candidate_rows(
                vec![
                    OrderedIndexCandidateBlock {
                        index_location: location.to_string(),
                        meta: meta.clone(),
                        block_meta: first_block,
                    },
                    OrderedIndexCandidateBlock {
                        index_location: "missing-ordered-overlap-limit-second.sst".to_string(),
                        meta,
                        block_meta: overlapping_missing_block,
                    },
                ],
                b"wallet-1|",
                None,
            )
            .await?;

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].encoded_key.as_slice(), b"wallet-1|001");
        assert_eq!(rows[1].encoded_key.as_slice(), b"wallet-1|002");
        Ok(())
    }

    #[tokio::test]
    async fn test_overlapped_candidate_rows_reads_until_limit_then_stops() -> Result<()> {
        crate::test_utils::init_test_globals()?;
        let operator = Operator::new(opendal::services::Memory::default())
            .unwrap()
            .finish();
        let first_location = "ordered-overlap-limit-one.sst";
        let second_location = "ordered-overlap-limit-two.sst";
        let meta = OrderedIndexMeta {
            columns: vec![],
            metadata: Default::default(),
        };
        let mut first_writer =
            OrderedIndexWriter::new(meta.clone(), "schema", "wallet ASC", "none")
                .with_data_block_size(usize::MAX);
        first_writer.add_row(
            bytes::Bytes::from_static(b"wallet-1|001"),
            bytes::Bytes::from(encode_ordered_payload(&[Scalar::String(
                "row-1".to_string(),
            )])?),
        );
        operator
            .write(first_location, first_writer.finish()?.to_vec())
            .await
            .map_err(|err| {
                ErrorCode::StorageOther(format!("write ordered index test file failed: {err:?}"))
            })?;

        let mut second_writer = OrderedIndexWriter::new(meta, "schema", "wallet ASC", "none")
            .with_data_block_size(usize::MAX);
        for key in ["wallet-1|002", "wallet-1|003"] {
            second_writer.add_row(
                bytes::Bytes::from(key.as_bytes().to_vec()),
                bytes::Bytes::from(encode_ordered_payload(&[Scalar::String(key.to_string())])?),
            );
        }
        operator
            .write(second_location, second_writer.finish()?.to_vec())
            .await
            .map_err(|err| {
                ErrorCode::StorageOther(format!("write ordered index test file failed: {err:?}"))
            })?;

        let first_meta = load_ordered_index_meta(operator.clone(), first_location, None).await?;
        let second_meta = load_ordered_index_meta(operator.clone(), second_location, None).await?;
        let mut first_block = first_meta.index_block().data_blocks[0].clone();
        first_block.last_key = b"wallet-1|002".to_vec();
        let second_block = second_meta.index_block().data_blocks[0].clone();
        let mut later_missing_block = second_block.clone();
        later_missing_block.first_key = b"wallet-1|003".to_vec();
        later_missing_block.last_key = b"wallet-1|004".to_vec();

        let field = TableField::new_from_column_id("wallet_address", TableDataType::String, 0);
        let source = test_source(operator, field, Some(2));
        let rows = source
            .read_candidate_rows(
                vec![
                    OrderedIndexCandidateBlock {
                        index_location: first_location.to_string(),
                        meta: first_meta,
                        block_meta: first_block,
                    },
                    OrderedIndexCandidateBlock {
                        index_location: second_location.to_string(),
                        meta: second_meta.clone(),
                        block_meta: second_block,
                    },
                    OrderedIndexCandidateBlock {
                        index_location: "missing-ordered-overlap-limit-third.sst".to_string(),
                        meta: second_meta,
                        block_meta: later_missing_block,
                    },
                ],
                b"wallet-1|",
                None,
            )
            .await?;

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].encoded_key.as_slice(), b"wallet-1|001");
        assert_eq!(rows[1].encoded_key.as_slice(), b"wallet-1|002");
        Ok(())
    }

    fn candidate_block(first_key: &[u8], last_key: &[u8]) -> OrderedIndexCandidateBlock {
        OrderedIndexCandidateBlock {
            index_location: "unused".to_string(),
            meta: Arc::new(OrderedIndexFileMeta {
                footer: databend_storages_common_index::OrderedIndexFooter {
                    version: databend_storages_common_index::ORDERED_INDEX_FILE_VERSION,
                    schema: String::new(),
                    key_order: String::new(),
                    compression: "none".to_string(),
                    meta: OrderedIndexMeta {
                        columns: vec![],
                        metadata: Default::default(),
                    },
                    checksum: 0,
                    sections: vec![],
                },
                index_block: databend_storages_common_index::OrderedIndexIndexBlock {
                    data_blocks: vec![],
                },
                filter_block: databend_storages_common_index::OrderedIndexFilterBlock {
                    equality_prefix_bloom: vec![],
                    equality_prefix_count: 0,
                },
            }),
            block_meta: OrderedIndexDataBlockMeta {
                first_key: first_key.to_vec(),
                last_key: last_key.to_vec(),
                offset: 0,
                length: 0,
                uncompressed_length: 0,
                row_count: 0,
                checksum: 0,
            },
        }
    }

    fn test_row(encoded_key: &[u8]) -> OrderedIndexRow {
        OrderedIndexRow {
            encoded_key: encoded_key.to_vec(),
            encoded_row_payload: Vec::new(),
        }
    }

    async fn write_ordered_test_file(
        operator: Operator,
        location: &str,
        key: &[u8],
        payload: &str,
    ) -> Result<()> {
        let meta = OrderedIndexMeta {
            columns: vec![],
            metadata: Default::default(),
        };
        let mut writer = OrderedIndexWriter::new(meta, "schema", "wallet ASC", "none");
        writer.add_equality_prefix(bytes::Bytes::from(ordered_equality_prefix(key, 1)?));
        writer.add_row(
            bytes::Bytes::from(key.to_vec()),
            bytes::Bytes::from(encode_ordered_payload(&[Scalar::String(
                payload.to_string(),
            )])?),
        );
        operator
            .write(location, writer.finish()?.to_vec())
            .await
            .map_err(|err| {
                ErrorCode::StorageOther(format!("write ordered index test file failed: {err:?}"))
            })?;
        Ok(())
    }

    fn test_source(
        operator: Operator,
        field: TableField,
        limit: Option<usize>,
    ) -> OrderedIndexSource {
        let payload_schema = DataSchema::new(vec![databend_common_expression::DataField::new(
            field.name(),
            DataType::from(field.data_type()),
        )]);
        OrderedIndexSource {
            operator,
            output_schema: payload_schema.clone(),
            payload_schema,
            func_ctx: FunctionContext::default(),
            partitions: None,
            receiver: None,
            ordered_index: OrderedIndexInfo {
                index_name: "idx".to_string(),
                index_version: "1".to_string(),
                key_columns: vec![OrderedIndexKeyColumn {
                    field: field.clone(),
                    order: OrderedIndexColumnOrder::Asc,
                }],
                payload_fields: vec![field],
                equality_prefix: vec![],
                limit,
                filters: None,
                use_block_ordered_index_size_hint: false,
            },
            worker_id: 0,
            is_finished: false,
            location_preference: OrderedIndexLocationPreference::Unknown,
        }
    }

    fn test_part(block_location: &str) -> PartInfoPtr {
        FuseBlockPartInfo::create(
            block_location.to_string(),
            None,
            0,
            None,
            0,
            None,
            1,
            Default::default(),
            None,
            None,
            databend_storages_common_table_meta::meta::Compression::None,
            None,
            None,
            None,
            None,
        )
    }

    fn encoded_test_key(values: &[(Scalar, OrderedIndexKeyOrder)]) -> Result<Vec<u8>> {
        let mut key = Vec::new();
        for (scalar, order) in values {
            encode_ordered_key_component(&mut key, scalar.as_ref(), *order)?;
        }
        Ok(key)
    }
}
