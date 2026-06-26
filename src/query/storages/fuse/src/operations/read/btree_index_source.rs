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

use async_channel::Receiver;
use databend_common_base::runtime::profile::Profile;
use databend_common_base::runtime::profile::ProfileStatisticsName;
use databend_common_catalog::plan::BtreeIndexColumnOrder;
use databend_common_catalog::plan::BtreeIndexInfo;
use databend_common_catalog::plan::DataSourcePlan;
use databend_common_catalog::plan::Filters;
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
use databend_common_expression::TableField;
use databend_common_expression::types::BooleanType;
use databend_common_expression::types::DataType;
use databend_common_functions::BUILTIN_FUNCTIONS;
use databend_common_pipeline::core::OutputPort;
use databend_common_pipeline::core::Pipeline;
use databend_common_pipeline::core::SourcePipeBuilder;
use databend_common_pipeline::sources::AsyncSource;
use databend_common_pipeline::sources::AsyncSourcer;
use databend_storages_common_index::BtreeIndexDataBlockMeta;
use databend_storages_common_index::BtreeIndexFileMeta;
use databend_storages_common_index::BtreeIndexKeyOrder;
use databend_storages_common_index::BtreeIndexRow;
use databend_storages_common_index::btree_equality_prefix;
use databend_storages_common_index::decode_btree_payload;
use databend_storages_common_index::encode_btree_key_component;
use futures::future;
use opendal::Operator;

use crate::fuse_part::FuseBlockPartInfo;
use crate::io::TableMetaLocationGenerator;
use crate::io::load_btree_index_data_block;
use crate::io::load_btree_index_meta;

const BTREE_INDEX_DATA_BLOCK_READ_BATCH_SIZE: usize = 32;

pub fn build_btree_index_source_pipeline(
    ctx: Arc<dyn TableContext>,
    operator: Operator,
    pipeline: &mut Pipeline,
    plan: &DataSourcePlan,
    btree_index: BtreeIndexInfo,
    max_threads: usize,
    receiver: Option<Receiver<Result<PartInfoPtr>>>,
) -> Result<()> {
    let max_threads = max_threads.max(1);
    let partitions =
        crate::operations::read::fuse_source::dispatch_partitions(ctx.clone(), plan, max_threads);
    let partitions = StealablePartitions::new(partitions, ctx.clone());
    let output_schema: DataSchema = plan.schema().as_ref().into();
    let payload_schema: DataSchema = DataSchema::new(
        btree_index
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
            BtreeIndexSource::create(
                ctx.clone(),
                output,
                operator.clone(),
                output_schema.clone(),
                payload_schema.clone(),
                func_ctx.clone(),
                Some(partitions.clone()),
                receiver.clone(),
                btree_index.clone(),
                worker_id,
            )?,
        );
    }
    pipeline.add_pipe(source_builder.finalize());
    Ok(())
}

struct BtreeIndexSource {
    operator: Operator,
    output_schema: DataSchema,
    payload_schema: DataSchema,
    func_ctx: FunctionContext,
    partitions: Option<StealablePartitions>,
    receiver: Option<Receiver<Result<PartInfoPtr>>>,
    btree_index: BtreeIndexInfo,
    worker_id: usize,
    is_finished: bool,
}

impl BtreeIndexSource {
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
        btree_index: BtreeIndexInfo,
        worker_id: usize,
    ) -> Result<databend_common_pipeline::core::ProcessorPtr> {
        AsyncSourcer::create(ctx.get_scan_progress(), output, Self {
            operator,
            output_schema,
            payload_schema,
            func_ctx,
            partitions,
            receiver,
            btree_index,
            worker_id,
            is_finished: false,
        })
    }
}

#[async_trait::async_trait]
impl AsyncSource for BtreeIndexSource {
    const NAME: &'static str = "BtreeIndexSource";

    #[async_backtrace::framed]
    async fn generate(&mut self) -> Result<Option<DataBlock>> {
        if self.is_finished {
            return Ok(None);
        }
        self.is_finished = true;

        let prefix = encode_prefix(&self.btree_index)?;
        let filter = self.build_filter_expr(self.btree_index.filters.as_ref())?;
        let mut candidates = Vec::new();
        while let Some(parts) = self.fetch_parts().await? {
            if parts.is_empty() {
                continue;
            }

            candidates.append(&mut self.load_candidate_blocks(parts, &prefix).await?);
        }

        let rows = self
            .read_candidate_rows(candidates, &prefix, filter.as_ref())
            .await?;
        if rows.is_empty() {
            return Ok(None);
        }
        let decode_start = Instant::now();
        let mut decoded_rows = Vec::with_capacity(rows.len());
        for row in rows {
            decoded_rows.push(decode_btree_payload(&row.encoded_row_payload)?);
        }
        record_elapsed(
            ProfileStatisticsName::BtreeIndexPayloadDecodeTime,
            decode_start,
        );

        let mut block = payload_rows_to_block(&self.btree_index.payload_fields, decoded_rows)?;
        block = block.resort(&self.payload_schema, &self.output_schema)?;
        Ok(Some(block))
    }
}

struct BtreeIndexCandidateBlock {
    index_location: String,
    meta: Arc<BtreeIndexFileMeta>,
    block_meta: BtreeIndexDataBlockMeta,
}

impl BtreeIndexSource {
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

    fn build_filter_expr(&self, filters: Option<&Filters>) -> Result<Option<Expr<usize>>> {
        let Some(filters) = filters else {
            return Ok(None);
        };
        filters
            .filter
            .as_expr(&BUILTIN_FUNCTIONS)
            .project_column_ref(|name| self.payload_schema.index_of(name))
            .map(Some)
    }

    fn filter_index_rows(
        &self,
        rows: Vec<BtreeIndexRow>,
        filter: &Expr<usize>,
        limit: Option<usize>,
    ) -> Result<Vec<BtreeIndexRow>> {
        if rows.is_empty() {
            return Ok(rows);
        }

        let start = Instant::now();
        let payload_rows = rows
            .iter()
            .map(|row| decode_btree_payload(&row.encoded_row_payload))
            .collect::<Result<Vec<_>>>()?;
        let block = payload_rows_to_block(&self.btree_index.payload_fields, payload_rows)?;
        let evaluator = Evaluator::new(&block, &self.func_ctx, &BUILTIN_FUNCTIONS);
        let filter = evaluator
            .run(filter)?
            .try_downcast::<BooleanType>()
            .unwrap();

        let filtered_rows = match filter {
            databend_common_expression::Value::Scalar(true) => truncate_rows(rows, limit),
            databend_common_expression::Value::Scalar(false) => Vec::new(),
            databend_common_expression::Value::Column(bitmap) => truncate_rows(
                rows.into_iter()
                    .enumerate()
                    .filter_map(|(idx, row)| bitmap.get_bit(idx).then_some(row))
                    .collect(),
                limit,
            ),
        };
        record_elapsed(ProfileStatisticsName::BtreeIndexFilterTime, start);
        Ok(filtered_rows)
    }

    async fn load_candidate_blocks(
        &self,
        parts: Vec<PartInfoPtr>,
        prefix: &[u8],
    ) -> Result<Vec<BtreeIndexCandidateBlock>> {
        let futures = parts.into_iter().map(|part| {
            Self::load_candidate_blocks_for_part(
                self.operator.clone(),
                part,
                self.btree_index.index_name.clone(),
                self.btree_index.index_version.clone(),
                self.btree_index.use_block_btree_index_size_hint,
                prefix.to_vec(),
            )
        });
        let candidate_batches = future::try_join_all(futures).await?;
        let candidates = candidate_batches.into_iter().flatten().collect::<Vec<_>>();
        Profile::record_usize_profile(
            ProfileStatisticsName::BtreeIndexCandidateBlocks,
            candidates.len(),
        );
        Ok(candidates)
    }

    async fn load_candidate_blocks_for_part(
        operator: Operator,
        part: PartInfoPtr,
        index_name: String,
        index_version: String,
        use_block_btree_index_size_hint: bool,
        prefix: Vec<u8>,
    ) -> Result<Vec<BtreeIndexCandidateBlock>> {
        let fuse_part = FuseBlockPartInfo::from_part(&part)?;
        let index_location =
            TableMetaLocationGenerator::gen_btree_index_location_from_block_location(
                &fuse_part.location,
                &index_name,
                &index_version,
            );
        let len_hint = use_block_btree_index_size_hint
            .then_some(fuse_part.btree_index_size)
            .flatten();
        let meta = load_btree_index_meta(operator, &index_location, len_hint).await?;
        if !meta.may_contain_equality_prefix(&prefix) {
            return Ok(Vec::new());
        }
        Ok(meta
            .blocks_for_prefix(&prefix)
            .into_iter()
            .map(|block_meta| BtreeIndexCandidateBlock {
                index_location: index_location.clone(),
                meta: meta.clone(),
                block_meta,
            })
            .collect())
    }

    async fn read_candidate_rows(
        &self,
        mut candidates: Vec<BtreeIndexCandidateBlock>,
        prefix: &[u8],
        filter: Option<&Expr<usize>>,
    ) -> Result<Vec<BtreeIndexRow>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        candidates.sort_by(|left, right| {
            left.block_meta
                .first_key
                .cmp(&right.block_meta.first_key)
                .then_with(|| left.block_meta.last_key.cmp(&right.block_meta.last_key))
                .then_with(|| left.index_location.cmp(&right.index_location))
                .then_with(|| left.block_meta.offset.cmp(&right.block_meta.offset))
        });

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
        candidates: Vec<BtreeIndexCandidateBlock>,
        prefix: &[u8],
        filter: Option<&Expr<usize>>,
    ) -> Result<Vec<BtreeIndexRow>> {
        let mut rows = Vec::new();
        for candidate in candidates {
            let remaining = self
                .btree_index
                .limit
                .map(|limit| limit.saturating_sub(rows.len()));
            if remaining == Some(0) {
                return Ok(rows);
            }
            let mut block_rows = self
                .read_candidate_block(&candidate, prefix, filter, remaining)
                .await?;
            rows.append(&mut block_rows);
            if self
                .btree_index
                .limit
                .is_some_and(|limit| rows.len() >= limit)
            {
                rows.truncate(self.btree_index.limit.unwrap());
                return Ok(rows);
            }
        }
        Ok(rows)
    }

    async fn read_candidate_rows_with_sort(
        &self,
        candidates: Vec<BtreeIndexCandidateBlock>,
        prefix: &[u8],
        filter: Option<&Expr<usize>>,
    ) -> Result<Vec<BtreeIndexRow>> {
        let Some(limit) = self.btree_index.limit else {
            let mut rows = Vec::new();
            for batch in candidates.chunks(BTREE_INDEX_DATA_BLOCK_READ_BATCH_SIZE) {
                for mut block_rows in self
                    .read_candidate_blocks_batch(batch, prefix, filter, None)
                    .await?
                {
                    rows.append(&mut block_rows);
                }
            }
            rows.sort_by(|left, right| left.encoded_key.cmp(&right.encoded_key));
            return Ok(rows);
        };

        let mut rows = Vec::new();
        let mut candidate_offset = 0;
        let mut batch_size = 1;
        while candidate_offset < candidates.len() {
            let batch_end = (candidate_offset + batch_size).min(candidates.len());
            for mut block_rows in self
                .read_candidate_blocks_batch(
                    &candidates[candidate_offset..batch_end],
                    prefix,
                    filter,
                    Some(limit),
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
            batch_size = BTREE_INDEX_DATA_BLOCK_READ_BATCH_SIZE;
        }
        sort_and_truncate_rows(&mut rows, limit);
        Ok(rows)
    }

    async fn read_candidate_blocks_batch(
        &self,
        candidates: &[BtreeIndexCandidateBlock],
        prefix: &[u8],
        filter: Option<&Expr<usize>>,
        limit: Option<usize>,
    ) -> Result<Vec<Vec<BtreeIndexRow>>> {
        let futures = candidates
            .iter()
            .map(|candidate| self.read_candidate_block(candidate, prefix, filter, limit));
        future::try_join_all(futures).await
    }

    async fn read_candidate_block(
        &self,
        candidate: &BtreeIndexCandidateBlock,
        prefix: &[u8],
        filter: Option<&Expr<usize>>,
        limit: Option<usize>,
    ) -> Result<Vec<BtreeIndexRow>> {
        let mut rows = load_btree_index_data_block(
            self.operator.clone(),
            &candidate.index_location,
            &candidate.meta,
            &candidate.block_meta,
        )
        .await?;
        Profile::record_usize_profile(ProfileStatisticsName::BtreeIndexRowsDecoded, rows.len());
        rows.retain(|row| row.encoded_key.starts_with(prefix));
        let rows = if let Some(filter) = filter {
            self.filter_index_rows(rows, filter, limit)?
        } else {
            truncate_rows(rows, limit)
        };
        Profile::record_usize_profile(ProfileStatisticsName::BtreeIndexRowsMatched, rows.len());
        Ok(rows)
    }
}

fn record_elapsed(name: ProfileStatisticsName, start: Instant) {
    Profile::record_usize_profile(name, start.elapsed().as_nanos() as usize);
}

fn truncate_rows(mut rows: Vec<BtreeIndexRow>, limit: Option<usize>) -> Vec<BtreeIndexRow> {
    if let Some(limit) = limit
        && rows.len() > limit
    {
        rows.truncate(limit);
    }
    rows
}

fn sort_and_truncate_rows(rows: &mut Vec<BtreeIndexRow>, limit: usize) {
    rows.sort_by(|left, right| left.encoded_key.cmp(&right.encoded_key));
    rows.truncate(limit);
}

fn candidate_blocks_are_disjoint(candidates: &[BtreeIndexCandidateBlock]) -> bool {
    candidates
        .windows(2)
        .all(|blocks| blocks[0].block_meta.last_key < blocks[1].block_meta.first_key)
}

fn encode_prefix(btree_index: &BtreeIndexInfo) -> Result<Vec<u8>> {
    let mut key = Vec::new();
    for (scalar, key_column) in btree_index
        .equality_prefix
        .iter()
        .zip(btree_index.key_columns.iter())
    {
        encode_btree_key_component(&mut key, scalar.as_ref(), match key_column.order {
            BtreeIndexColumnOrder::Asc => BtreeIndexKeyOrder::Asc,
            BtreeIndexColumnOrder::Desc => BtreeIndexKeyOrder::Desc,
        })?;
    }
    btree_equality_prefix(&key, btree_index.equality_prefix.len())
}

fn payload_rows_to_block(
    payload_fields: &[TableField],
    rows: Vec<Vec<Scalar>>,
) -> Result<DataBlock> {
    let num_rows = rows.len();
    let mut builders = payload_fields
        .iter()
        .map(|field| ColumnBuilder::with_capacity(&DataType::from(field.data_type()), num_rows))
        .collect::<Vec<_>>();
    for row in rows {
        if row.len() != builders.len() {
            return Err(ErrorCode::StorageOther(format!(
                "invalid btree payload width {}, expected {}",
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
    use databend_common_catalog::plan::BtreeIndexKeyColumn;
    use databend_common_expression::ColumnRef;
    use databend_common_expression::Constant;
    use databend_common_expression::FunctionContext;
    use databend_common_expression::ScalarRef;
    use databend_common_expression::TableDataType;
    use databend_common_expression::type_check::check_function;
    use databend_common_expression::types::NumberDataType;
    use databend_common_expression::types::NumberScalar;
    use databend_storages_common_index::BtreeIndexMeta;
    use databend_storages_common_index::BtreeIndexRow;
    use databend_storages_common_index::BtreeIndexWriter;
    use databend_storages_common_index::encode_btree_payload;
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
        let source = BtreeIndexSource {
            operator: Operator::new(opendal::services::Memory::default())
                .unwrap()
                .finish(),
            output_schema: payload_schema.clone(),
            payload_schema,
            func_ctx: FunctionContext::default(),
            partitions: None,
            receiver: None,
            btree_index: BtreeIndexInfo {
                index_name: "idx".to_string(),
                index_version: "1".to_string(),
                key_columns: vec![BtreeIndexKeyColumn {
                    field: wallet.clone(),
                    order: BtreeIndexColumnOrder::Asc,
                }],
                payload_fields,
                equality_prefix: vec![Scalar::String("wallet-a".to_string())],
                limit: Some(100),
                filters: None,
                use_block_btree_index_size_hint: false,
            },
            worker_id: 0,
            is_finished: false,
        };
        let rows = vec![
            BtreeIndexRow {
                encoded_key: b"k1".to_vec(),
                encoded_row_payload: encode_btree_payload(&[
                    Scalar::String("wallet-a".to_string()),
                    Scalar::Number(NumberScalar::UInt64(1)),
                ])?,
            },
            BtreeIndexRow {
                encoded_key: b"k2".to_vec(),
                encoded_row_payload: encode_btree_payload(&[
                    Scalar::String("wallet-a".to_string()),
                    Scalar::Number(NumberScalar::UInt64(2)),
                ])?,
            },
            BtreeIndexRow {
                encoded_key: b"k3".to_vec(),
                encoded_row_payload: encode_btree_payload(&[
                    Scalar::String("wallet-a".to_string()),
                    Scalar::Number(NumberScalar::UInt64(3)),
                ])?,
            },
        ];
        let filter = check_function(
            None,
            "eq",
            &[],
            &[
                Expr::ColumnRef(ColumnRef {
                    span: None,
                    id: "balance".to_string(),
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
        )?
        .project_column_ref(|name| source.payload_schema.index_of(name))?;

        let rows = source.filter_index_rows(rows, &filter, None)?;
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].encoded_key.as_slice(), b"k2");
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

    #[tokio::test]
    async fn test_ordered_candidate_rows_stop_after_limit() -> Result<()> {
        crate::test_utils::init_test_globals()?;
        let operator = Operator::new(opendal::services::Memory::default())
            .unwrap()
            .finish();
        let location = "btree-limit-first.sst";
        let meta = BtreeIndexMeta {
            columns: vec![],
            metadata: Default::default(),
        };
        let mut writer = BtreeIndexWriter::new(meta, "schema", "wallet ASC", "none");
        writer.add_row(
            bytes::Bytes::from_static(b"wallet-1|001"),
            bytes::Bytes::from(encode_btree_payload(&[Scalar::String(
                "row-1".to_string(),
            )])?),
        );
        let data = writer.finish()?;
        operator
            .write(location, data.to_vec())
            .await
            .map_err(|err| {
                ErrorCode::StorageOther(format!("write btree index test file failed: {err:?}"))
            })?;

        let meta = load_btree_index_meta(operator.clone(), location, None).await?;
        let first_block = meta.index_block().data_blocks[0].clone();
        let mut missing_block = first_block.clone();
        missing_block.first_key = b"wallet-2|001".to_vec();
        missing_block.last_key = b"wallet-2|999".to_vec();

        let field = TableField::new_from_column_id("wallet_address", TableDataType::String, 0);
        let source = test_source(operator, field, Some(1));
        let rows = source
            .read_candidate_rows(
                vec![
                    BtreeIndexCandidateBlock {
                        index_location: location.to_string(),
                        meta: meta.clone(),
                        block_meta: first_block,
                    },
                    BtreeIndexCandidateBlock {
                        index_location: "missing-btree-limit-second.sst".to_string(),
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
    async fn test_overlapped_candidate_rows_stop_after_topk_boundary() -> Result<()> {
        crate::test_utils::init_test_globals()?;
        let operator = Operator::new(opendal::services::Memory::default())
            .unwrap()
            .finish();
        let location = "btree-overlap-limit-first.sst";
        let meta = BtreeIndexMeta {
            columns: vec![],
            metadata: Default::default(),
        };
        let mut writer = BtreeIndexWriter::new(meta, "schema", "wallet ASC", "none")
            .with_data_block_size(usize::MAX);
        for key in ["wallet-1|001", "wallet-1|002", "wallet-1|003"] {
            writer.add_row(
                bytes::Bytes::from(key.as_bytes().to_vec()),
                bytes::Bytes::from(encode_btree_payload(&[Scalar::String(key.to_string())])?),
            );
        }
        let data = writer.finish()?;
        operator
            .write(location, data.to_vec())
            .await
            .map_err(|err| {
                ErrorCode::StorageOther(format!("write btree index test file failed: {err:?}"))
            })?;

        let meta = load_btree_index_meta(operator.clone(), location, None).await?;
        let first_block = meta.index_block().data_blocks[0].clone();
        let mut overlapping_missing_block = first_block.clone();
        overlapping_missing_block.first_key = b"wallet-1|003".to_vec();
        overlapping_missing_block.last_key = b"wallet-1|004".to_vec();

        let field = TableField::new_from_column_id("wallet_address", TableDataType::String, 0);
        let source = test_source(operator, field, Some(2));
        let rows = source
            .read_candidate_rows(
                vec![
                    BtreeIndexCandidateBlock {
                        index_location: location.to_string(),
                        meta: meta.clone(),
                        block_meta: first_block,
                    },
                    BtreeIndexCandidateBlock {
                        index_location: "missing-btree-overlap-limit-second.sst".to_string(),
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

    fn candidate_block(first_key: &[u8], last_key: &[u8]) -> BtreeIndexCandidateBlock {
        BtreeIndexCandidateBlock {
            index_location: "unused".to_string(),
            meta: Arc::new(BtreeIndexFileMeta {
                footer: databend_storages_common_index::BtreeIndexFooter {
                    version: databend_storages_common_index::BTREE_INDEX_FILE_VERSION,
                    schema: String::new(),
                    key_order: String::new(),
                    compression: "none".to_string(),
                    meta: BtreeIndexMeta {
                        columns: vec![],
                        metadata: Default::default(),
                    },
                    checksum: 0,
                    sections: vec![],
                },
                index_block: databend_storages_common_index::BtreeIndexIndexBlock {
                    data_blocks: vec![],
                },
                filter_block: databend_storages_common_index::BtreeIndexFilterBlock {
                    equality_prefix_bloom: vec![],
                    equality_prefix_count: 0,
                },
            }),
            block_meta: BtreeIndexDataBlockMeta {
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

    fn test_source(
        operator: Operator,
        field: TableField,
        limit: Option<usize>,
    ) -> BtreeIndexSource {
        let payload_schema = DataSchema::new(vec![databend_common_expression::DataField::new(
            field.name(),
            DataType::from(field.data_type()),
        )]);
        BtreeIndexSource {
            operator,
            output_schema: payload_schema.clone(),
            payload_schema,
            func_ctx: FunctionContext::default(),
            partitions: None,
            receiver: None,
            btree_index: BtreeIndexInfo {
                index_name: "idx".to_string(),
                index_version: "1".to_string(),
                key_columns: vec![BtreeIndexKeyColumn {
                    field: field.clone(),
                    order: BtreeIndexColumnOrder::Asc,
                }],
                payload_fields: vec![field],
                equality_prefix: vec![],
                limit,
                filters: None,
                use_block_btree_index_size_hint: false,
            },
            worker_id: 0,
            is_finished: false,
        }
    }
}
