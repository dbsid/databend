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

use std::any::Any;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use databend_common_catalog::catalog::CatalogManager;
use databend_common_catalog::plan::DataSourceInfo;
use databend_common_catalog::plan::DataSourcePlan;
use databend_common_catalog::plan::Filters;
use databend_common_catalog::plan::InternalColumn;
use databend_common_catalog::plan::OrderedIndexColumnOrder;
use databend_common_catalog::plan::OrderedIndexInfo;
use databend_common_catalog::plan::OrderedIndexKeyColumn;
use databend_common_catalog::plan::PartStatistics;
use databend_common_catalog::plan::PartitionsShuffleKind;
use databend_common_catalog::plan::PrewhereInfo;
use databend_common_catalog::plan::Projection;
use databend_common_catalog::plan::PushDownInfo;
use databend_common_catalog::plan::VirtualColumnField;
use databend_common_catalog::plan::VirtualColumnInfo;
use databend_common_exception::ErrorCode;
use databend_common_exception::Result;
use databend_common_expression::ConstantFolder;
use databend_common_expression::DataField;
use databend_common_expression::DataSchema;
use databend_common_expression::DataSchemaRef;
use databend_common_expression::FieldIndex;
use databend_common_expression::ROW_ID_COL_NAME;
use databend_common_expression::RemoteExpr;
use databend_common_expression::Scalar;
use databend_common_expression::TableDataType;
use databend_common_expression::TableSchema;
use databend_common_expression::TableSchemaRef;
use databend_common_expression::type_check::check_function;
use databend_common_expression::types::DataType;
use databend_common_functions::BUILTIN_FUNCTIONS;
use databend_common_meta_app::schema::TableIndexColumn;
use databend_common_meta_app::schema::TableIndexColumnOrder;
use databend_common_meta_app::schema::TableIndexType;
use databend_common_pipeline_transforms::TransformPipelineHelper;
use databend_common_pipeline_transforms::blocks::CompoundBlockOperator;
use databend_common_pipeline_transforms::columns::TransformAddInternalColumns;
use databend_common_sql::BaseTableColumn;
use databend_common_sql::ColumnEntry;
use databend_common_sql::ColumnSet;
use databend_common_sql::DUMMY_TABLE_INDEX;
use databend_common_sql::DerivedColumn;
use databend_common_sql::IndexType;
use databend_common_sql::Metadata;
use databend_common_sql::ScalarExpr;
use databend_common_sql::Symbol;
use databend_common_sql::TableInternalColumn;
use databend_common_sql::TypeCheck;
use databend_common_sql::VirtualColumn;
use databend_common_sql::binder::INTERNAL_COLUMN_FACTORY;
use databend_common_sql::evaluator::BlockOperator;
use databend_common_sql::executor::cast_expr_to_non_null_boolean;
use databend_common_sql::executor::table_read_plan::ToReadDataSourcePlan;
use databend_common_sql::plans::FunctionCall;
use databend_common_sql::plans::SortItem;
use databend_common_storages_fuse::FuseTable;
use databend_common_storages_fuse::operations::need_reserve_block_info;
use rand::distributions::Bernoulli;
use rand::distributions::Distribution;
use rand::thread_rng;
use sha2::Digest;
use sha2::Sha256;

use crate::physical_plans::AddStreamColumn;
use crate::physical_plans::PhysicalPlanBuilder;
use crate::physical_plans::explain::PlanStatsInfo;
use crate::physical_plans::format::PhysicalFormat;
use crate::physical_plans::format::TableScanFormatter;
use crate::physical_plans::physical_plan::IPhysicalPlan;
use crate::physical_plans::physical_plan::PhysicalPlan;
use crate::physical_plans::physical_plan::PhysicalPlanMeta;
use crate::pipelines::PipelineBuilder;
use crate::sessions::TableContextPartitionStats;
use crate::sessions::TableContextSettings;
use crate::sessions::TableContextTableFactory;

const ORDERED_INDEX_OPTION_COVERED_TYPE: &str = "index_covered_type";
const ORDERED_INDEX_COVERED_ALL_COLUMNS: &str = "covered_all_columns_in_schema";

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TableScan {
    pub meta: PhysicalPlanMeta,
    pub scan_id: usize,
    pub name_mapping: BTreeMap<String, String>,
    pub source: Box<DataSourcePlan>,
    pub internal_column: Option<BTreeMap<FieldIndex, InternalColumn>>,

    pub table_index: Option<IndexType>,
    pub stat_info: Option<PlanStatsInfo>,
}

#[typetag::serde]
impl IPhysicalPlan for TableScan {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn get_meta(&self) -> &PhysicalPlanMeta {
        &self.meta
    }

    fn get_meta_mut(&mut self) -> &mut PhysicalPlanMeta {
        &mut self.meta
    }

    #[recursive::recursive]
    fn output_schema(&self) -> Result<DataSchemaRef> {
        Self::output_fields(self.source.schema(), &self.name_mapping).map(DataSchema::new_ref)
    }

    fn formatter(&self) -> Result<Box<dyn PhysicalFormat + '_>> {
        Ok(TableScanFormatter::create(self))
    }

    fn try_find_single_data_source(&self) -> Option<&DataSourcePlan> {
        Some(&self.source)
    }

    fn get_all_data_source(&self, sources: &mut Vec<(u32, Box<DataSourcePlan>)>) {
        sources.push((self.get_id(), self.source.clone()));
    }

    fn set_pruning_stats(&mut self, stats: &mut HashMap<u32, PartStatistics>) {
        if let Some(stat) = stats.remove(&self.get_id()) {
            self.source.statistics = stat;
        }
    }

    fn is_warehouse_distributed_plan(&self) -> bool {
        self.source.parts.kind == PartitionsShuffleKind::BroadcastWarehouse
    }

    fn get_desc(&self) -> Result<String> {
        Ok(format!(
            "{}.{}",
            self.source.source_info.catalog_name(),
            self.source.source_info.desc()
        ))
    }

    fn get_labels(&self) -> Result<HashMap<String, Vec<String>>> {
        let mut labels = HashMap::from([
            (String::from("Full table name"), vec![format!(
                "{}.{}",
                self.source.source_info.catalog_name(),
                self.source.source_info.desc()
            )]),
            (
                format!(
                    "Columns ({} / {})",
                    self.output_schema()?.num_fields(),
                    std::cmp::max(
                        self.output_schema()?.num_fields(),
                        self.source.source_info.schema().num_fields(),
                    )
                ),
                self.name_mapping.keys().cloned().collect(),
            ),
            (String::from("Total partitions"), vec![
                self.source.statistics.partitions_total.to_string(),
            ]),
        ]);
        if let Some(ordered_index) = self
            .source
            .push_downs
            .as_ref()
            .and_then(|push_downs| push_downs.ordered_index.as_ref())
        {
            labels.insert(String::from("Ordered index"), vec![format!(
                "{}@{}",
                ordered_index.index_name, ordered_index.index_version
            )]);
        }
        Ok(labels)
    }

    fn derive(&self, children: Vec<PhysicalPlan>) -> PhysicalPlan {
        assert!(children.is_empty());
        PhysicalPlan::new(TableScan {
            meta: self.meta.clone(),
            scan_id: self.scan_id,
            name_mapping: self.name_mapping.clone(),
            source: self.source.clone(),
            internal_column: self.internal_column.clone(),
            table_index: self.table_index,
            stat_info: self.stat_info.clone(),
        })
    }

    fn build_pipeline2(&self, builder: &mut PipelineBuilder) -> Result<()> {
        let table = builder.ctx.build_table_from_source_plan(&self.source)?;
        builder.ctx.set_partitions(self.source.parts.clone())?;

        if self.source.parts.kind != PartitionsShuffleKind::PreserveOrder
            && builder.ctx.get_settings().get_enable_prune_pipeline()?
        {
            if let Some(prune_pipeline) = table.build_prune_pipeline(
                builder.ctx.clone(),
                &self.source,
                &mut builder.main_pipeline,
                self.get_id(),
            )? {
                builder.pipelines.push(prune_pipeline);
            }
        }

        table.read_data(
            builder.ctx.clone(),
            &self.source,
            &mut builder.main_pipeline,
            true,
        )?;

        // Fill internal columns if needed.
        if let Some(internal_columns) = &self.internal_column {
            builder
                .main_pipeline
                .add_transformer(|| TransformAddInternalColumns::new(internal_columns.clone()));
        }

        let schema = self.source.schema();
        let mut projection = self
            .name_mapping
            .keys()
            .map(|name| schema.index_of(name.as_str()))
            .collect::<Result<Vec<usize>>>()?;
        projection.sort();

        // if projection is sequential, no need to add projection
        if projection != (0..schema.fields().len()).collect::<Vec<usize>>() {
            let ops = vec![BlockOperator::Project { projection }];
            let num_input_columns = schema.num_fields();
            builder.main_pipeline.add_transformer(|| {
                CompoundBlockOperator::new(ops.clone(), builder.func_ctx.clone(), num_input_columns)
            });
        }

        Ok(())
    }
}

impl TableScan {
    pub fn create(
        scan_id: usize,
        name_mapping: BTreeMap<String, String>,
        source: Box<DataSourcePlan>,
        table_index: Option<IndexType>,
        stat_info: Option<PlanStatsInfo>,
        internal_column: Option<BTreeMap<FieldIndex, InternalColumn>>,
    ) -> PhysicalPlan {
        let name = match &source.source_info {
            DataSourceInfo::TableSource(_) => "TableScan".to_string(),
            DataSourceInfo::StageSource(_) => "StageScan".to_string(),
            DataSourceInfo::ParquetSource(_) => "ParquetScan".to_string(),
            DataSourceInfo::ResultScanSource(_) => "ResultScan".to_string(),
            DataSourceInfo::ORCSource(_) => "OrcScan".to_string(),
        };

        PhysicalPlan::new(TableScan {
            meta: PhysicalPlanMeta::new(name),
            source,
            scan_id,
            name_mapping,
            table_index,
            stat_info,
            internal_column,
        })
    }

    pub fn output_fields(
        schema: TableSchemaRef,
        name_mapping: &BTreeMap<String, String>,
    ) -> Result<Vec<DataField>> {
        let mut fields = Vec::with_capacity(name_mapping.len());
        let mut name_and_ids = name_mapping
            .iter()
            .map(|(name, id)| {
                let index = schema.index_of(name)?;
                Ok((name, id, index))
            })
            .collect::<Result<Vec<_>>>()?;
        // Make the order of output fields the same as their indexes in te table schema.
        name_and_ids.sort_by_key(|(_, _, index)| *index);

        for (name, id, _) in name_and_ids {
            let orig_field = schema.field_with_name(name)?;
            let data_type = DataType::from(orig_field.data_type());
            fields.push(DataField::new(id, data_type));
        }
        Ok(fields)
    }
}

impl PhysicalPlanBuilder {
    pub async fn build_table_scan(
        &mut self,
        scan: &databend_common_sql::plans::Scan,
        required: ColumnSet,
        stat_info: PlanStatsInfo,
    ) -> Result<PhysicalPlan> {
        // 1. Prune unused Columns.
        // Some table may not have any column,
        // e.g. `system.sync_crash_me`
        let scan = if scan.columns.is_empty() {
            scan.clone()
        } else {
            let mut columns = scan.columns.clone();

            let required_column_ids: Vec<_> = required.difference(&columns).cloned().collect();
            if !required_column_ids.is_empty() {
                // add virtual columns to table scan columns.
                let read_guard = self.metadata.read();
                let virtual_column_id_set = read_guard
                    .virtual_columns_by_table_index(scan.table_index)
                    .iter()
                    .map(|column| column.index())
                    .collect::<HashSet<_>>();
                for required_column_id in required_column_ids {
                    if virtual_column_id_set.contains(&required_column_id) {
                        columns.insert(required_column_id);
                    }
                }
            }

            let mut prewhere = scan.prewhere.clone();
            let mut used: ColumnSet = required.intersection(&columns).cloned().collect();

            // Secure predicates reference columns that must survive pruning,
            // even if they are not in the query output. Without this, a policy
            // on tenant_id would fail when the query only selects id.
            if let Some(secure_preds) = &scan.secure_predicates {
                for pred in secure_preds {
                    used = used.union(&pred.used_columns()).cloned().collect();
                }
            }

            let supported_lazy_materialize = {
                self.metadata
                    .read()
                    .table(scan.table_index)
                    .table()
                    .supported_lazy_materialize()
            };

            if scan.is_lazy_table && supported_lazy_materialize {
                let ordered_user_filters = if let Some(predicates) = scan
                    .push_down_predicates
                    .as_ref()
                    .filter(|preds| !preds.is_empty())
                {
                    let metadata = self.metadata.read().clone();
                    let predicates = predicates.iter().collect::<Vec<_>>();
                    self.create_scan_push_down_filters(&metadata, &predicates)?
                        .0
                } else {
                    None
                };
                let lazy_columns = if self
                    .try_build_ordered_index_info(
                        scan,
                        &self
                            .metadata
                            .read()
                            .table(scan.table_index)
                            .table()
                            .schema_with_stream(),
                        ordered_user_filters,
                    )?
                    .is_some()
                {
                    used = columns.clone();
                    ColumnSet::new()
                } else {
                    columns.difference(&used).cloned().collect()
                };
                let mut metadata = self.metadata.write();
                metadata.set_table_lazy_columns(scan.table_index, lazy_columns);
                for column_index in used.iter() {
                    metadata.add_retained_column(*column_index);
                }
            }
            if let Some(ref mut pw) = prewhere {
                debug_assert!(
                    pw.prewhere_columns.is_subset(&columns),
                    "prewhere columns should be a subset of scan columns"
                );
                pw.output_columns = used.clone();
                // `prune_columns` is after `prewhere_optimize`,
                // so we need to add prewhere columns to scan columns.
                used = used.union(&pw.prewhere_columns).cloned().collect();
            }
            scan.prune_columns(used, prewhere)
        };

        // 2. Build physical plan.
        let mut has_inner_column = false;
        let mut name_mapping = BTreeMap::new();
        let mut project_internal_columns: BTreeMap<FieldIndex, InternalColumn> = BTreeMap::new();
        let mut project_virtual_columns = BTreeMap::new();
        let metadata = self.metadata.read().clone();

        for index in scan.columns.iter() {
            if metadata.is_lazy_column(*index) {
                continue;
            }
            let column = metadata.column(*index);
            match column {
                ColumnEntry::BaseTableColumn(BaseTableColumn { path_indices, .. }) => {
                    if path_indices.is_some() {
                        has_inner_column = true;
                    }
                }
                ColumnEntry::InternalColumn(TableInternalColumn {
                    internal_column, ..
                }) => {
                    project_internal_columns.insert(index.as_usize(), internal_column.to_owned());
                }
                ColumnEntry::VirtualColumn(virtual_column) => {
                    project_virtual_columns.insert(*index, virtual_column.clone());
                }
                _ => {}
            }

            if let Some(prewhere) = &scan.prewhere {
                // if there is a prewhere optimization,
                // we can prune `PhysicalScan`'s output schema.
                if prewhere.output_columns.contains(index) {
                    name_mapping.insert(column.name().to_string(), index.to_string());
                }
            } else {
                name_mapping.insert(column.name().to_string(), index.to_string());
            }
        }

        if !name_mapping.contains_key(ROW_ID_COL_NAME) {
            let metadata = self.metadata.read();
            if metadata
                .get_table_lazy_columns(&scan.table_index)
                .is_some_and(|columns| !columns.is_empty())
                && let Some(index) = metadata.row_id_index_by_table_index(scan.table_index)
            {
                let internal_column = INTERNAL_COLUMN_FACTORY
                    .get_internal_column(ROW_ID_COL_NAME)
                    .unwrap();
                name_mapping.insert(ROW_ID_COL_NAME.to_string(), index.to_string());
                project_internal_columns.insert(index.as_usize(), internal_column);
            }
        }

        let table_entry = metadata.table(scan.table_index);
        let table = table_entry.table();

        if !table.result_can_be_cached() {
            self.ctx.result_cache_state().set_cacheable(false);
        }

        let mut table_schema = table.schema_with_stream();
        if !project_internal_columns.is_empty() {
            let mut schema = table_schema.as_ref().clone();
            for internal_column in project_internal_columns.values() {
                schema.add_internal_field(
                    internal_column.column_name(),
                    internal_column.table_data_type(),
                    internal_column.column_id(),
                );
            }
            table_schema = Arc::new(schema);
        }

        let push_downs = self.push_downs(
            &scan,
            &table_schema,
            project_virtual_columns,
            has_inner_column,
        )?;

        // Generate secure cache key extra for Row Access Policy predicates.
        // Constant-fold so session-dependent functions (e.g. GETVARIABLE)
        // resolve to concrete values. Include scan.table_index so that two
        // scans on different tables with identical policy text produce
        // distinct cache-key extras.
        if scan
            .secure_predicates
            .as_ref()
            .is_some_and(|p| !p.is_empty())
        {
            let metadata = self.metadata.read().clone();
            let secure_preds = scan.secure_predicates.as_deref().unwrap_or_default();
            let mut serialized: Vec<String> = Vec::with_capacity(secure_preds.len());
            for pred in secure_preds {
                let expr = pred
                    .as_raw_expr()
                    .type_check(&metadata)?
                    .project_column_ref(|col| Ok(col.column_name.clone()))?;
                let (folded, _) = ConstantFolder::fold(&expr, &self.func_ctx, &BUILTIN_FUNCTIONS);
                let remote = folded.as_remote_expr();
                serialized.push(serde_json::to_string(&remote).map_err(|e| {
                    ErrorCode::Internal(format!(
                        "Failed to serialize secure predicate for cache key: {}",
                        e
                    ))
                })?);
            }
            serialized.sort();
            let combined = format!("{}|{}", scan.table_index, serialized.join("|"));
            let hash = format!("{:x}", Sha256::digest(combined.as_bytes()));
            self.ctx
                .result_cache_state()
                .add_cache_key_extra(format!("secure:{}", hash));
        }

        let mut source = table
            .read_plan(
                self.ctx.clone(),
                Some(push_downs),
                if project_internal_columns.is_empty() {
                    None
                } else {
                    Some(project_internal_columns.clone())
                },
                scan.update_stream_columns,
                self.dry_run,
            )
            .await?;
        if let Some(ref sample) = scan.sample
            && !table.use_own_sample_block()
        {
            if let Some(block_sample_value) = sample.block_level {
                if block_sample_value > 100.0 {
                    return Err(ErrorCode::SyntaxException(format!(
                        "Sample value should be less than or equal to 100, but got {}",
                        block_sample_value
                    )));
                }
                let probability = block_sample_value / 100.0;
                let original_parts = source.parts.partitions.len();
                let mut sample_parts = Vec::with_capacity(original_parts);
                let mut rng = thread_rng();
                let bernoulli = Bernoulli::new(probability).unwrap();
                for part in source.parts.partitions.iter() {
                    if bernoulli.sample(&mut rng) {
                        sample_parts.push(part.clone());
                    }
                }
                source.parts.partitions = sample_parts;
            }
        }
        source.table_index = scan.table_index;
        source.scan_id = scan.scan_id;
        source.block_meta_options.reserve_block_index =
            need_reserve_block_info(self.ctx.clone(), scan.table_index).0;
        if let Some(agg_index) = &scan.agg_index {
            let source_schema = source.schema();
            let push_down = source.push_downs.as_mut().unwrap();
            let output_fields = TableScan::output_fields(source_schema, &name_mapping)?;
            let agg_index = Self::build_agg_index(agg_index, &output_fields)?;
            push_down.agg_index = Some(agg_index);
        }
        let internal_column = if project_internal_columns.is_empty() {
            None
        } else {
            Some(project_internal_columns)
        };

        if scan.is_lazy_table {
            let mut metadata = self.metadata.write();
            metadata.set_table_source(scan.table_index, source.clone());
        }

        let mut plan = TableScan::create(
            scan.scan_id,
            name_mapping,
            Box::new(source),
            Some(scan.table_index),
            Some(stat_info.clone()),
            internal_column,
        );

        // Update stream columns if needed.
        if scan.update_stream_columns {
            plan = AddStreamColumn::create(
                &self.metadata,
                plan,
                scan.table_index,
                table.get_table_info().ident.seq,
            )?;
        }

        if let Some(secure_preds) = &scan.secure_predicates {
            if !secure_preds.is_empty() && scan.has_secure_predicates_not_applied_by_prewhere() {
                let input_schema = plan.output_schema()?;
                let retained = self.metadata.read().get_retained_column().clone();
                let mut projections = BTreeSet::new();
                for col in required.union(&retained) {
                    if let Some((index, _)) = input_schema.column_with_name(&col.to_string()) {
                        projections.insert(index);
                    }
                }
                let predicates = secure_preds
                    .iter()
                    .map(|scalar| {
                        let expr = scalar
                            .as_raw_expr()
                            .type_check(&metadata)?
                            .project_column_ref(|col| {
                                input_schema.index_of(&col.index.to_string())
                            })?;
                        let expr = cast_expr_to_non_null_boolean(expr)?;
                        let (expr, _) =
                            ConstantFolder::fold(&expr, &self.func_ctx, &BUILTIN_FUNCTIONS);
                        Ok(expr.as_remote_expr())
                    })
                    .collect::<Result<Vec<_>>>()?;

                // After constant folding, skip the Filter if all predicates
                // folded to constant true (e.g. current_role() matched).
                let all_true = predicates.iter().all(|p| {
                    matches!(
                        p,
                        RemoteExpr::Constant { scalar, .. }
                            if scalar == &databend_common_expression::Scalar::Boolean(true)
                    )
                });
                if !all_true {
                    plan = PhysicalPlan::new(crate::physical_plans::Filter {
                        meta: PhysicalPlanMeta::new("Filter"),
                        projections,
                        input: plan,
                        predicates,
                        stat_info: Some(stat_info.clone()),
                        is_secure: true,
                    });
                }
            }
        }

        Ok(plan)
    }

    pub async fn build_dummy_table_scan(
        &mut self,
        dummy_scan: &databend_common_sql::plans::DummyTableScan,
    ) -> Result<PhysicalPlan> {
        let catalogs = CatalogManager::instance();
        let table = catalogs
            .get_default_catalog(self.ctx.session_state()?)?
            .get_table(&self.ctx.get_tenant(), "system", "one")
            .await?;

        // Add cache invalidation keys for DummyTableScan's source tables.
        //
        // When DummyTableScan is created by optimizations like count(*) folding, we need to
        // track which tables the result depends on for proper cache invalidation.
        //
        // Example problem without this fix:
        //   SELECT * FROM t1 WHERE a > (SELECT COUNT(*) FROM t2)
        //
        //   1. t1 has data, t2 is empty:
        //      - t1 (TableScan) adds t1's partition SHA
        //      - t2 (DummyTableScan, empty table) adds nothing if we skip empty tables
        //      - Cache writes: partitions_shas = [t1_sha]
        //   2. After inserting data into t2:
        //      - Cache check: [t1_sha] == [t1_sha] → cache hit → WRONG result!
        //
        // Solution: For FuseTables, use snapshot_location as the cache invalidation key.
        //
        // Why snapshot_location instead of partition SHA256?
        // - Simpler: No need to call read_partitions() (async I/O) and compute SHA
        // - Equivalent semantics: Any mutation (INSERT, UPDATE, DELETE, COMPACT, RECLUSTER)
        //   creates a new snapshot, so snapshot_location uniquely identifies table state
        // - snapshot_loc() is synchronous (reads from table metadata, no storage I/O)
        //   vs read_table_snapshot() which would need to fetch and deserialize the snapshot file
        let settings = self.ctx.get_settings();
        if settings.get_enable_query_result_cache()? && !dummy_scan.source_table_indexes.is_empty()
        {
            let metadata = self.metadata.read();
            for &idx in &dummy_scan.source_table_indexes {
                let source_table = metadata.table(idx).table();

                // Check if the source table supports result caching at all.
                if !source_table.result_can_be_cached() {
                    self.ctx.result_cache_state().set_cacheable(false);
                    break;
                }

                // Record a cache invalidation ID for DummyTableScan's source tables.
                // Only FuseTable provides query_result_cache_id; for other table engines
                // we conservatively disable caching to avoid returning stale results.
                if let Ok(fuse_table) = FuseTable::try_from_table(source_table.as_ref()) {
                    self.ctx
                        .result_cache_state()
                        .add_partitions_sha(fuse_table.query_result_cache_id());
                } else {
                    // Non-FuseTable (system table, memory table, etc.), disable caching.
                    self.ctx.result_cache_state().set_cacheable(false);
                    break;
                }
            }
        }

        let source = table
            .read_plan(self.ctx.clone(), None, None, false, self.dry_run)
            .await?;

        Ok(TableScan::create(
            DUMMY_TABLE_INDEX,
            BTreeMap::from([("dummy".to_string(), Symbol::DUMMY_COLUMN.to_string())]),
            Box::new(source),
            Some(DUMMY_TABLE_INDEX),
            Some(PlanStatsInfo {
                estimated_rows: 1.0,
            }),
            None,
        ))
    }

    fn push_downs(
        &self,
        scan: &databend_common_sql::plans::Scan,
        table_schema: &TableSchema,
        virtual_columns: BTreeMap<Symbol, VirtualColumn>,
        has_inner_column: bool,
    ) -> Result<PushDownInfo> {
        let metadata = self.metadata.read().clone();
        let projection = Self::build_projection(
            &metadata,
            table_schema,
            scan.columns.iter(),
            has_inner_column,
            // for projection, we need to ignore read data from internal column,
            // or else in read_partition when search internal column from table schema will core.
            true,
            true,
        );
        let has_virtual_column = !virtual_columns.is_empty();

        let output_columns = if has_virtual_column {
            Some(Self::build_projection(
                &metadata,
                table_schema,
                scan.columns.iter(),
                has_inner_column,
                true,
                false,
            ))
        } else {
            None
        };

        let user_predicates = scan.push_down_predicates.as_deref().unwrap_or_default();
        let secure_predicates = scan.secure_predicates.as_deref().unwrap_or_default();

        let (secure_filters, secure_is_deterministic) = if secure_predicates.is_empty() {
            (None, true)
        } else {
            let preds = secure_predicates.iter().collect::<Vec<_>>();
            self.create_scan_push_down_filters(&metadata, &preds)?
        };
        let (user_filters, user_is_deterministic) = if user_predicates.is_empty() {
            (None, true)
        } else {
            let preds = user_predicates.iter().collect::<Vec<_>>();
            self.create_scan_push_down_filters(&metadata, &preds)?
        };
        let is_deterministic = user_is_deterministic && secure_is_deterministic;

        let prewhere_info = scan
            .prewhere
            .as_ref()
            .map(|prewhere| -> Result<PrewhereInfo> {
                let remain_columns = scan
                    .columns
                    .difference(&prewhere.prewhere_columns)
                    .copied()
                    .collect::<HashSet<Symbol>>();

                let output_columns = Self::build_projection(
                    &metadata,
                    table_schema,
                    prewhere.output_columns.iter(),
                    has_inner_column,
                    true,
                    false,
                );
                let prewhere_columns = Self::build_projection(
                    &metadata,
                    table_schema,
                    prewhere.prewhere_columns.iter(),
                    has_inner_column,
                    true,
                    true,
                );
                let remain_columns = Self::build_projection(
                    &metadata,
                    table_schema,
                    remain_columns.iter(),
                    has_inner_column,
                    true,
                    true,
                );

                let predicate = prewhere
                    .predicates
                    .iter()
                    .cloned()
                    .reduce(|lhs, rhs| {
                        ScalarExpr::FunctionCall(FunctionCall {
                            span: None,
                            func_name: "and_filters".to_string(),
                            params: vec![],
                            arguments: vec![lhs, rhs],
                        })
                    })
                    .expect("there should be at least one predicate in prewhere");

                let filter = cast_expr_to_non_null_boolean(
                    predicate
                        .as_raw_expr()
                        .type_check(&metadata)?
                        .project_column_ref(|col| Ok(col.column_name.clone()))?,
                )?;
                let (filter, _) = ConstantFolder::fold(&filter, &self.func_ctx, &BUILTIN_FUNCTIONS);
                let filter = filter.as_remote_expr();
                let virtual_column_ids =
                    self.build_prewhere_virtual_column_ids(&prewhere.prewhere_columns);

                Ok::<PrewhereInfo, ErrorCode>(PrewhereInfo {
                    output_columns,
                    prewhere_columns,
                    remain_columns,
                    filter,
                    virtual_column_ids,
                })
            })
            .transpose()?;

        let order_by = scan.order_by.clone().map(|items| {
            items
                .into_iter()
                .filter_map(|item| {
                    let metadata = self.metadata.read();
                    let column = metadata.column(item.index);
                    let (name, data_type) = match column {
                        ColumnEntry::BaseTableColumn(BaseTableColumn {
                            column_name,
                            data_type,
                            ..
                        }) => (column_name.clone(), DataType::from(data_type)),
                        ColumnEntry::InternalColumn(TableInternalColumn {
                            internal_column,
                            ..
                        }) => (
                            internal_column.column_name().to_owned(),
                            internal_column.data_type(),
                        ),
                        ColumnEntry::VirtualColumn(_) | ColumnEntry::DerivedColumn(_) => {
                            return None;
                        }
                    };

                    // sort item is already a column
                    let scalar = RemoteExpr::ColumnRef {
                        span: None,
                        id: name.clone(),
                        data_type,
                        display_name: name,
                    };

                    Some((scalar, item.asc, item.nulls_first))
                })
                .collect::<Vec<_>>()
        });

        let order_by = order_by.unwrap_or_default();
        let mut limit = scan.limit;
        if let Some(scan_order_by) = &scan.order_by {
            // If some order by columns can't be pushed down, then the limit can't be pushed down either,
            // as this may cause some blocks are pruned by the limit pruner.
            if scan_order_by.len() != order_by.len() {
                limit = None;
            }
        }

        let virtual_column = self.build_virtual_column(virtual_columns)?;

        let ordered_index =
            self.try_build_ordered_index_info(scan, table_schema, user_filters.clone())?;
        if ordered_index
            .as_ref()
            .is_some_and(|ordered_index| ordered_index.limit.is_none())
        {
            // A filter-only ORDERED access path does not preserve the query ORDER BY,
            // so neither the source nor FUSE block pruning may consume the LIMIT.
            limit = None;
        }
        let prewhere = if ordered_index.is_some() {
            // The ORDERED source scans row payloads from the covered SST and applies the
            // complete pushed filter there, so the normal block-read prewhere path is
            // redundant and should not be mixed into the ORDERED read path.
            None
        } else {
            prewhere_info
        };

        Ok(PushDownInfo {
            projection: Some(projection),
            output_columns,
            filters: user_filters.clone(),
            is_deterministic,
            prewhere,
            limit,
            order_by,
            virtual_column,
            lazy_materialization: !metadata.lazy_columns().is_empty(),
            agg_index: None,
            change_type: scan.change_type.clone(),
            inverted_index: scan.inverted_index.clone(),
            vector_index: scan.vector_index.clone(),
            ordered_index,
            sample: scan.sample.clone(),
            read_partitions_pruning_mode: Default::default(),
            secure_filters,
        })
    }

    fn try_build_ordered_index_info(
        &self,
        scan: &databend_common_sql::plans::Scan,
        table_schema: &TableSchema,
        filters: Option<Filters>,
    ) -> Result<Option<OrderedIndexInfo>> {
        if scan
            .secure_predicates
            .as_ref()
            .is_some_and(|preds| !preds.is_empty())
            || scan.limit.is_none()
            || scan.push_down_predicates.as_ref().is_none_or(Vec::is_empty)
            || scan.order_by.as_ref().is_none_or(Vec::is_empty)
        {
            return Ok(None);
        }

        let metadata = self.metadata.read();
        let table_entry = metadata.table(scan.table_index);
        let table_info = table_entry.table().get_table_info().clone();
        let table_meta = &table_info.meta;
        let ordered_index_count = table_meta
            .indexes
            .values()
            .filter(|index| matches!(index.index_type, TableIndexType::Ordered))
            .count();
        let mut prefix_constants_by_column = HashMap::new();
        for predicate in scan.push_down_predicates.as_deref().unwrap_or_default() {
            collect_ordered_prefix_column_constants(predicate, &mut prefix_constants_by_column);
        }
        let filter_column_names = ordered_filter_column_names(filters.as_ref());
        let mut best_candidate: Option<(OrderedIndexCandidateScore, OrderedIndexInfo)> = None;

        for index in table_meta.indexes.values() {
            if !matches!(index.index_type, TableIndexType::Ordered) {
                continue;
            }

            let key_columns = if index.key_columns.is_empty() {
                index
                    .column_ids
                    .iter()
                    .map(|column_id| TableIndexColumn {
                        column_id: *column_id,
                        order: TableIndexColumnOrder::Asc,
                    })
                    .collect::<Vec<_>>()
            } else {
                index.key_columns.clone()
            };
            if key_columns.is_empty() {
                continue;
            }

            let Some(payload_fields) = ordered_payload_fields(
                table_meta,
                &key_columns,
                index.include_column_ids.as_slice(),
                is_ordered_covered_all_columns(&index.options),
            ) else {
                continue;
            };

            let mut prefix_values = Vec::new();
            let mut prefix_len = 0;
            for key_column in &key_columns {
                let Ok(field) = table_meta.schema.field_of_column_id(key_column.column_id) else {
                    prefix_values.clear();
                    break;
                };
                let Some(column_index) =
                    table_symbol_for_field(&metadata, scan.table_index, field.name())
                else {
                    prefix_values.clear();
                    break;
                };
                let Some(value) = prefix_constants_by_column.get(&column_index) else {
                    break;
                };
                prefix_values.push(value.clone());
                prefix_len += 1;
            }

            if prefix_values.is_empty() || prefix_len >= key_columns.len() {
                continue;
            }

            let order_by = scan.order_by.as_ref().unwrap();
            let preserves_order = ordered_index_matches_scan_order(
                table_meta,
                &metadata,
                scan.table_index,
                &key_columns,
                prefix_len,
                order_by,
            );
            let extra_filter_key_columns = ordered_extra_filter_key_column_count(
                table_meta,
                &key_columns,
                if preserves_order {
                    prefix_len + order_by.len()
                } else {
                    prefix_len
                },
                &filter_column_names,
            );
            if !preserves_order && extra_filter_key_columns == 0 {
                continue;
            }

            if !scan
                .columns
                .iter()
                .filter_map(|symbol| match metadata.column(*symbol) {
                    ColumnEntry::BaseTableColumn(BaseTableColumn { column_name, .. }) => {
                        Some(column_name.as_str())
                    }
                    _ => None,
                })
                .all(|column_name| {
                    payload_fields
                        .iter()
                        .any(|field| field.name() == column_name)
                })
            {
                continue;
            }
            if !ordered_filters_are_covered(filters.as_ref(), &payload_fields) {
                continue;
            }

            let mut ordered_key_columns = Vec::with_capacity(key_columns.len());
            let mut valid_key_fields = true;
            for key_column in &key_columns {
                let Ok(field) = table_meta.schema.field_of_column_id(key_column.column_id) else {
                    valid_key_fields = false;
                    break;
                };
                ordered_key_columns.push(OrderedIndexKeyColumn {
                    field: field.clone(),
                    order: to_ordered_pushdown_order(&key_column.order),
                });
            }
            if !valid_key_fields {
                continue;
            }

            let _ = table_schema;
            let ordered_index_info = OrderedIndexInfo {
                index_name: index.name.clone(),
                index_version: index.version.clone(),
                key_columns: ordered_key_columns,
                payload_fields,
                equality_prefix: prefix_values,
                limit: preserves_order.then_some(scan.limit).flatten(),
                filters: filters.clone(),
                use_block_ordered_index_size_hint: ordered_index_count == 1,
            };
            let score = OrderedIndexCandidateScore {
                prefix_len,
                extra_filter_key_columns,
                preserves_order,
                payload_width: ordered_index_info.payload_fields.len(),
                key_width: ordered_index_info.key_columns.len(),
            };
            if best_candidate
                .as_ref()
                .is_none_or(|(best_score, best_index)| {
                    ordered_candidate_score_is_better(
                        &score,
                        &ordered_index_info.index_name,
                        best_score,
                        &best_index.index_name,
                    )
                })
            {
                best_candidate = Some((score, ordered_index_info));
            }
        }

        Ok(best_candidate.map(|(_, index)| index))
    }

    fn create_scan_push_down_filters(
        &self,
        metadata: &Metadata,
        predicates: &[&ScalarExpr],
    ) -> Result<(Option<Filters>, bool)> {
        if predicates.is_empty() {
            return Ok((None, true));
        }

        let predicates = predicates
            .iter()
            .map(|p| {
                p.as_raw_expr()
                    .type_check(metadata)?
                    .project_column_ref(|col| Ok(col.column_name.clone()))
            })
            .collect::<Result<Vec<_>>>()?;

        let expr = predicates
            .into_iter()
            .try_reduce(|lhs, rhs| {
                check_function(None, "and_filters", &[], &[lhs, rhs], &BUILTIN_FUNCTIONS)
            })?
            .unwrap();

        let expr = cast_expr_to_non_null_boolean(expr)?;
        let (expr, _) = ConstantFolder::fold(&expr, &self.func_ctx, &BUILTIN_FUNCTIONS);

        let is_deterministic = expr.is_deterministic(&BUILTIN_FUNCTIONS);
        let inverted_filter =
            check_function(None, "not", &[], &[expr.clone()], &BUILTIN_FUNCTIONS)?;

        Ok((
            Some(Filters {
                filter: expr.as_remote_expr(),
                inverted_filter: inverted_filter.as_remote_expr(),
            }),
            is_deterministic,
        ))
    }

    fn build_prewhere_virtual_column_ids(&self, indices: &ColumnSet) -> Option<Vec<u32>> {
        let mut virtual_column_ids = Vec::new();
        for index in indices.iter() {
            if let ColumnEntry::VirtualColumn(virtual_column) = self.metadata.read().column(*index)
            {
                virtual_column_ids.push(virtual_column.column_id);
            }
        }
        if !virtual_column_ids.is_empty() {
            Some(virtual_column_ids)
        } else {
            None
        }
    }

    fn build_virtual_column(
        &self,
        virtual_columns: BTreeMap<Symbol, VirtualColumn>,
    ) -> Result<Option<VirtualColumnInfo>> {
        if virtual_columns.is_empty() {
            return Ok(None);
        }
        let mut source_column_ids = HashSet::new();
        let mut virtual_column_fields = Vec::with_capacity(virtual_columns.len());

        for (_, virtual_column) in virtual_columns.into_iter() {
            source_column_ids.insert(virtual_column.source_column_id);
            let target_type = virtual_column.data_type.remove_nullable();
            let cast_func_name = if target_type != TableDataType::Variant {
                Some(format!("to_{}", target_type.to_string().to_lowercase()))
            } else {
                None
            };

            let virtual_column_field = VirtualColumnField {
                source_column_id: virtual_column.source_column_id,
                source_name: virtual_column.source_column_name.clone(),
                column_id: virtual_column.column_id,
                name: virtual_column.column_name.clone(),
                key_paths: virtual_column.key_paths.clone(),
                cast_func_name,
                data_type: Box::new(virtual_column.data_type.clone()),
            };
            virtual_column_fields.push(virtual_column_field);
        }

        let virtual_column_info = VirtualColumnInfo {
            source_column_ids,
            virtual_column_fields,
        };
        Ok(Some(virtual_column_info))
    }

    pub fn build_agg_index(
        agg: &databend_common_sql::plans::AggIndexInfo,
        source_fields: &[DataField],
    ) -> Result<databend_common_catalog::plan::AggIndexInfo> {
        // Build projection
        let used_columns = agg.used_columns();
        let mut col_indices = Vec::with_capacity(used_columns.len());
        for index in used_columns.iter() {
            col_indices.push(agg.schema.index_of(&index.to_string())?);
        }
        let projection = Projection::Columns(col_indices);
        let output_schema = projection.project_schema(&agg.schema);

        let predicate = agg.predicates.iter().cloned().reduce(|lhs, rhs| {
            ScalarExpr::FunctionCall(FunctionCall {
                span: None,
                func_name: "and".to_string(),
                params: vec![],
                arguments: vec![lhs, rhs],
            })
        });
        let filter = predicate
            .map(|pred| -> Result<_> {
                Ok(cast_expr_to_non_null_boolean(
                    pred.as_expr()?
                        .project_column_ref(|col| output_schema.index_of(&col.index.to_string()))?,
                )?
                .as_remote_expr())
            })
            .transpose()?;
        let selection = agg
            .selection
            .iter()
            .map(|sel| {
                let offset = source_fields
                    .iter()
                    .position(|f| sel.index.to_string() == f.name().as_str());
                Ok((
                    sel.scalar
                        .as_expr()?
                        .project_column_ref(|col| output_schema.index_of(&col.index.to_string()))?
                        .as_remote_expr(),
                    offset,
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(databend_common_catalog::plan::AggIndexInfo {
            index_id: agg.index_id,
            filter,
            selection,
            schema: agg.schema.clone(),
            actual_table_field_len: source_fields.len(),
            is_agg: agg.is_agg,
            projection,
            num_agg_funcs: agg.num_agg_funcs,
        })
    }

    pub fn build_projection<'a>(
        metadata: &Metadata,
        schema: &TableSchema,
        columns: impl Iterator<Item = &'a Symbol>,
        has_inner_column: bool,
        ignore_internal_column: bool,
        add_virtual_source_column: bool,
    ) -> Projection {
        if !has_inner_column {
            let mut col_indices = Vec::new();
            let mut virtual_col_indices = HashSet::new();
            for index in columns {
                let name = match metadata.column(*index) {
                    ColumnEntry::BaseTableColumn(BaseTableColumn { column_name, .. }) => {
                        column_name
                    }
                    ColumnEntry::DerivedColumn(DerivedColumn { alias, .. }) => alias,
                    ColumnEntry::InternalColumn(TableInternalColumn {
                        internal_column, ..
                    }) => {
                        if ignore_internal_column {
                            continue;
                        }
                        internal_column.column_name()
                    }
                    ColumnEntry::VirtualColumn(VirtualColumn {
                        source_column_name, ..
                    }) => {
                        if add_virtual_source_column {
                            virtual_col_indices
                                .insert(schema.index_of(source_column_name).unwrap());
                        }
                        continue;
                    }
                };
                col_indices.push(schema.index_of(name).unwrap());
            }
            if !virtual_col_indices.is_empty() {
                for index in virtual_col_indices {
                    if !col_indices.contains(&index) {
                        col_indices.push(index);
                    }
                }
            }
            col_indices.sort();
            Projection::Columns(col_indices)
        } else {
            let mut col_indices = BTreeMap::new();
            for index in columns {
                let column = metadata.column(*index);
                match column {
                    ColumnEntry::BaseTableColumn(BaseTableColumn {
                        column_name,
                        path_indices,
                        ..
                    }) => match path_indices {
                        Some(path_indices) => {
                            col_indices.insert(column.index().as_usize(), path_indices.to_vec());
                        }
                        None => {
                            let idx = schema.index_of(column_name).unwrap();
                            col_indices.insert(column.index().as_usize(), vec![idx]);
                        }
                    },
                    ColumnEntry::DerivedColumn(DerivedColumn { alias, .. }) => {
                        let idx = schema.index_of(alias).unwrap();
                        col_indices.insert(column.index().as_usize(), vec![idx]);
                    }
                    ColumnEntry::InternalColumn(TableInternalColumn { column_index, .. }) => {
                        if !ignore_internal_column {
                            col_indices
                                .insert(column_index.as_usize(), vec![column_index.as_usize()]);
                        }
                    }
                    ColumnEntry::VirtualColumn(VirtualColumn {
                        source_column_name, ..
                    }) => {
                        if add_virtual_source_column {
                            let idx = schema.index_of(source_column_name).unwrap();
                            col_indices.insert(idx, vec![idx]);
                        }
                    }
                }
            }
            Projection::InnerColumns(col_indices)
        }
    }
}

fn is_ordered_covered_all_columns(options: &BTreeMap<String, String>) -> bool {
    options
        .get(ORDERED_INDEX_OPTION_COVERED_TYPE)
        .is_some_and(|value| value.eq_ignore_ascii_case(ORDERED_INDEX_COVERED_ALL_COLUMNS))
}

fn ordered_payload_fields(
    table_meta: &databend_common_meta_app::schema::TableMeta,
    key_columns: &[TableIndexColumn],
    include_column_ids: &[u32],
    covered_all_columns: bool,
) -> Option<Vec<databend_common_expression::TableField>> {
    if covered_all_columns {
        return Some(
            table_meta
                .schema
                .remove_virtual_computed_fields()
                .fields
                .clone(),
        );
    }

    let mut fields = Vec::with_capacity(key_columns.len() + include_column_ids.len());
    let mut seen_column_ids = BTreeSet::new();
    for key_column in key_columns {
        if !seen_column_ids.insert(key_column.column_id) {
            continue;
        }
        let Ok(field) = table_meta.schema.field_of_column_id(key_column.column_id) else {
            return None;
        };
        fields.push(field.clone());
    }
    for column_id in include_column_ids {
        if !seen_column_ids.insert(*column_id) {
            continue;
        }
        let Ok(field) = table_meta.schema.field_of_column_id(*column_id) else {
            return None;
        };
        fields.push(field.clone());
    }
    Some(fields)
}

fn ordered_filters_are_covered(
    filters: Option<&Filters>,
    payload_fields: &[databend_common_expression::TableField],
) -> bool {
    let Some(filters) = filters else {
        return true;
    };
    let covered_columns = payload_fields
        .iter()
        .map(|field| field.name())
        .collect::<HashSet<_>>();
    filters
        .filter
        .as_expr(&BUILTIN_FUNCTIONS)
        .column_refs()
        .keys()
        .all(|name| covered_columns.contains(name))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct OrderedIndexCandidateScore {
    prefix_len: usize,
    extra_filter_key_columns: usize,
    preserves_order: bool,
    payload_width: usize,
    key_width: usize,
}

fn ordered_filter_column_names(filters: Option<&Filters>) -> HashSet<String> {
    filters
        .map(|filters| {
            filters
                .filter
                .as_expr(&BUILTIN_FUNCTIONS)
                .column_refs()
                .keys()
                .map(|name| name.to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn ordered_extra_filter_key_column_count(
    table_meta: &databend_common_meta_app::schema::TableMeta,
    key_columns: &[TableIndexColumn],
    first_extra_key_index: usize,
    filter_column_names: &HashSet<String>,
) -> usize {
    if filter_column_names.is_empty() {
        return 0;
    }

    key_columns
        .iter()
        .skip(first_extra_key_index)
        .filter_map(|key_column| {
            table_meta
                .schema
                .field_of_column_id(key_column.column_id)
                .ok()
        })
        .filter(|field| filter_column_names.contains(field.name()))
        .count()
}

fn ordered_index_matches_scan_order(
    table_meta: &databend_common_meta_app::schema::TableMeta,
    metadata: &Metadata,
    table_index: IndexType,
    key_columns: &[TableIndexColumn],
    prefix_len: usize,
    order_by: &[SortItem],
) -> bool {
    if prefix_len + order_by.len() > key_columns.len() {
        return false;
    }

    for (offset, order_item) in order_by.iter().enumerate() {
        let key_column = &key_columns[prefix_len + offset];
        let Ok(field) = table_meta.schema.field_of_column_id(key_column.column_id) else {
            return false;
        };
        let Some(symbol) = table_symbol_for_field(metadata, table_index, field.name()) else {
            return false;
        };
        if symbol != order_item.index
            || key_order_is_asc(&key_column.order) != order_item.asc
            || order_item.nulls_first
        {
            return false;
        }
    }

    true
}

fn ordered_candidate_score_is_better(
    candidate: &OrderedIndexCandidateScore,
    candidate_name: &str,
    best: &OrderedIndexCandidateScore,
    best_name: &str,
) -> bool {
    if candidate.prefix_len != best.prefix_len {
        return candidate.prefix_len > best.prefix_len;
    }
    if candidate.preserves_order != best.preserves_order {
        return candidate.preserves_order;
    }
    if candidate.extra_filter_key_columns != best.extra_filter_key_columns {
        return candidate.extra_filter_key_columns > best.extra_filter_key_columns;
    }
    if candidate.payload_width != best.payload_width {
        return candidate.payload_width < best.payload_width;
    }
    if candidate.key_width != best.key_width {
        return candidate.key_width < best.key_width;
    }
    candidate_name < best_name
}

fn key_order_is_asc(order: &TableIndexColumnOrder) -> bool {
    matches!(order, TableIndexColumnOrder::Asc)
}

fn to_ordered_pushdown_order(order: &TableIndexColumnOrder) -> OrderedIndexColumnOrder {
    match order {
        TableIndexColumnOrder::Asc => OrderedIndexColumnOrder::Asc,
        TableIndexColumnOrder::Desc => OrderedIndexColumnOrder::Desc,
    }
}

fn table_symbol_for_field(
    metadata: &Metadata,
    table_index: IndexType,
    name: &str,
) -> Option<Symbol> {
    metadata.columns().iter().find_map(|entry| match entry {
        ColumnEntry::BaseTableColumn(BaseTableColumn {
            table_index: column_table_index,
            column_name,
            column_index,
            path_indices,
            ..
        }) if *column_table_index == table_index
            && path_indices.is_none()
            && column_name == name =>
        {
            Some(*column_index)
        }
        _ => None,
    })
}

fn collect_ordered_prefix_column_constants(
    expr: &ScalarExpr,
    constants_by_column: &mut HashMap<Symbol, Scalar>,
) {
    if let ScalarExpr::FunctionCall(func) = expr
        && matches!(func.func_name.as_str(), "and" | "and_filters")
    {
        for argument in &func.arguments {
            collect_ordered_prefix_column_constants(argument, constants_by_column);
        }
        return;
    }

    if let Some((column, value)) = extract_ordered_prefix_column_constant(expr) {
        constants_by_column.insert(column, value);
    }
}

fn extract_ordered_prefix_column_constant(expr: &ScalarExpr) -> Option<(Symbol, Scalar)> {
    if let Some(column) = extract_is_null_column(expr) {
        return Some((column, Scalar::Null));
    }

    let ScalarExpr::FunctionCall(func) = expr else {
        return None;
    };
    if func.func_name != "eq" || func.arguments.len() != 2 {
        return None;
    }
    match (&func.arguments[0], &func.arguments[1]) {
        (ScalarExpr::BoundColumnRef(column), ScalarExpr::ConstantExpr(constant))
        | (ScalarExpr::BoundColumnRef(column), ScalarExpr::TypedConstantExpr(constant, _)) => {
            Some((column.column.index, constant.value.clone()))
        }
        (ScalarExpr::ConstantExpr(constant), ScalarExpr::BoundColumnRef(column))
        | (ScalarExpr::TypedConstantExpr(constant, _), ScalarExpr::BoundColumnRef(column)) => {
            Some((column.column.index, constant.value.clone()))
        }
        _ => None,
    }
}

fn extract_is_null_column(expr: &ScalarExpr) -> Option<Symbol> {
    let ScalarExpr::FunctionCall(func) = expr else {
        return None;
    };
    if func.func_name != "not" || func.arguments.len() != 1 {
        return None;
    }

    let ScalarExpr::FunctionCall(inner) = &func.arguments[0] else {
        return None;
    };
    if inner.func_name != "is_not_null" || inner.arguments.len() != 1 {
        return None;
    }

    let ScalarExpr::BoundColumnRef(column) = &inner.arguments[0] else {
        return None;
    };
    Some(column.column.index)
}

#[cfg(test)]
mod tests {
    use databend_common_expression::types::DataType;
    use databend_common_expression::types::NumberDataType;
    use databend_common_expression::types::NumberScalar;
    use databend_common_sql::ColumnBindingBuilder;
    use databend_common_sql::Visibility;
    use databend_common_sql::plans::BoundColumnRef;
    use databend_common_sql::plans::ConstantExpr;

    use super::*;

    #[test]
    fn test_ordered_prefix_constants_collect_eq_and_is_null() {
        let platform = Symbol::new(0);
        let tag = Symbol::new(1);
        let expr = and(
            eq(
                test_column("platform_id", platform),
                ConstantExpr {
                    span: None,
                    value: Scalar::Number(NumberScalar::Int64(14)),
                }
                .into(),
            ),
            is_null(test_column("tag_sniper", tag)),
        );

        let mut constants_by_column = HashMap::new();
        collect_ordered_prefix_column_constants(&expr, &mut constants_by_column);

        assert_eq!(
            constants_by_column.get(&platform),
            Some(&Scalar::Number(NumberScalar::Int64(14)))
        );
        assert_eq!(constants_by_column.get(&tag), Some(&Scalar::Null));
    }

    #[test]
    fn test_ordered_prefix_constant_extracts_reversed_eq() {
        let wallet = Symbol::new(0);
        let expr = eq(
            ConstantExpr {
                span: None,
                value: Scalar::String("wallet-a".to_string()),
            }
            .into(),
            test_column("wallet_address", wallet),
        );

        assert_eq!(
            extract_ordered_prefix_column_constant(&expr),
            Some((wallet, Scalar::String("wallet-a".to_string())))
        );
    }

    #[test]
    fn test_ordered_prefix_constant_ignores_is_not_null() {
        let tag = Symbol::new(0);
        assert_eq!(
            extract_ordered_prefix_column_constant(&is_not_null(test_column("tag_sniper", tag))),
            None
        );
    }

    #[test]
    fn test_ordered_candidate_score_prefers_longer_prefix() {
        let shorter_payload = OrderedIndexCandidateScore {
            prefix_len: 2,
            extra_filter_key_columns: 0,
            preserves_order: true,
            payload_width: 3,
            key_width: 3,
        };
        let longer_prefix = OrderedIndexCandidateScore {
            prefix_len: 3,
            extra_filter_key_columns: 0,
            preserves_order: false,
            payload_width: 20,
            key_width: 4,
        };

        assert!(ordered_candidate_score_is_better(
            &longer_prefix,
            "idx_longer_prefix",
            &shorter_payload,
            "idx_shorter_payload"
        ));
    }

    #[test]
    fn test_ordered_candidate_score_prefers_order_preserving_before_extra_filter_keys() {
        let no_extra_filter_keys = OrderedIndexCandidateScore {
            prefix_len: 2,
            extra_filter_key_columns: 0,
            preserves_order: true,
            payload_width: 3,
            key_width: 3,
        };
        let with_extra_filter_keys = OrderedIndexCandidateScore {
            prefix_len: 2,
            extra_filter_key_columns: 2,
            preserves_order: false,
            payload_width: 10,
            key_width: 5,
        };

        assert!(ordered_candidate_score_is_better(
            &no_extra_filter_keys,
            "idx_plain",
            &with_extra_filter_keys,
            "idx_with_tags"
        ));
    }

    #[test]
    fn test_ordered_candidate_score_prefers_extra_filter_key_columns_on_order_tie() {
        let no_extra_filter_keys = OrderedIndexCandidateScore {
            prefix_len: 2,
            extra_filter_key_columns: 0,
            preserves_order: true,
            payload_width: 3,
            key_width: 3,
        };
        let with_extra_filter_keys = OrderedIndexCandidateScore {
            prefix_len: 2,
            extra_filter_key_columns: 2,
            preserves_order: true,
            payload_width: 10,
            key_width: 5,
        };

        assert!(ordered_candidate_score_is_better(
            &with_extra_filter_keys,
            "idx_with_tags",
            &no_extra_filter_keys,
            "idx_plain"
        ));
    }

    #[test]
    fn test_ordered_candidate_score_prefers_order_preserving_on_tie() {
        let filter_only = OrderedIndexCandidateScore {
            prefix_len: 2,
            extra_filter_key_columns: 1,
            preserves_order: false,
            payload_width: 3,
            key_width: 4,
        };
        let ordered = OrderedIndexCandidateScore {
            prefix_len: 2,
            extra_filter_key_columns: 1,
            preserves_order: true,
            payload_width: 20,
            key_width: 5,
        };

        assert!(ordered_candidate_score_is_better(
            &ordered,
            "idx_ordered",
            &filter_only,
            "idx_filter_only"
        ));
    }

    #[test]
    fn test_ordered_candidate_score_prefers_narrower_payload_on_tie() {
        let wide_payload = OrderedIndexCandidateScore {
            prefix_len: 2,
            extra_filter_key_columns: 1,
            preserves_order: true,
            payload_width: 20,
            key_width: 5,
        };
        let narrow_payload = OrderedIndexCandidateScore {
            prefix_len: 2,
            extra_filter_key_columns: 1,
            preserves_order: true,
            payload_width: 8,
            key_width: 5,
        };

        assert!(ordered_candidate_score_is_better(
            &narrow_payload,
            "idx_narrow",
            &wide_payload,
            "idx_wide"
        ));
    }

    fn test_column(name: &str, index: Symbol) -> ScalarExpr {
        BoundColumnRef {
            span: None,
            column: ColumnBindingBuilder::new(
                name.to_string(),
                index,
                Box::new(DataType::Number(NumberDataType::Int64).wrap_nullable()),
                Visibility::Visible,
            )
            .build(),
        }
        .into()
    }

    fn and(left: ScalarExpr, right: ScalarExpr) -> ScalarExpr {
        FunctionCall {
            span: None,
            func_name: "and_filters".to_string(),
            params: vec![],
            arguments: vec![left, right],
        }
        .into()
    }

    fn eq(left: ScalarExpr, right: ScalarExpr) -> ScalarExpr {
        FunctionCall {
            span: None,
            func_name: "eq".to_string(),
            params: vec![],
            arguments: vec![left, right],
        }
        .into()
    }

    fn is_null(expr: ScalarExpr) -> ScalarExpr {
        FunctionCall {
            span: None,
            func_name: "not".to_string(),
            params: vec![],
            arguments: vec![is_not_null(expr)],
        }
        .into()
    }

    fn is_not_null(expr: ScalarExpr) -> ScalarExpr {
        FunctionCall {
            span: None,
            func_name: "is_not_null".to_string(),
            params: vec![],
            arguments: vec![expr],
        }
        .into()
    }
}
