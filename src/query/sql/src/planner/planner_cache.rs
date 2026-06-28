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
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::LazyLock;

use databend_common_ast::ast::FunctionCall;
use databend_common_ast::ast::Identifier;
use databend_common_ast::ast::IdentifierType;
use databend_common_ast::ast::Statement;
use databend_common_ast::ast::TableReference;
use databend_common_catalog::table::Table;
use databend_common_catalog::table_context::TableContext;
use databend_common_exception::Result;
use databend_common_expression::ColumnId;
use databend_common_expression::Scalar;
use databend_common_expression::TableSchemaRef;
use databend_common_functions::is_cacheable_function;
use databend_common_meta_app::schema::SecurityPolicyColumnMap;
use databend_common_meta_app::schema::TableMeta;
use databend_common_settings::ChangeValue;
use databend_meta_client::types::MetaId;
use databend_storages_common_cache::CacheAccessor;
use databend_storages_common_cache::CacheValue;
use databend_storages_common_cache::InMemoryLruCache;
use databend_storages_common_table_meta::table::OPT_KEY_SNAPSHOT_LOCATION;
use derive_visitor::Drive;
use derive_visitor::Visitor;
use itertools::Itertools;
use sha2::Digest;
use sha2::Sha256;

use crate::NameResolutionContext;
use crate::Planner;
use crate::TableEntry;
use crate::normalize_identifier;
use crate::plans::Plan;

#[derive(Clone)]
pub struct PlanCacheItem {
    pub(crate) plan: Plan,
    table_snapshots: Vec<TableSnapshot>,
    pub(crate) setting_changes: Vec<(String, ChangeValue)>,
    pub(crate) variables: HashMap<String, Scalar>,
}

static PLAN_CACHE: LazyLock<InMemoryLruCache<PlanCacheItem>> =
    LazyLock::new(|| InMemoryLruCache::with_items_capacity("planner_cache".to_string(), 512));

impl From<PlanCacheItem> for CacheValue<PlanCacheItem> {
    fn from(val: PlanCacheItem) -> Self {
        CacheValue::new(val, 1024)
    }
}

impl Planner {
    pub(crate) fn planner_cache_key(format_sql: &str) -> String {
        // use sha2 to encode the sql
        format!("{:x}", Sha256::digest(format_sql))
    }

    pub(crate) fn build_plan_cache_context(
        &self,
        name_resolution_ctx: NameResolutionContext,
        stmt: &Statement,
    ) -> Result<Option<PlanCacheContext>> {
        if !matches!(stmt, Statement::Query(_))
            || !self.ctx.get_settings().get_enable_planner_cache()?
        {
            return Ok(None);
        }

        let mut visitor = TableRefVisitor {
            ctx: self.ctx.clone(),
            table_snapshots: vec![],
            name_resolution_ctx,
            cache_miss: false,
            has_security_policy: false,
        };
        stmt.drive(&mut visitor);

        if visitor.cache_miss || visitor.table_snapshots.is_empty() {
            return Ok(None);
        }

        let cache_key = self.planner_cache_key_for_stmt(stmt, visitor.has_security_policy)?;
        Ok(Some(PlanCacheContext {
            cache_key,
            table_snapshots: visitor.table_snapshots,
        }))
    }

    pub(crate) async fn get_cache_for_stmt(
        &self,
        stmt: &Statement,
    ) -> Result<Option<PlanCacheItem>> {
        if !matches!(stmt, Statement::Query(_))
            || !self.ctx.get_settings().get_enable_planner_cache()?
        {
            return Ok(None);
        }

        let plain_key = self.planner_cache_key_for_stmt(stmt, false)?;
        if let Some(item) = self.get_cache_by_key(&plain_key).await? {
            return Ok(Some(item));
        }

        Ok(None)
    }

    fn planner_cache_key_for_stmt(
        &self,
        stmt: &Statement,
        has_security_policy: bool,
    ) -> Result<String> {
        let context_prefix = self.planner_cache_context_prefix();
        if has_security_policy {
            return Ok(Self::planner_cache_key(&format!(
                "{}\0{}\0{}",
                context_prefix,
                self.security_policy_cache_key_prefix()?,
                stmt
            )));
        }

        Ok(Self::planner_cache_key(&format!(
            "{context_prefix}\0{stmt}"
        )))
    }

    fn planner_cache_context_prefix(&self) -> String {
        format!(
            "ctx\0{}\0{}\0{}",
            self.ctx.get_tenant().tenant_name(),
            self.ctx.get_current_catalog(),
            self.ctx.get_current_database()
        )
    }

    fn security_policy_cache_key_prefix(&self) -> Result<String> {
        let user = self
            .ctx
            .get_current_user()?
            .identity()
            .display()
            .to_string();
        let role = self
            .ctx
            .get_current_role()
            .map(|r| r.name)
            .unwrap_or_default();

        let mut secondary_roles = self.ctx.get_secondary_roles();
        if let Some(roles) = &mut secondary_roles {
            roles.sort();
        }

        Ok(format!(
            "secure\0{}\0{}\0{}",
            user,
            role,
            Self::secondary_roles_cache_key(secondary_roles.as_deref()),
        ))
    }

    fn secondary_roles_cache_key(secondary_roles: Option<&[String]>) -> String {
        match secondary_roles {
            None => "ALL".to_string(),
            Some([]) => "NONE".to_string(),
            Some(roles) => format!("SOME:{}", roles.join(",")),
        }
    }

    pub(crate) fn get_cache(&self, cache_ctx: &PlanCacheContext) -> Option<PlanCacheItem> {
        debug_assert!(!cache_ctx.table_snapshots.is_empty());

        let cache = LazyLock::force(&PLAN_CACHE);
        let plan_item = cache.get(&cache_ctx.cache_key)?;

        if !self.plan_cache_item_matches_env(plan_item.as_ref()) {
            return None;
        }

        let Plan::Query { metadata, .. } = &plan_item.plan else {
            return None;
        };

        let metadata = metadata.read();
        cache_ctx
            .matches_metadata_tables(metadata.tables())
            .then(|| plan_item.as_ref().clone())
    }

    async fn get_cache_by_key(&self, cache_key: &str) -> Result<Option<PlanCacheItem>> {
        let cache = LazyLock::force(&PLAN_CACHE);
        let Some(plan_item) = cache.get(cache_key) else {
            return Ok(None);
        };
        let plan_item = plan_item.as_ref();

        if !self.plan_cache_item_matches_env(plan_item) {
            return Ok(None);
        }

        let Plan::Query { metadata, .. } = &plan_item.plan else {
            return Ok(None);
        };

        let matches_cached_metadata = {
            let metadata = metadata.read();
            PlanCacheContext::matches_snapshots_with_metadata_tables(
                &plan_item.table_snapshots,
                metadata.tables(),
            )
        };
        if !matches_cached_metadata {
            return Ok(None);
        }

        if !self
            .matches_current_tables(&plan_item.table_snapshots)
            .await?
        {
            return Ok(None);
        }

        Ok(Some(plan_item.clone()))
    }

    fn plan_cache_item_matches_env(&self, plan_item: &PlanCacheItem) -> bool {
        let settings = self.ctx.get_settings();
        settings.changes().len() == plan_item.setting_changes.len()
            && self.setting_changes() == plan_item.setting_changes
            && self.ctx.get_all_variables() == plan_item.variables
    }

    async fn matches_current_tables(&self, snapshots: &[TableSnapshot]) -> Result<bool> {
        if snapshots.is_empty() {
            return Ok(false);
        }

        for snapshot in snapshots {
            let Ok(catalog) = self.ctx.get_catalog(&snapshot.catalog_name).await else {
                return Ok(false);
            };

            let Some(table_meta) = catalog.get_table_meta_by_id(snapshot.table_id).await? else {
                return Ok(false);
            };

            if !snapshot.matches_table_meta(table_meta.seq, &table_meta.data) {
                return Ok(false);
            }

            let Ok(current_table) = catalog
                .get_table(
                    &self.ctx.get_tenant(),
                    &snapshot.database_name,
                    &snapshot.table_name,
                )
                .await
            else {
                return Ok(false);
            };

            if !snapshot.matches_table(current_table.as_ref()) {
                return Ok(false);
            }
        }

        Ok(true)
    }

    pub(crate) fn set_cache(&self, cache_ctx: PlanCacheContext, plan: Plan) {
        let plan_item = PlanCacheItem {
            plan,
            table_snapshots: cache_ctx.table_snapshots,
            setting_changes: self.setting_changes(),
            variables: self.ctx.get_all_variables(),
        };
        let cache = LazyLock::force(&PLAN_CACHE);
        cache.insert(cache_ctx.cache_key, plan_item);
    }

    fn setting_changes(&self) -> Vec<(String, ChangeValue)> {
        self.ctx
            .get_settings()
            .changes()
            .iter()
            .map(|s| (s.key().clone(), s.value().clone()))
            .sorted_by(|a, b| Ord::cmp(&a.0, &b.0))
            .collect()
    }
}

#[derive(Visitor)]
#[visitor(TableReference(enter), FunctionCall(enter))]
struct TableRefVisitor {
    ctx: Arc<dyn TableContext>,
    table_snapshots: Vec<TableSnapshot>,
    name_resolution_ctx: NameResolutionContext,
    cache_miss: bool,
    has_security_policy: bool,
}

pub(crate) struct PlanCacheContext {
    cache_key: String,
    table_snapshots: Vec<TableSnapshot>,
}

impl PlanCacheContext {
    fn matches_metadata_tables(&self, tables: &[TableEntry]) -> bool {
        Self::matches_snapshots_with_metadata_tables(&self.table_snapshots, tables)
    }

    fn matches_snapshots_with_metadata_tables(
        table_snapshots: &[TableSnapshot],
        tables: &[TableEntry],
    ) -> bool {
        table_snapshots.iter().all(|snapshot| {
            tables
                .iter()
                .any(|table| snapshot.matches_table_entry(table))
        })
    }
}

#[derive(Clone)]
struct TableSnapshot {
    catalog_name: String,
    database_name: String,
    table_name: String,
    table_id: MetaId,
    table_seq: u64,
    schema: TableSchemaRef,
    snapshot_location: String,
    security_policy: SecurityPolicySnapshot,
}

impl TableSnapshot {
    fn from_resolved_table(
        table: &dyn Table,
        catalog_name: String,
        database_name: String,
        table_name: String,
    ) -> Option<Self> {
        if table.is_temp() || table.is_stage_table() || table.is_stream() {
            return None;
        }

        let table_info = table.get_table_info();
        let snapshot_location = table.options().get(OPT_KEY_SNAPSHOT_LOCATION)?.clone();
        Some(Self {
            catalog_name,
            database_name,
            table_name,
            table_id: table_info.ident.table_id,
            table_seq: table_info.ident.seq,
            schema: table.schema(),
            snapshot_location,
            security_policy: SecurityPolicySnapshot::from(&table_info.meta),
        })
    }

    fn has_security_policy(&self) -> bool {
        self.security_policy.has_policy()
    }

    fn matches_table_entry(&self, table_entry: &TableEntry) -> bool {
        let table = table_entry.table();
        self.matches_table(table.as_ref())
    }

    fn matches_table(&self, table: &dyn Table) -> bool {
        let table_info = table.get_table_info();
        if table.is_temp()
            || table_info.catalog() != self.catalog_name
            || table_info.database_name().ok() != Some(self.database_name.as_str())
            || table_info.name != self.table_name
            || table_info.ident.table_id != self.table_id
            || table_info.ident.seq != self.table_seq
            || table.schema().ne(&self.schema)
        {
            return false;
        }

        self.matches_table_meta(table_info.ident.seq, &table_info.meta)
    }

    fn matches_table_meta(&self, table_seq: u64, meta: &TableMeta) -> bool {
        table_seq == self.table_seq
            && meta.schema.eq(&self.schema)
            && meta.options.get(OPT_KEY_SNAPSHOT_LOCATION) == Some(&self.snapshot_location)
            && SecurityPolicySnapshot::from(meta) == self.security_policy
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SecurityPolicySnapshot {
    column_mask_policy_columns_ids: BTreeMap<ColumnId, SecurityPolicyColumnMap>,
    row_access_policy_columns_ids: Option<SecurityPolicyColumnMap>,
}

impl SecurityPolicySnapshot {
    fn has_policy(&self) -> bool {
        !self.column_mask_policy_columns_ids.is_empty()
            || self.row_access_policy_columns_ids.is_some()
    }
}

impl From<&TableMeta> for SecurityPolicySnapshot {
    fn from(meta: &TableMeta) -> Self {
        Self {
            column_mask_policy_columns_ids: meta.column_mask_policy_columns_ids.clone(),
            row_access_policy_columns_ids: meta.row_access_policy_columns_ids.clone(),
        }
    }
}

impl TableRefVisitor {
    fn enter_function_call(&mut self, func: &FunctionCall) {
        if self.cache_miss {
            return;
        }

        let func_name = func.name.name.to_lowercase();
        // If the function is not suitable for caching, we should not cache the plan
        if !is_cacheable_function(&func_name) {
            self.cache_miss = true;
        }
    }

    fn enter_table_reference(&mut self, table_ref: &TableReference) {
        if self.cache_miss {
            return;
        }
        if let TableReference::Table {
            table,
            temporal,
            with_options,
            ..
        } = table_ref
        {
            if temporal.is_some() || with_options.is_some() {
                self.cache_miss = true;
                return;
            }

            let catalog = table.catalog.to_owned().unwrap_or(Identifier {
                span: None,
                name: self.ctx.get_current_catalog(),
                quote: None,
                ident_type: IdentifierType::None,
            });
            let database = table.database.to_owned().unwrap_or(Identifier {
                span: None,
                name: self.ctx.get_current_database(),
                quote: None,
                ident_type: IdentifierType::None,
            });

            let catalog_name = normalize_identifier(&catalog, &self.name_resolution_ctx).name;
            let database_name = normalize_identifier(&database, &self.name_resolution_ctx).name;
            let table_name = normalize_identifier(&table.table, &self.name_resolution_ctx).name;
            let branch = table
                .branch
                .as_ref()
                .map(|v| normalize_identifier(v, &self.name_resolution_ctx).name);

            databend_common_base::runtime::block_on(async move {
                if let Ok(table) = self
                    .ctx
                    .resolve_data_source(
                        &catalog_name,
                        &database_name,
                        &table_name,
                        branch.as_deref(),
                        None,
                    )
                    .await
                {
                    if let Some(snapshot) = TableSnapshot::from_resolved_table(
                        table.as_ref(),
                        catalog_name,
                        database_name,
                        table_name,
                    ) {
                        self.has_security_policy |= snapshot.has_security_policy();
                        self.table_snapshots.push(snapshot);
                        return;
                    }
                }
                self.cache_miss = true;
            });
        }
    }
}
