#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/dependency_list.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/transaction/local_storage.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"
#include "duckdb/storage/data_table.hpp"

#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"
#include "hnsw/hnsw_index_scan.hpp"

namespace duckdb {

// BIND
static unique_ptr<FunctionData> HNSWindexInfoBind(ClientContext &context, TableFunctionBindInput &input,
                                                  vector<LogicalType> &return_types, vector<string> &names) {
	names.emplace_back("catalog_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("schema_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("index_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("table_name");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("metric");
	return_types.emplace_back(LogicalType::VARCHAR);

	names.emplace_back("dimensions");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("count");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("capacity");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("approx_memory_usage");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("levels");
	return_types.emplace_back(LogicalType::BIGINT);

	names.emplace_back("levels_stats");
	return_types.emplace_back(LogicalType::LIST(LogicalType::STRUCT({{"nodes", LogicalType::BIGINT},
	                                                                 {"edges", LogicalType::BIGINT},
	                                                                 {"max_edges", LogicalType::BIGINT},
	                                                                 {"allocated_bytes", LogicalType::BIGINT}})));

	return nullptr;
}

// INIT GLOBAL
struct HNSWIndexInfoGlobalState : public GlobalTableFunctionState {
	idx_t offset = 0;
	vector<reference<IndexCatalogEntry>> entries;
};

static unique_ptr<GlobalTableFunctionState> HNSWIndexInfoInitGlobal(ClientContext &context,
                                                                    TableFunctionInitInput &input) {
	auto result = make_uniq<HNSWIndexInfoGlobalState>();

	// scan all the schemas for indexes and collect them
	auto schemas = Catalog::GetAllSchemas(context);
	for (auto &schema : schemas) {
		schema.get().Scan(context, CatalogType::INDEX_ENTRY, [&](CatalogEntry &entry) {
			auto &index_entry = entry.Cast<IndexCatalogEntry>();
			if (index_entry.index_type == HNSWIndex::TYPE_NAME) {
				result->entries.push_back(index_entry);
			}
		});
	};
	return std::move(result);
}

// EXECUTE
static void HNSWIndexInfoExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {
	auto &data = data_p.global_state->Cast<HNSWIndexInfoGlobalState>();
	if (data.offset >= data.entries.size()) {
		return;
	}

	idx_t row = 0;
	while (data.offset < data.entries.size() && row < STANDARD_VECTOR_SIZE) {
		auto &index_entry = data.entries[data.offset++].get();
		auto &table_entry = index_entry.schema.catalog.GetEntry<TableCatalogEntry>(context, index_entry.GetSchemaName(),
		                                                                           index_entry.GetTableName());
		auto &storage = table_entry.GetStorage();
		HNSWIndex *hnsw_index = nullptr;

		auto &table_info = *storage.GetDataTableInfo();
		table_info.GetIndexes().BindAndScan<HNSWIndex>(context, table_info, [&](HNSWIndex &index) {
			if (index.name == index_entry.name) {
				hnsw_index = &index;
				return true;
			}
			return false;
		});

		if (!hnsw_index) {
			throw BinderException("Index %s not found", index_entry.name);
		}

		idx_t col = 0;

		output.data[col++].SetValue(row, Value(index_entry.catalog.GetName()));
		output.data[col++].SetValue(row, Value(index_entry.schema.name));
		output.data[col++].SetValue(row, Value(index_entry.name));
		output.data[col++].SetValue(row, Value(table_entry.name));

		auto stats = hnsw_index->GetStats();

		output.data[col++].SetValue(row, Value(hnsw_index->GetMetric()));
		output.data[col++].SetValue(row, Value::BIGINT(hnsw_index->GetVectorSize()));
		output.data[col++].SetValue(row, Value::BIGINT(stats->count));
		output.data[col++].SetValue(row, Value::BIGINT(stats->capacity));
		output.data[col++].SetValue(row, Value::BIGINT(stats->approx_size));
		output.data[col++].SetValue(row, Value::BIGINT(stats->max_level));

		vector<Value> level_stats;
		for (auto &stat : stats->level_stats) {
			level_stats.push_back(Value::STRUCT({{"nodes", Value::BIGINT(stat.nodes)},
			                                     {"edges", Value::BIGINT(stat.edges)},
			                                     {"max_edges", Value::BIGINT(stat.max_edges)},
			                                     {"allocated_bytes", Value::BIGINT(stat.allocated_bytes)}}));
		}
		auto level_stat_value = Value::LIST(LogicalType::STRUCT({{{"nodes", LogicalType::BIGINT},
		                                                          {"edges", LogicalType::BIGINT},
		                                                          {"max_edges", LogicalType::BIGINT},
		                                                          {"allocated_bytes", LogicalType::BIGINT}}}),
		                                    level_stats);

		output.data[col++].SetValue(row, level_stat_value);

		row++;
	}
	output.SetCardinality(row);
}

//-------------------------------------------------------------------------
// Compact PRAGMA
//-------------------------------------------------------------------------

static void CompactIndexPragma(ClientContext &context, const FunctionParameters &parameters) {
	if (parameters.values.size() != 1) {
		throw BinderException("Expected one argument for hnsw_compact_index");
	}
	auto &param = parameters.values[0];
	if (param.type() != LogicalType::VARCHAR) {
		throw BinderException("Expected a string argument for hnsw_compact_index");
	}
	auto index_name = param.GetValue<string>();

	auto qname = QualifiedName::Parse(index_name);

	// look up the index name in the catalog
	Binder::BindSchemaOrCatalog(context, qname.catalog, qname.schema);
	auto &index_entry = Catalog::GetEntry(context, CatalogType::INDEX_ENTRY, qname.catalog, qname.schema, qname.name)
	                        .Cast<IndexCatalogEntry>();
	auto &table_entry = Catalog::GetEntry(context, CatalogType::TABLE_ENTRY, qname.catalog, index_entry.GetSchemaName(),
	                                      index_entry.GetTableName())
	                        .Cast<TableCatalogEntry>();

	auto &storage = table_entry.GetStorage();
	bool found_index = false;

	auto &table_info = *storage.GetDataTableInfo();
	table_info.GetIndexes().BindAndScan<HNSWIndex>(context, table_info, [&](HNSWIndex &hnsw_index) {
		if (index_entry.name == index_name) {
			hnsw_index.Compact();
			found_index = true;
			return true;
		}
		return false;
	});

	if (!found_index) {
		throw BinderException("Index %s not found", index_name);
	}
}

//-------------------------------------------------------------------------
// Register
//-------------------------------------------------------------------------
void HNSWModule::RegisterIndexPragmas(DatabaseInstance &db) {
	ExtensionUtil::RegisterFunction(
	    db, PragmaFunction::PragmaCall("hnsw_compact_index", CompactIndexPragma, {LogicalType::VARCHAR}));

	// TODO: This is kind of ugly and maybe should just take a parameter instead...
	TableFunction info_function("pragma_hnsw_index_info", {}, HNSWIndexInfoExecute, HNSWindexInfoBind,
	                            HNSWIndexInfoInitGlobal);
	ExtensionUtil::RegisterFunction(db, info_function);
}

} // namespace duckdb