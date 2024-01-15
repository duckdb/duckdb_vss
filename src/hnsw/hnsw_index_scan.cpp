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

BindInfo HNSWIndexScanBindInfo(const optional_ptr<FunctionData> bind_data_p) {
	auto &bind_data = bind_data_p->Cast<HNSWIndexScanBindData>();
	return BindInfo(bind_data.table);
}

//-------------------------------------------------------------------------
// Global State
//-------------------------------------------------------------------------
struct HNSWIndexScanGlobalState : public GlobalTableFunctionState {
	ColumnFetchState fetch_state;
	TableScanState local_storage_state;
	vector<storage_t> column_ids;

	// Index scan state
	unique_ptr<IndexScanState> index_state;
	Vector row_ids = Vector(LogicalType::ROW_TYPE);
};

static unique_ptr<GlobalTableFunctionState> HNSWIndexScanInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind_data = input.bind_data->Cast<HNSWIndexScanBindData>();

	auto result = make_uniq<HNSWIndexScanGlobalState>();

	// Setup the scan state for the local storage
	auto &local_storage = LocalStorage::Get(context, bind_data.table.catalog);
	result->column_ids.reserve(input.column_ids.size());

	// Figure out the storage column ids
	for (auto &id : input.column_ids) {
		storage_t col_id = id;
		if(id != DConstants::INVALID_INDEX) {
			col_id = bind_data.table.GetColumn(LogicalIndex(id)).StorageOid();
		}
		result->column_ids.push_back(col_id);
	}

	// Initialize the storage scan state
	result->local_storage_state.Initialize(result->column_ids, input.filters.get());
	local_storage.InitializeScan(bind_data.table.GetStorage(), result->local_storage_state.local_state, input.filters);

	// Initialize the scan state for the index
	result->index_state = bind_data.index.Cast<HNSWIndex>().InitializeScan(bind_data.query.get(), bind_data.limit);

	return std::move(result);
}

//-------------------------------------------------------------------------
// Execute
//-------------------------------------------------------------------------
static void HNSWIndexScanExecute(ClientContext &context, TableFunctionInput &data_p, DataChunk &output) {

	auto &bind_data = data_p.bind_data->Cast<HNSWIndexScanBindData>();
	auto &state = data_p.global_state->Cast<HNSWIndexScanGlobalState>();
	auto &transaction = DuckTransaction::Get(context, bind_data.table.catalog);

	// Scan the index for row id's
	auto row_count = bind_data.index.Cast<HNSWIndex>().Scan(*state.index_state, state.row_ids);
	if(row_count == 0){
		// Short-circuit if the index had no more rows
		output.SetCardinality(0);
		return;
	}

	// Fetch the data from the local storage given the row ids
	bind_data.table.GetStorage().Fetch(transaction, output, state.column_ids, state.row_ids, row_count, state.fetch_state);
}

//-------------------------------------------------------------------------
// Statistics
//-------------------------------------------------------------------------
static unique_ptr<BaseStatistics> HNSWIndexScanStatistics(ClientContext &context, const FunctionData *bind_data_p,
                                                            column_t column_id) {
	auto &bind_data = bind_data_p->Cast<HNSWIndexScanBindData>();
	auto &local_storage = LocalStorage::Get(context, bind_data.table.catalog);
	if (local_storage.Find(bind_data.table.GetStorage())) {
		// we don't emit any statistics for tables that have outstanding transaction-local data
		return nullptr;
	}
	return bind_data.table.GetStatistics(context, column_id);
}

//-------------------------------------------------------------------------
// Dependency
//-------------------------------------------------------------------------
void HNSWIndexScanDependency(DependencyList &entries, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<HNSWIndexScanBindData>();
	entries.AddDependency(bind_data.table);

	// TODO: Add dependency to index here?
}

//-------------------------------------------------------------------------
// Cardinality
//-------------------------------------------------------------------------
unique_ptr<NodeStatistics> HNSWIndexScanCardinality(ClientContext &context, const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<HNSWIndexScanBindData>();
	auto &local_storage = LocalStorage::Get(context, bind_data.table.catalog);
	auto &storage = bind_data.table.GetStorage();
	idx_t estimated_cardinality = storage.info->cardinality + local_storage.AddedRows(bind_data.table.GetStorage());
	return make_uniq<NodeStatistics>(storage.info->cardinality, estimated_cardinality);
}

//-------------------------------------------------------------------------
// ToString
//-------------------------------------------------------------------------
static string HNSWIndexScanToString(const FunctionData *bind_data_p) {
	auto &bind_data = bind_data_p->Cast<HNSWIndexScanBindData>();
	return bind_data.table.name + " (HNSW INDEX SCAN : " + bind_data.index.name + ")";
}

//-------------------------------------------------------------------------
// Get Function
//-------------------------------------------------------------------------
TableFunction HNSWIndexScanFunction::GetFunction() {
	TableFunction func("hnsw_index_scan", {}, HNSWIndexScanExecute);
	func.init_local = nullptr;
	func.init_global = HNSWIndexScanInitGlobal;
	func.statistics = HNSWIndexScanStatistics;
	func.dependency = HNSWIndexScanDependency;
	func.cardinality = HNSWIndexScanCardinality;
	func.pushdown_complex_filter = nullptr;
	func.to_string = HNSWIndexScanToString;
	func.table_scan_progress = nullptr;
	func.get_batch_index = nullptr;
	func.projection_pushdown = true;
	func.filter_pushdown = false;
	func.get_bind_info = HNSWIndexScanBindInfo;

	return func;
}

//-------------------------------------------------------------------------
// Register
//-------------------------------------------------------------------------
void HNSWModule::RegisterIndexScan(DatabaseInstance &db) {
	ExtensionUtil::RegisterFunction(db, HNSWIndexScanFunction::GetFunction());
}


}