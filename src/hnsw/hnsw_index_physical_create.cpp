#include "hnsw/hnsw_index_physical_create.hpp"
#include "hnsw/hnsw_index.hpp"

#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"
#include "duckdb/storage/table_io_manager.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "duckdb/common/exception/transaction_exception.hpp"

namespace duckdb {

PhysicalCreateHNSWIndex::PhysicalCreateHNSWIndex(LogicalOperator &op, TableCatalogEntry &table,
                                                 const vector<column_t> &column_ids, unique_ptr<CreateIndexInfo> info,
                                                 vector<unique_ptr<Expression>> unbound_expressions,
                                                 idx_t estimated_cardinality)
    // Declare this operators as a EXTENSION operator
    : PhysicalOperator(PhysicalOperatorType::EXTENSION, op.types, estimated_cardinality),
      table(table.Cast<DuckTableEntry>()), info(std::move(info)), unbound_expressions(std::move(unbound_expressions)),
      sorted(false) {

	// convert virtual column ids to storage column ids
	for (auto &column_id : column_ids) {
		storage_ids.push_back(table.GetColumns().LogicalToPhysical(LogicalIndex(column_id)).index);
	}
}

//-------------------------------------------------------------
// Global State
//-------------------------------------------------------------
class CreateHNSWIndexGlobalState : public GlobalSinkState {
public:
	//! Global index to be added to the table
	unique_ptr<Index> global_index;
	atomic<idx_t> next_thread_id = {0};
};

unique_ptr<GlobalSinkState> PhysicalCreateHNSWIndex::GetGlobalSinkState(ClientContext &context) const {
	auto gstate = make_uniq<CreateHNSWIndexGlobalState>();

	// Create the global index
	auto &storage = table.GetStorage();
	auto &table_manager = TableIOManager::Get(storage);
	auto &constraint_type = info->constraint_type;
	auto &db = storage.db;

	gstate->global_index = make_uniq<HNSWIndex>(info->index_name, constraint_type, storage_ids, table_manager,
	                                            unbound_expressions, db, info->options);

	return std::move(gstate);
}

//-------------------------------------------------------------
// Local State
//-------------------------------------------------------------
class CreateHNSWIndexLocalState : public LocalSinkState {
public:
	idx_t thread_id = idx_t(-1);
};

unique_ptr<LocalSinkState> PhysicalCreateHNSWIndex::GetLocalSinkState(ExecutionContext &context) const {
	auto state = make_uniq<CreateHNSWIndexLocalState>();
	return std::move(state);
}

//-------------------------------------------------------------
// Sink
//-------------------------------------------------------------

SinkResultType PhysicalCreateHNSWIndex::Sink(ExecutionContext &context, DataChunk &chunk,
                                             OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<CreateHNSWIndexGlobalState>();
	auto &lstate = input.local_state.Cast<CreateHNSWIndexLocalState>();
	auto &index = gstate.global_index->Cast<HNSWIndex>();

	if(lstate.thread_id == idx_t(-1)) {
		lstate.thread_id = gstate.next_thread_id++;
	}

	if (chunk.ColumnCount() != 2) {
		throw NotImplementedException("Custom index creation only supported for single-column indexes");
	}

	auto &row_identifiers = chunk.data[1];

	// Construct the index
	index.Construct(chunk, row_identifiers, lstate.thread_id);

	return SinkResultType::NEED_MORE_INPUT;
}

//-------------------------------------------------------------
// Combine
//-------------------------------------------------------------
SinkCombineResultType PhysicalCreateHNSWIndex::Combine(ExecutionContext &context,
                                                       OperatorSinkCombineInput &input) const {
	return SinkCombineResultType::FINISHED;
}

//-------------------------------------------------------------
// Finalize
//-------------------------------------------------------------
SinkFinalizeType PhysicalCreateHNSWIndex::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                   OperatorSinkFinalizeInput &input) const {

	// Get the global index we created
	auto &gstate = input.global_state.Cast<CreateHNSWIndexGlobalState>();

	// Vacuum excess memory and verify
	// gstate.global_index->Vacuum();
	// D_ASSERT(!state.global_index->VerifyAndToString(true).empty());

	// Check that the table hasnt been altered in the meantime
	auto &storage = table.GetStorage();

	// If not in memory, persist the index to disk
	if (!storage.db.GetStorageManager().InMemory()) {
		// Finalize the index
		gstate.global_index->Cast<HNSWIndex>().PersistToDisk();
	}

	if (!storage.IsRoot()) {
		throw TransactionException("Cannot create index on non-root transaction");
	}

	// Create the index entry in the catalog
	auto &schema = table.schema;
	info->column_ids = storage_ids;
	auto index_entry = schema.CreateIndex(context, *info, table).get();
	if (!index_entry) {
		D_ASSERT(info->on_conflict == OnCreateConflict::IGNORE_ON_CONFLICT);
		// index already exists, but error ignored because of IF NOT EXISTS
		return SinkFinalizeType::READY;
	}

	// Get the entry as a DuckIndexEntry
	auto &index = index_entry->Cast<DuckIndexEntry>();
	index.initial_index_size = gstate.global_index->GetInMemorySize();
	index.info = make_uniq<IndexDataTableInfo>(storage.info, index.name);
	for (auto &parsed_expr : info->parsed_expressions) {
		index.parsed_expressions.push_back(parsed_expr->Copy());
	}

	// Finally add it to storage
	storage.info->indexes.AddIndex(std::move(gstate.global_index));

	return SinkFinalizeType::READY;
}

} // namespace duckdb