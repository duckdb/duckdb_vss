#include "hnsw/hnsw_index_physical_create.hpp"

#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/exception/transaction_exception.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "duckdb/storage/table_io_manager.hpp"
#include "hnsw/hnsw_index.hpp"

#include "duckdb/parallel/base_pipeline_event.hpp"

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
class CreateHNSWIndexGlobalState final : public GlobalSinkState {
public:
	//! Global index to be added to the table
	unique_ptr<HNSWIndex> global_index;

	mutex glock;
	unique_ptr<ColumnDataCollection> collection;
	shared_ptr<ClientContext> context;

	// Parallel scan state
	ColumnDataParallelScanState scan_state;
};

unique_ptr<GlobalSinkState> PhysicalCreateHNSWIndex::GetGlobalSinkState(ClientContext &context) const {
	auto gstate = make_uniq<CreateHNSWIndexGlobalState>();

	vector<LogicalType> data_types = {unbound_expressions[0]->return_type, LogicalType::ROW_TYPE};
	gstate->collection = make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(context), data_types);
	gstate->context = context.shared_from_this();

	// Create the index
	auto &storage = table.GetStorage();
	auto &table_manager = TableIOManager::Get(storage);
	auto &constraint_type = info->constraint_type;
	auto &db = storage.db;
	gstate->global_index =
	    make_uniq<HNSWIndex>(info->index_name, constraint_type, storage_ids, table_manager, unbound_expressions, db,
	                         info->options, IndexStorageInfo(), estimated_cardinality);

	return std::move(gstate);
}

//-------------------------------------------------------------
// Local State
//-------------------------------------------------------------
class CreateHNSWIndexLocalState final : public LocalSinkState {
public:
	unique_ptr<ColumnDataCollection> collection;
	ColumnDataAppendState append_state;
};

unique_ptr<LocalSinkState> PhysicalCreateHNSWIndex::GetLocalSinkState(ExecutionContext &context) const {
	auto state = make_uniq<CreateHNSWIndexLocalState>();

	vector<LogicalType> data_types = {unbound_expressions[0]->return_type, LogicalType::ROW_TYPE};
	state->collection = make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(context.client), data_types);
	state->collection->InitializeAppend(state->append_state);
	return std::move(state);
}

//-------------------------------------------------------------
// Sink
//-------------------------------------------------------------

SinkResultType PhysicalCreateHNSWIndex::Sink(ExecutionContext &context, DataChunk &chunk,
                                             OperatorSinkInput &input) const {

	auto &lstate = input.local_state.Cast<CreateHNSWIndexLocalState>();
	lstate.collection->Append(lstate.append_state, chunk);
	return SinkResultType::NEED_MORE_INPUT;
}

//-------------------------------------------------------------
// Combine
//-------------------------------------------------------------
SinkCombineResultType PhysicalCreateHNSWIndex::Combine(ExecutionContext &context,
                                                       OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<CreateHNSWIndexGlobalState>();
	auto &lstate = input.local_state.Cast<CreateHNSWIndexLocalState>();

	if (lstate.collection->Count() == 0) {
		return SinkCombineResultType::FINISHED;
	}

	lock_guard<mutex> l(gstate.glock);
	if (!gstate.collection) {
		gstate.collection = std::move(lstate.collection);
	} else {
		gstate.collection->Combine(*lstate.collection);
	}

	return SinkCombineResultType::FINISHED;
}

//-------------------------------------------------------------
// Finalize
//-------------------------------------------------------------

class HNSWIndexConstructTask final : public ExecutorTask {
public:
	HNSWIndexConstructTask(shared_ptr<Event> event_p, ClientContext &context, CreateHNSWIndexGlobalState &gstate_p,
	                       size_t thread_id_p)
	    : ExecutorTask(context, std::move(event_p)), gstate(gstate_p), thread_id(thread_id_p), local_scan_state() {
		// Initialize the scan chunk
		gstate.collection->InitializeScanChunk(scan_chunk);
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode mode) override {

		auto &index = gstate.global_index->index;
		auto &scan_state = gstate.scan_state;
		auto &collection = gstate.collection;

		const auto array_size = ArrayType::GetSize(scan_chunk.data[0].GetType());

		while (collection->Scan(scan_state, local_scan_state, scan_chunk)) {

			const auto count = scan_chunk.size();
			auto &vec_vec = scan_chunk.data[0];
			auto &data_vec = ArrayVector::GetEntry(vec_vec);
			auto &rowid_vec = scan_chunk.data[1];

			UnifiedVectorFormat vec_format;
			UnifiedVectorFormat data_format;
			UnifiedVectorFormat rowid_format;

			vec_vec.ToUnifiedFormat(count, vec_format);
			data_vec.ToUnifiedFormat(count * array_size, data_format);
			rowid_vec.ToUnifiedFormat(count, rowid_format);

			const auto row_ptr = UnifiedVectorFormat::GetData<row_t>(rowid_format);
			const auto data_ptr = UnifiedVectorFormat::GetData<float>(data_format);

			for (idx_t i = 0; i < count; i++) {
				const auto vec_idx = vec_format.sel->get_index(i);
				const auto row_idx = rowid_format.sel->get_index(i);

				// Check for NULL values
				const auto vec_valid = vec_format.validity.RowIsValid(vec_idx);
				const auto rowid_valid = rowid_format.validity.RowIsValid(row_idx);
				if (!vec_valid || !rowid_valid) {
					executor.PushError(
					    ErrorData("Invalid data in HNSW index construction: Cannot construct index with NULL values."));
					return TaskExecutionResult::TASK_ERROR;
				}

				// Add the vector to the index
				const auto result = index.add(row_ptr[row_idx], data_ptr + (vec_idx * array_size), thread_id);

				// Check for errors
				if (!result) {
					executor.PushError(ErrorData(result.error.what()));
					return TaskExecutionResult::TASK_ERROR;
				}
			}

			if (mode == TaskExecutionMode::PROCESS_PARTIAL) {
				// yield!
				return TaskExecutionResult::TASK_NOT_FINISHED;
			}
		}

		// Finish task!
		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

private:
	CreateHNSWIndexGlobalState &gstate;
	size_t thread_id;

	DataChunk scan_chunk;
	ColumnDataLocalScanState local_scan_state;
};

class HNSWIndexConstructionEvent final : public BasePipelineEvent {
public:
	HNSWIndexConstructionEvent(CreateHNSWIndexGlobalState &gstate_p, Pipeline &pipeline_p, CreateIndexInfo &info_p,
	                           const vector<column_t> &storage_ids_p, DuckTableEntry &table_p)
	    : BasePipelineEvent(pipeline_p), gstate(gstate_p), info(info_p), storage_ids(storage_ids_p), table(table_p) {
	}

	CreateHNSWIndexGlobalState &gstate;
	CreateIndexInfo &info;
	const vector<column_t> &storage_ids;
	DuckTableEntry &table;

public:
	void Schedule() override {
		auto &context = pipeline->GetClientContext();

		// Schedule tasks equal to the number of threads, which will construct the index
		auto &ts = TaskScheduler::GetScheduler(context);
		const auto num_threads = NumericCast<size_t>(ts.NumberOfThreads());

		vector<shared_ptr<Task>> construct_tasks;
		for (size_t tnum = 0; tnum < num_threads; tnum++) {
			construct_tasks.push_back(make_uniq<HNSWIndexConstructTask>(shared_from_this(), context, gstate, tnum));
		}
		SetTasks(std::move(construct_tasks));
	}

	void FinishEvent() override {

		// Mark the index as dirty, update its count
		gstate.global_index->SetDirty();
		gstate.global_index->SyncSize();

		auto &storage = table.GetStorage();

		// If not in memory, persist the index to disk
		if (!storage.db.GetStorageManager().InMemory()) {
			// Finalize the index
			gstate.global_index->PersistToDisk();
		}

		if (!storage.IsRoot()) {
			throw TransactionException("Cannot create index on non-root transaction");
		}

		// Create the index entry in the catalog
		auto &schema = table.schema;
		info.column_ids = storage_ids;
		const auto index_entry = schema.CreateIndex(*gstate.context, info, table).get();
		if (!index_entry) {
			D_ASSERT(info.on_conflict == OnCreateConflict::IGNORE_ON_CONFLICT);
			// index already exists, but error ignored because of IF NOT EXISTS
			// return SinkFinalizeType::READY;
			return;
		}

		// Get the entry as a DuckIndexEntry
		auto &duck_index = index_entry->Cast<DuckIndexEntry>();
		duck_index.initial_index_size = gstate.global_index->Cast<BoundIndex>().GetInMemorySize();
		duck_index.info = make_uniq<IndexDataTableInfo>(storage.GetDataTableInfo(), duck_index.name);
		for (auto &parsed_expr : info.parsed_expressions) {
			duck_index.parsed_expressions.push_back(parsed_expr->Copy());
		}

		// Finally add it to storage
		storage.AddIndex(std::move(gstate.global_index));
	}
};

SinkFinalizeType PhysicalCreateHNSWIndex::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                   OperatorSinkFinalizeInput &input) const {

	// Get the global collection we've been appending to
	auto &gstate = input.global_state.Cast<CreateHNSWIndexGlobalState>();
	auto &collection = gstate.collection;

	// Reserve the index size
	auto &index = gstate.global_index->index;
	index.reserve(collection->Count());

	// Initialize a parallel scan for the index construction
	collection->InitializeScan(gstate.scan_state, ColumnDataScanProperties::ALLOW_ZERO_COPY);

	// Create a new event that will construct the index
	auto new_event = make_shared_ptr<HNSWIndexConstructionEvent>(gstate, pipeline, *info, storage_ids, table);
	event.InsertEvent(std::move(new_event));

	return SinkFinalizeType::READY;
}

} // namespace duckdb