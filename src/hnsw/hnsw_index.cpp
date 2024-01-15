#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/storage/metadata/metadata_reader.hpp"

#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"


namespace duckdb {

//------------------------------------------------------------------------------
// HNSWIndex Methods
//------------------------------------------------------------------------------

// Constructor
HNSWIndex::HNSWIndex(
    const string &name, IndexConstraintType index_constraint_type,
    const vector<column_t> &column_ids, TableIOManager &table_io_manager,
    const vector<unique_ptr<Expression>> &unbound_expressions, AttachedDatabase &db,
    const IndexStorageInfo &info
    ) : Index(name, TYPE_NAME, index_constraint_type, column_ids, table_io_manager, unbound_expressions, db) {

	if(index_constraint_type != IndexConstraintType::NONE) {
		throw NotImplementedException("HNSW indexes do not support unique or primary key constraints");
	}

	// We only support one ARRAY column
	D_ASSERT(logical_types.size() == 1);
	auto &vector_type = logical_types[0];
	D_ASSERT(vector_type.id() == LogicalTypeId::ARRAY);

	// Get the size of the vector
	auto vector_size = ArrayType::GetSize(vector_type);

	// Create the usearch index
	unum::usearch::metric_punned_t metric(vector_size, unum::usearch::metric_kind_t::l2sq_k, unum::usearch::scalar_kind_t::f32_k);
	index = unum::usearch::index_dense_t::make(metric);

	// Is this a new index or an existing index?
	if(!info.allocator_infos.empty()) {
		// Hack: store the root block id in the fixed size allocator info
		root_block_ptr = info.allocator_infos[0].block_pointers[0];

		// This is an old index that needs to be loaded
		auto &meta_manager = table_io_manager.GetMetadataManager();
		MetadataReader reader(meta_manager, root_block_ptr);
		index.load_from_stream([&](void* data, size_t size) {
			reader.ReadData(static_cast<data_ptr_t>(data), size);
			return true;
		});
	} else {
		index.reserve(0);
	}
}

idx_t HNSWIndex::GetVectorSize() const {
	return index.dimensions();
}

// Scan State
struct HNSWIndexScanState : public IndexScanState {
	idx_t current_row;
	idx_t total_rows;
	unique_array<row_t> row_ids;
};

unique_ptr<IndexScanState> HNSWIndex::InitializeScan(float *query_vector, idx_t limit) const {
	auto state = make_uniq<HNSWIndexScanState>();
	auto search_result = index.search(query_vector, limit);

	state->current_row = 0;
	state->total_rows = search_result.size();
	state->row_ids = make_uniq_array<row_t>(search_result.size());

	search_result.dump_to(reinterpret_cast<uint64_t*>(state->row_ids.get()));
	return std::move(state);
}

idx_t HNSWIndex::Scan(IndexScanState &state, Vector &result) {
	auto &scan_state = state.Cast<HNSWIndexScanState>();

	idx_t count = 0;
	auto row_ids = FlatVector::GetData<row_t>(result);

	// Push the row ids into the result vector, up to STANDARD_VECTOR_SIZE or the end of the result set
	while(count < STANDARD_VECTOR_SIZE && scan_state.current_row < scan_state.total_rows) {
		row_ids[count++] = scan_state.row_ids[scan_state.current_row++];
	}

	return count;
}

void HNSWIndex::CommitDrop(IndexLock &index_lock) {
	// meta_manager.MarkBlocksAsModified();

	// Nothing to do here, the map will be deleted when the index is deleted
}

void HNSWIndex::Construct(DataChunk &input, Vector &row_ids) {
	D_ASSERT(row_ids.GetType().InternalType() == ROW_TYPE);
	D_ASSERT(logical_types[0] == input.data[0].GetType());

	auto count = input.size();
	input.Flatten();

	// TODO: Do we need to track this atomically globally?
	index.reserve(index.capacity() + count);

	auto &vec_vec = input.data[0];
	auto &vec_child_vec = ArrayVector::GetEntry(vec_vec);
	auto array_size = ArrayType::GetSize(vec_vec.GetType());

	auto vec_child_data = FlatVector::GetData<float>(vec_child_vec);
	auto rowid_data = FlatVector::GetData<row_t>(row_ids);

	// TODO: Maybe set a local state id?
	// auto thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());

	for(idx_t out_idx = 0; out_idx < count; out_idx++) {
		auto rowid = rowid_data[out_idx];
		index.add(rowid, vec_child_data + (out_idx * array_size));
	}
}


void HNSWIndex::Delete(IndexLock &lock, DataChunk &input, Vector &row_ids_vector) {
	throw NotImplementedException("Cannot update a HNSW index after it has been created");
}

PreservedError HNSWIndex::Insert(IndexLock &lock, DataChunk &input, Vector &rowid_vec) {
	throw NotImplementedException("Cannot update a HNSW index after it has been created");
}

PreservedError HNSWIndex::Append(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) {
	throw NotImplementedException("Cannot update a HNSW index after it has been created");
}

void HNSWIndex::VerifyAppend(DataChunk &chunk) {
	throw NotImplementedException("HNSWIndex::VerifyAppend() not implemented");
}

void HNSWIndex::VerifyAppend(DataChunk &chunk, ConflictManager &conflict_manager) {
	throw NotImplementedException("HNSWIndex::VerifyAppend() not implemented");
}

void HNSWIndex::PersistToDisk() {
	// Write
	auto &meta_manager = table_io_manager.GetMetadataManager();
	MetadataWriter writer(meta_manager);

	// Get the first block pointer
	root_block_ptr = writer.GetBlockPointer();
	index.save_to_stream([&](const void* data, size_t size) {
		writer.WriteData(static_cast<const_data_ptr_t>(data), size);
		return true;
	});
	writer.Flush();
	meta_manager.Flush();
}

IndexStorageInfo HNSWIndex::GetStorageInfo(const bool get_buffers) {

	IndexStorageInfo info;
	info.name = name;

	// HACK: store the root block pointer in the fixed size allocator info
	info.allocator_infos.emplace_back();
	info.allocator_infos.back().block_pointers.emplace_back(root_block_ptr);
	return info;
}

idx_t HNSWIndex::GetInMemorySize(IndexLock &state) {
	// TODO: This is not correct: its a lower bound, but it's a start
	return index.memory_usage();
}

bool HNSWIndex::MergeIndexes(IndexLock &state, Index &other_index) {
	throw NotImplementedException("HNSWIndex::MergeIndexes() not implemented");
}

void HNSWIndex::Vacuum(IndexLock &state) {
	// Re-compact the index
	index.compact();
}

void HNSWIndex::CheckConstraintsForChunk(DataChunk &input, ConflictManager &conflict_manager) {
	throw NotImplementedException("HNSWIndex::CheckConstraintsForChunk() not implemented");
}

string HNSWIndex::VerifyAndToString(IndexLock &state, const bool only_verify) {
	throw NotImplementedException("HNSWIndex::VerifyAndToString() not implemented");
}

//------------------------------------------------------------------------------
// Register Index Type
//------------------------------------------------------------------------------
void HNSWModule::RegisterIndex(DatabaseInstance &db) {

	IndexType index_type;
	index_type.name = HNSWIndex::TYPE_NAME;
	index_type.create_instance = [](const string &name, const IndexConstraintType index_constraint_type,
	                                const vector<column_t> &column_ids,
	                                const vector<unique_ptr<Expression>> &unbound_expressions,
	                                TableIOManager &table_io_manager,
	                                AttachedDatabase &db,
	                                const IndexStorageInfo &storage_info) -> unique_ptr<Index> {

		auto res = make_uniq<HNSWIndex>(name, index_constraint_type, column_ids, table_io_manager, unbound_expressions,
		                                db, storage_info);
		return std::move(res);
	};

	// Register the index type
	db.config.GetIndexTypes().RegisterIndexType(index_type);
}


}