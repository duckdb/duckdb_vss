#pragma once

#include "duckdb/function/table_function.hpp"

namespace duckdb {

class DuckTableEntry;
class Index;

// This is created by the optimizer rule
struct HNSWIndexScanBindData : public TableFunctionData {
	explicit HNSWIndexScanBindData(DuckTableEntry &table, Index &index, idx_t limit, unsafe_unique_array<float> query)
	    : table(table), index(index), limit(limit), query(std::move(query)) {
	}

	//! The table to scan
	DuckTableEntry &table;

	//! The index to use
	Index &index;

	//! The limit of the scan
	idx_t limit;

	//! The query vector
	unsafe_unique_array<float> query;

public:
	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<HNSWIndexScanBindData>();
		return &other.table == &table;
	}
};

struct HNSWIndexScanFunction {
	static TableFunction GetFunction();
};

} // namespace duckdb