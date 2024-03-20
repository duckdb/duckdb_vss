#pragma once

#include "duckdb.hpp"

namespace duckdb {

struct HNSWModule {
public:
	static void Register(DatabaseInstance &db) {
		RegisterIndex(db);
		RegisterIndexScan(db);
		RegisterIndexPragmas(db);
		RegisterPlanIndexScan(db);
		RegisterPlanIndexCreate(db);
	}

private:
	static void RegisterIndex(DatabaseInstance &db);
	static void RegisterIndexScan(DatabaseInstance &db);
	static void RegisterIndexPragmas(DatabaseInstance &db);
	static void RegisterPlanIndexScan(DatabaseInstance &db);
	static void RegisterPlanIndexCreate(DatabaseInstance &db);
};

} // namespace duckdb