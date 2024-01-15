#define DUCKDB_EXTENSION_MAIN

#include "vss_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

#include "hnsw/hnsw.hpp"

namespace duckdb {


static void LoadInternal(DatabaseInstance &instance) {
    // Register the HNSW index module
    HNSWModule::Register(instance);
}

void VssExtension::Load(DuckDB &db) {
	LoadInternal(*db.instance);
}

std::string VssExtension::Name() {
	return "vss";
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void vss_init(duckdb::DatabaseInstance &db) {
    duckdb::DuckDB db_wrapper(db);
    db_wrapper.LoadExtension<duckdb::VssExtension>();
}

DUCKDB_EXTENSION_API const char *vss_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
