#include "duckdb.hpp"
#include <iostream>
#include <benchmark/benchmark.h>

using namespace duckdb;

// Benchmark for search before clustering
static void BM_SearchBeforeClustering(benchmark::State& state) {
    DuckDB db(nullptr);
	Connection con(db);

	con.Query("SET enable_progress_bar = true;");

	con.Query("ATTACH 'raw.db' AS raw (READ_ONLY);");
	std::cout << "Attached raw.db" << std::endl;

	con.Query("CREATE OR REPLACE TABLE memory.fashion_mnist_train (vec FLOAT[784])");
    con.Query("INSERT INTO memory.fashion_mnist_train SELECT * FROM raw.fashion_mnist_train;");
	std::cout << "Copied from raw to memory" << std::endl;

	con.Query("DETACH raw;");
	std::cout << "Detached raw.db" << std::endl;

	con.Query("CREATE INDEX clustered_hnsw_index ON memory.fashion_mnist_train USING HNSW (vec);");

	// Example query vector
    std::vector<float> query_vector(784, 0.1f); 

    // Start building the query string
    std::ostringstream query_stream;
    query_stream << "SELECT * FROM memory.fashion_mnist_train ORDER BY array_distance(vec, [";

    // Append each element of query_vector to the query string
    for (size_t i = 0; i < query_vector.size(); ++i) {
        query_stream << query_vector[i];
        if (i < query_vector.size() - 1) {
            // If not the last element, add a comma
            query_stream << ",";
        }
    }

    // Finish the query string
    query_stream << "]::FLOAT[" << query_vector.size() << "]) LIMIT 100";

    for (auto _ : state) {
		// Convert the stream into a string and execute the query
        con.Query(query_stream.str());
    }
}

BENCHMARK(BM_SearchBeforeClustering);
// BENCHMARK(BM_SearchAfterClustering);

BENCHMARK_MAIN();
