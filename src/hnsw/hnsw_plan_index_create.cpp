#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/operator/logical_create_index.hpp"
#include "duckdb/parser/parsed_data/create_index_info.hpp"

#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"
#include "hnsw/hnsw_index_logical_create.hpp"

namespace duckdb {

//-----------------------------------------------------------------------------
// Plan rewriter
//-----------------------------------------------------------------------------
class HNSWIndexInsertionRewriter : public OptimizerExtension {
public:
	HNSWIndexInsertionRewriter() {
		optimize_function = HNSWIndexInsertionRewriter::Optimize;
	}

	static void TryOptimize(ClientContext &context, OptimizerExtensionInfo *info, unique_ptr<LogicalOperator> &plan) {
		auto &op = *plan;

		// Look for a CREATE INDEX operator
		if (op.type != LogicalOperatorType::LOGICAL_CREATE_INDEX) {
			return;
		}
		auto &create_index = op.Cast<LogicalCreateIndex>();

		if (create_index.info->index_type != HNSWIndex::TYPE_NAME) {
			// Not the index type we are looking for
			return;
		}

		// Verify the options
		for (auto &opt : create_index.info->options) {
			if (opt.first == "dimensions") {
				if (opt.second.type() != LogicalType::INTEGER) {
					throw BinderException("HNSW index dimensions must be an integer.");
				}
			}
		}

		// We have a create index operator for our index
		// We can replace this with a operator that creates the index
		// The "LogicalCreateHNSWINdex" operator is a custom operator that we defined in the extension
		auto physical_create_index = make_uniq<LogicalCreateHNSWIndex>(
		    std::move(create_index.info), std::move(create_index.expressions), create_index.table);

		// Move the children
		physical_create_index->children = std::move(create_index.children);

		// Replace the operator
		plan = std::move(physical_create_index);
	}

	static void Optimize(ClientContext &context, OptimizerExtensionInfo *info, unique_ptr<LogicalOperator> &plan) {

		TryOptimize(context, info, plan);

		// Recursively traverse the children
		for (auto &child : plan->children) {
			Optimize(context, info, child);
		}
	};
};

//-------------------------------------------------------------
// Register
//-------------------------------------------------------------
void HNSWModule::RegisterPlanIndexCreate(DatabaseInstance &db) {
	// Register the optimizer extension
	db.config.optimizer_extensions.push_back(HNSWIndexInsertionRewriter());
}

} // namespace duckdb