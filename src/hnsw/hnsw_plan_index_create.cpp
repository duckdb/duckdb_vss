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
		for (auto &option : create_index.info->options) {
			auto &k = option.first;
			auto &v = option.second;
			if (StringUtil::CIEquals(k, "metric")) {
				if (v.type() != LogicalType::VARCHAR) {
					throw BinderException("HNSW index metric must be a string");
				}
				auto metric = v.GetValue<string>();
				if (HNSWIndex::METRIC_KIND_MAP.find(metric) == HNSWIndex::METRIC_KIND_MAP.end()) {
					vector<string> allowed_metrics;
					for (auto &entry : HNSWIndex::METRIC_KIND_MAP) {
						allowed_metrics.push_back(StringUtil::Format("'%s'", entry.first));
					}
					throw BinderException("HNSW index metric must be one of: %s",
					                      StringUtil::Join(allowed_metrics, ", "));
				}
			}
		}

		// Verify the expression type
		if (create_index.expressions.size() != 1) {
			throw BinderException("HNSW indexes can only be created over a single column of keys.");
		}
		auto &arr_type = create_index.expressions[0]->return_type;
		if (arr_type.id() != LogicalTypeId::ARRAY) {
			throw BinderException("HNSW index keys must be of type FLOAT[N]");
		}
		auto &child_type = ArrayType::GetChildType(arr_type);
		auto child_type_val = HNSWIndex::SCALAR_KIND_MAP.find(static_cast<uint8_t>(child_type.id()));
		if (child_type_val == HNSWIndex::SCALAR_KIND_MAP.end()) {
			vector<string> allowed_types;
			for (auto &entry : HNSWIndex::SCALAR_KIND_MAP) {
				auto id = static_cast<LogicalTypeId>(entry.first);
				allowed_types.push_back(StringUtil::Format("'%s[N]'", LogicalType(id).ToString()));
			}
			throw BinderException("HNSW index key type must be one of: %s", StringUtil::Join(allowed_types, ", "));
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