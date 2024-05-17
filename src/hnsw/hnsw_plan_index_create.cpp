#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/planner/operator/logical_create_index.hpp"
#include "duckdb/parser/parsed_data/create_index_info.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"

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

	static void TryOptimize(ClientContext &context, unique_ptr<LogicalOperator> &plan) {
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

		Value enable_persistence;
		context.TryGetCurrentSetting("hnsw_enable_experimental_persistence", enable_persistence);

		auto is_disk_db = !create_index.table.GetStorage().db.GetStorageManager().InMemory();
		auto is_persistence_disabled = !enable_persistence.GetValue<bool>();

		if (is_disk_db && is_persistence_disabled) {
			throw BinderException("HNSW indexes can only be created in in-memory databases, or when the configuration "
			                      "option 'hnsw_enable_experimental_persistence' is set to true.");
		}

		// Verify the options
		for (auto &option : create_index.info->options) {
			auto &k = option.first;
			auto &v = option.second;
			if (StringUtil::CIEquals(k, "metric")) {
				if (v.type() != LogicalType::VARCHAR) {
					throw BinderException("HNSW index 'metric' must be a string");
				}
				auto metric = v.GetValue<string>();
				if (HNSWIndex::METRIC_KIND_MAP.find(metric) == HNSWIndex::METRIC_KIND_MAP.end()) {
					vector<string> allowed_metrics;
					for (auto &entry : HNSWIndex::METRIC_KIND_MAP) {
						allowed_metrics.push_back(StringUtil::Format("'%s'", entry.first));
					}
					throw BinderException("HNSW index 'metric' must be one of: %s",
					                      StringUtil::Join(allowed_metrics, ", "));
				}
			} else if (StringUtil::CIEquals(k, "ef_construction")) {
				if (v.type() != LogicalType::INTEGER) {
					throw BinderException("HNSW index 'ef_construction' must be an integer");
				}
				if (v.GetValue<int32_t>() < 1) {
					throw BinderException("HNSW index 'ef_construction' must be at least 1");
				}
			} else if (StringUtil::CIEquals(k, "ef_search")) {
				if (v.type() != LogicalType::INTEGER) {
					throw BinderException("HNSW index 'ef_search' must be an integer");
				}
				if (v.GetValue<int32_t>() < 1) {
					throw BinderException("HNSW index 'ef_search' must be at least 1");
				}
			} else if (StringUtil::CIEquals(k, "M")) {
				if (v.type() != LogicalType::INTEGER) {
					throw BinderException("HNSW index 'M' must be an integer");
				}
				if (v.GetValue<int32_t>() < 2) {
					throw BinderException("HNSW index 'M' must be at least 2");
				}
			} else if (StringUtil::CIEquals(k, "M0")) {
				if (v.type() != LogicalType::INTEGER) {
					throw BinderException("HNSW index 'M0' must be an integer");
				}
				if (v.GetValue<int32_t>() < 2) {
					throw BinderException("HNSW index 'M0' must be at least 2");
				}
			} else {
				throw BinderException("Unknown option for HNSW index: '%s'", k);
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

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {

		TryOptimize(input.context, plan);

		// Recursively traverse the children
		for (auto &child : plan->children) {
			Optimize(input, child);
		}
	};
};

//-------------------------------------------------------------
// Register
//-------------------------------------------------------------
void HNSWModule::RegisterPlanIndexCreate(DatabaseInstance &db) {
	// Register the optimizer extension
	db.config.AddExtensionOption("hnsw_enable_experimental_persistence",
	                             "experimental: enable creating HNSW indexes in persistent databases",
	                             LogicalType::BOOLEAN, Value::BOOLEAN(false));
	db.config.optimizer_extensions.push_back(HNSWIndexInsertionRewriter());
}

} // namespace duckdb