#include "duckdb/planner/operator/logical_create_index.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"

#include "duckdb/parser/parsed_data/create_index_info.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"

#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"

#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"
#include "hnsw/hnsw_index_physical_create.hpp"

namespace duckdb {

unique_ptr<PhysicalOperator> HNSWIndex::CreatePlan(PlanIndexInput &input) {
	auto &create_index = input.op;
	auto &context = input.context;

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

	// projection to execute expressions on the key columns

	vector<LogicalType> new_column_types;
	vector<unique_ptr<Expression>> select_list;
	for (auto &expression : create_index.expressions) {
		new_column_types.push_back(expression->return_type);
		select_list.push_back(std::move(expression));
	}
	new_column_types.emplace_back(LogicalType::ROW_TYPE);
	select_list.push_back(
	    make_uniq<BoundReferenceExpression>(LogicalType::ROW_TYPE, create_index.info->scan_types.size() - 1));

	auto projection =
	    make_uniq<PhysicalProjection>(new_column_types, std::move(select_list), create_index.estimated_cardinality);
	projection->children.push_back(std::move(input.table_scan));

	// filter operator for IS_NOT_NULL on each key column
	vector<LogicalType> filter_types;
	vector<unique_ptr<Expression>> filter_select_list;

	for (idx_t i = 0; i < new_column_types.size() - 1; i++) {
		filter_types.push_back(new_column_types[i]);
		auto is_not_null_expr =
		    make_uniq<BoundOperatorExpression>(ExpressionType::OPERATOR_IS_NOT_NULL, LogicalType::BOOLEAN);
		auto bound_ref = make_uniq<BoundReferenceExpression>(new_column_types[i], i);
		is_not_null_expr->children.push_back(std::move(bound_ref));
		filter_select_list.push_back(std::move(is_not_null_expr));
	}

	auto null_filter = make_uniq<PhysicalFilter>(std::move(filter_types), std::move(filter_select_list),
	                                             create_index.estimated_cardinality);
	null_filter->types.emplace_back(LogicalType::ROW_TYPE);
	null_filter->children.push_back(std::move(projection));

	auto physical_create_index = make_uniq<PhysicalCreateHNSWIndex>(
	    create_index.types, create_index.table, create_index.info->column_ids, std::move(create_index.info),
	    std::move(create_index.unbound_expressions), create_index.estimated_cardinality);

	physical_create_index->children.push_back(std::move(null_filter));

	return std::move(physical_create_index);
}

} // namespace duckdb