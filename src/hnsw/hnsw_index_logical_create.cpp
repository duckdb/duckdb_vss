#include "hnsw/hnsw_index_logical_create.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/operator/logical_create_index.hpp"

#include "hnsw/hnsw_index.hpp"
#include "hnsw/hnsw_index_physical_create.hpp"

namespace duckdb {

LogicalCreateHNSWIndex::LogicalCreateHNSWIndex(unique_ptr<CreateIndexInfo> info_p,
                                               vector<unique_ptr<Expression>> expressions_p, TableCatalogEntry &table_p)
    : LogicalExtensionOperator(), info(std::move(info_p)), table(table_p) {
	for (auto &expr : expressions_p) {
		this->unbound_expressions.push_back(expr->Copy());
	}
	this->expressions = std::move(expressions_p);
}

void LogicalCreateHNSWIndex::ResolveTypes() {
	types.emplace_back(LogicalType::BIGINT);
}

void LogicalCreateHNSWIndex::ResolveColumnBindings(ColumnBindingResolver &res, vector<ColumnBinding> &bindings) {
	bindings = LogicalOperator::GenerateColumnBindings(0, table.GetColumns().LogicalColumnCount());

	// Visit the operator's expressions
	LogicalOperatorVisitor::EnumerateExpressions(*this,
	                                             [&](unique_ptr<Expression> *child) { res.VisitExpression(child); });
}

string LogicalCreateHNSWIndex::GetExtensionName() const {
	return "hnsw_create_index";
}

unique_ptr<PhysicalOperator> LogicalCreateHNSWIndex::CreatePlan(ClientContext &context,
                                                                PhysicalPlanGenerator &generator) {

	auto &op = *this;

	// generate a physical plan for the parallel index creation which consists of the following operators
	// table scan - projection (for expression execution) - filter (NOT NULL) - order - create index
	D_ASSERT(op.children.size() == 1);
	auto table_scan = generator.CreatePlan(std::move(op.children[0]));

	// Validate that we only have one expression
	if (op.unbound_expressions.size() != 1) {
		throw BinderException("HNSW indexes can only be created over a single column of keys.");
	}

	auto &expr = op.unbound_expressions[0];

	// Validate that the expression does not have side effects
	if (!expr->IsConsistent()) {
		throw BinderException("HNSW index keys cannot contain expressions with side "
		                      "effects.");
	}

	// Validate that we have the right type of expression (float array)
	auto &type = expr->return_type;
	if (type.id() != LogicalTypeId::ARRAY || ArrayType::GetChildType(type).id() != LogicalTypeId::FLOAT) {
		throw BinderException("HNSW index can only be created over FLOAT[N] keys.");
	}

	// Assert that we got the right index type
	D_ASSERT(op.info->index_type == HNSWIndex::TYPE_NAME);

	// table scan operator for index key columns and row IDs
	generator.dependencies.AddDependency(op.table);

	D_ASSERT(op.info->scan_types.size() - 1 <= op.info->names.size());
	D_ASSERT(op.info->scan_types.size() - 1 <= op.info->column_ids.size());

	// projection to execute expressions on the key columns

	vector<LogicalType> new_column_types;
	vector<unique_ptr<Expression>> select_list;
	for (idx_t i = 0; i < op.expressions.size(); i++) {
		new_column_types.push_back(op.expressions[i]->return_type);
		select_list.push_back(std::move(op.expressions[i]));
	}
	new_column_types.emplace_back(LogicalType::ROW_TYPE);
	select_list.push_back(make_uniq<BoundReferenceExpression>(LogicalType::ROW_TYPE, op.info->scan_types.size() - 1));

	auto projection = make_uniq<PhysicalProjection>(new_column_types, std::move(select_list), op.estimated_cardinality);
	projection->children.push_back(std::move(table_scan));

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

	auto null_filter =
	    make_uniq<PhysicalFilter>(std::move(filter_types), std::move(filter_select_list), op.estimated_cardinality);
	null_filter->types.emplace_back(LogicalType::ROW_TYPE);
	null_filter->children.push_back(std::move(projection));

	auto physical_create_index =
	    make_uniq<PhysicalCreateHNSWIndex>(op, op.table, op.info->column_ids, std::move(op.info),
	                                       std::move(op.unbound_expressions), op.estimated_cardinality);

	physical_create_index->children.push_back(std::move(null_filter));

	return std::move(physical_create_index);
}

} // namespace duckdb
