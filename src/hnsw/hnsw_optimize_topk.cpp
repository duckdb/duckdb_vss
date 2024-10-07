#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"

#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"
#include "hnsw/hnsw_index_scan.hpp"

namespace duckdb {

//------------------------------------------------------------------------------
// Optimizer Helpers
//------------------------------------------------------------------------------

static unique_ptr<Expression> CreateListOrderByExpr(ClientContext &context, unique_ptr<Expression> elem_expr,
                                                    unique_ptr<Expression> order_expr,
                                                    unique_ptr<Expression> filter_expr) {
	auto func_entry =
	    Catalog::GetEntry<AggregateFunctionCatalogEntry>(context, "", "", "list", OnEntryNotFound::RETURN_NULL);
	if (!func_entry) {
		return nullptr;
	}

	auto func = func_entry->functions.GetFunctionByOffset(0);
	vector<unique_ptr<Expression>> arguments;
	arguments.push_back(std::move(elem_expr));

	auto agg_bind_data = func.bind(context, func, arguments);
	auto new_agg_expr =
	    make_uniq<BoundAggregateExpression>(func, std::move(arguments), std::move(std::move(filter_expr)),
	                                        std::move(agg_bind_data), AggregateType::NON_DISTINCT);

	// We also need to order the list items by the distance
	BoundOrderByNode order_by_node(OrderType::ASCENDING, OrderByNullType::NULLS_LAST, std::move(order_expr));
	new_agg_expr->order_bys = make_uniq<BoundOrderModifier>();
	new_agg_expr->order_bys->orders.push_back(std::move(order_by_node));

	return new_agg_expr;
}

//------------------------------------------------------------------------------
// Main Optimizer
//------------------------------------------------------------------------------
// This optimizer rewrites
//
//	AGG(MIN_BY(t1.col1, distance_func(t1.col2, query_vector), k)) <- TABLE_SCAN(t1)
//  =>
//	AGG(LIST(col1 ORDER BY distance_func(col2, query_vector) ASC)) <- HNSW_INDEX_SCAN(t1, query_vector, k)
//

class HNSWTopKOptimizer : public OptimizerExtension {
public:
	HNSWTopKOptimizer() {
		optimize_function = Optimize;
	}

	static bool TryOptimize(Binder &binder, ClientContext &context, unique_ptr<LogicalOperator> &plan) {
		// Look for a Aggregate operator
		if (plan->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
			return false;
		}
		// Look for a expression that is a distance expression
		auto &agg = plan->Cast<LogicalAggregate>();
		if (!agg.groups.empty() || agg.expressions.size() != 1) {
			return false;
		}

		auto &agg_expr = agg.expressions[0];
		if (agg_expr->type != ExpressionType::BOUND_AGGREGATE) {
			return false;
		}
		auto &agg_func_expr = agg_expr->Cast<BoundAggregateExpression>();
		if (agg_func_expr.function.name != "min_by") {
			return false;
		}
		if (agg_func_expr.children.size() != 3) {
			return false;
		}
		if (agg_func_expr.children[2]->type != ExpressionType::VALUE_CONSTANT) {
			return false;
		}
		const auto &col_expr = agg_func_expr.children[0];
		const auto &dist_expr = agg_func_expr.children[1];
		const auto &limit_expr = agg_func_expr.children[2];

		// we need the aggregate to be on top of a projection
		if (agg.children.size() != 1) {
			return false;
		}

		// we also need the projection to be directly on top of a table scan that has a hnsw index
		if (agg.children[0]->type != LogicalOperatorType::LOGICAL_GET) {
			return false;
		}

		auto &get = agg.children[0]->Cast<LogicalGet>();
		if (get.function.name != "seq_scan") {
			return false;
		}

		// Get the table
		auto &table = *get.GetTable();
		if (!table.IsDuckTable()) {
			return false;
		}

		auto &duck_table = table.Cast<DuckTableEntry>();
		auto &table_info = *table.GetStorage().GetDataTableInfo();

		unique_ptr<HNSWIndexScanBindData> bind_data = nullptr;
		vector<reference<Expression>> bindings;

		table_info.GetIndexes().BindAndScan<HNSWIndex>(context, table_info, [&](HNSWIndex &hnsw_index) {
			// Reset the bindings
			bindings.clear();

			// Check that the projection expression is a distance function that matches the index
			if (!hnsw_index.TryMatchDistanceFunction(dist_expr, bindings)) {
				return false;
			}
			// Check that the HNSW index actually indexes the expression
			unique_ptr<Expression> index_expr;
			if (!hnsw_index.TryBindIndexExpression(get, index_expr)) {
				return false;
			}

			// Now, ensure that one of the bindings is a constant vector, and the other our index expression
			auto &const_expr_ref = bindings[1];
			auto &index_expr_ref = bindings[2];

			if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT || !index_expr->Equals(index_expr_ref)) {
				// Swap the bindings and try again
				std::swap(const_expr_ref, index_expr_ref);
				if (const_expr_ref.get().type != ExpressionType::VALUE_CONSTANT ||
				    !index_expr->Equals(index_expr_ref)) {
					// Nope, not a match, we can't optimize.
					return false;
				}
			}

			const auto vector_size = hnsw_index.GetVectorSize();
			const auto &matched_vector = const_expr_ref.get().Cast<BoundConstantExpression>().value;

			auto query_vector = make_unsafe_uniq_array<float>(vector_size);
			auto vector_elements = ArrayValue::GetChildren(matched_vector);
			for (idx_t i = 0; i < vector_size; i++) {
				query_vector[i] = vector_elements[i].GetValue<float>();
			}
			const auto k_limit = limit_expr->Cast<BoundConstantExpression>().value.GetValue<int32_t>();
			if (k_limit <= 0 || k_limit >= STANDARD_VECTOR_SIZE) {
				return false;
			}
			bind_data = make_uniq<HNSWIndexScanBindData>(duck_table, hnsw_index, k_limit, std::move(query_vector));
			return true;
		});

		if (!bind_data) {
			// No index found
			return false;
		}

		// Replace the aggregate with a index scan + projection
		get.function = HNSWIndexScanFunction::GetFunction();
		const auto cardinality = get.function.cardinality(context, bind_data.get());
		get.has_estimated_cardinality = cardinality->has_estimated_cardinality;
		get.estimated_cardinality = cardinality->estimated_cardinality;
		get.bind_data = std::move(bind_data);

		// Replace the aggregate with a list() aggregate function ordered by the distance
		agg.expressions[0] = CreateListOrderByExpr(context, col_expr->Copy(), dist_expr->Copy(),
		                                           agg_func_expr.filter ? agg_func_expr.filter->Copy() : nullptr);
		return true;
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		if (!TryOptimize(input.optimizer.binder, input.context, plan)) {
			// Recursively optimize the children
			for (auto &child : plan->children) {
				Optimize(input, child);
			}
		}
	}
};

void HNSWModule::RegisterTopKOptimizer(DatabaseInstance &db) {
	// Register the TopKOptimizer
	db.config.optimizer_extensions.push_back(HNSWTopKOptimizer());
}

} // namespace duckdb