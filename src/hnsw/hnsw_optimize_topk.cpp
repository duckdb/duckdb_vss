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

#include "aggregate_function_matcher.hpp"

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

// bindings[0] = the aggregate_function
// bindings[1] = the column ref
// bindings[2] = the distance function
// bindings[3] = the arg ref
// bindings[4] = the matched vector
// bindings[5] = the k value
static bool MatchDistanceFunction(vector<reference<Expression>> &bindings, Expression &agg_expr, Expression &column_ref,
                                  idx_t vector_size) {

	AggregateFunctionExpressionMatcher min_by_matcher;
	min_by_matcher.function = make_uniq<SpecificFunctionMatcher>("min_by");
	min_by_matcher.policy = SetMatcher::Policy::ORDERED;

	unordered_set<string> distance_functions = {
	    "array_distance", "<->", "array_cosine_distance", "<=>", "array_negative_inner_product", "<#>"};

	auto distance_matcher = make_uniq<FunctionExpressionMatcher>();
	distance_matcher->function = make_uniq<ManyFunctionMatcher>(distance_functions);
	distance_matcher->expr_type = make_uniq<SpecificExpressionTypeMatcher>(ExpressionType::BOUND_FUNCTION);
	distance_matcher->policy = SetMatcher::Policy::UNORDERED;
	distance_matcher->matchers.push_back(make_uniq<ExpressionEqualityMatcher>(column_ref));

	auto vector_matcher = make_uniq<ConstantExpressionMatcher>();
	vector_matcher->type = make_uniq<SpecificTypeMatcher>(LogicalType::ARRAY(LogicalType::FLOAT, vector_size));
	distance_matcher->matchers.push_back(std::move(vector_matcher)); // The vector to match

	min_by_matcher.matchers.push_back(make_uniq<ExpressionMatcher>()); // Dont care about the column
	min_by_matcher.matchers.push_back(std::move(distance_matcher));
	min_by_matcher.matchers.push_back(make_uniq<ConstantExpressionMatcher>()); // The k value

	return min_by_matcher.Match(agg_expr, bindings);
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
			// Check that the HNSW index actually indexes the expression
			const auto index_expr = hnsw_index.unbound_expressions[0]->Copy();
			if (!hnsw_index.CanRewriteIndexExpression(get, *index_expr)) {
				return false;
			}

			const auto vector_size = hnsw_index.GetVectorSize();

			// Reset the bindings
			bindings.clear();
			if (!MatchDistanceFunction(bindings, *agg.expressions[0], *index_expr, vector_size)) {
				return false;
			}
			// bindings[0] = the aggregate_function
			// bindings[1] = the column ref
			// bindings[2] = the distance function
			// bindings[3] = the arg ref
			// bindings[4] = the matched vector
			// bindings[5] = the k value

			const auto &distance_func = bindings[2].get().Cast<BoundFunctionExpression>();
			if (!hnsw_index.MatchesDistanceFunction(distance_func.function.name)) {
				return false;
			}

			const auto &matched_vector = bindings[4].get().Cast<BoundConstantExpression>().value;
			auto query_vector = make_unsafe_uniq_array<float>(vector_size);
			auto vector_elements = ArrayValue::GetChildren(matched_vector);
			for (idx_t i = 0; i < vector_size; i++) {
				query_vector[i] = vector_elements[i].GetValue<float>();
			}

			const auto k_limit = bindings[5].get().Cast<BoundConstantExpression>().value.GetValue<int32_t>();
			bind_data = make_uniq<HNSWIndexScanBindData>(duck_table, hnsw_index, k_limit, std::move(query_vector));

			return true;
		});

		if (!bind_data) {
			// No index found
			return false;
		}

		const auto &agg_expr = bindings[0].get().Cast<BoundAggregateExpression>();
		const auto &col_expr = bindings[1].get();
		const auto &distance_func = bindings[2].get().Cast<BoundFunctionExpression>();

		// Replace the aggregate with a index scan + projection
		get.function = HNSWIndexScanFunction::GetFunction();
		const auto cardinality = get.function.cardinality(context, bind_data.get());
		get.has_estimated_cardinality = cardinality->has_estimated_cardinality;
		get.estimated_cardinality = cardinality->estimated_cardinality;
		get.bind_data = std::move(bind_data);

		// Replace the aggregate with a list() aggregate function ordered by the distance
		agg.expressions[0] = CreateListOrderByExpr(context, col_expr.Copy(), distance_func.Copy(),
		                                           agg_expr.filter ? agg_expr.filter->Copy() : nullptr);
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