#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/optimizer/optimizer.hpp"

#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"

namespace duckdb {

//------------------------------------------------------------------------------
// Rewrite rules
//------------------------------------------------------------------------------
// This optimizer rewrites expressions of the form:
//	(1.0 - array_cosine_similarity)		=>	(array_cosine_distance)
//	(-array_inner_product)				=>	(array_negative_inner_product)

class CosineDistanceRule final : public Rule {
public:
	explicit CosineDistanceRule(ExpressionRewriter &rewriter);
	unique_ptr<Expression> Apply(LogicalOperator &op, vector<reference<Expression>> &bindings, bool &changes_made,
	                             bool is_root) override;
};

CosineDistanceRule::CosineDistanceRule(ExpressionRewriter &rewriter) : Rule(rewriter) {
	auto func = make_uniq<FunctionExpressionMatcher>();
	func->matchers.push_back(make_uniq<ExpressionMatcher>());
	func->matchers.push_back(make_uniq<ExpressionMatcher>());
	func->policy = SetMatcher::Policy::UNORDERED;
	func->function = make_uniq<SpecificFunctionMatcher>("array_cosine_similarity");

	auto op = make_uniq<FunctionExpressionMatcher>();
	op->matchers.push_back(make_uniq<ConstantExpressionMatcher>());
	op->matchers[0]->type = make_uniq<SpecificTypeMatcher>(LogicalType::FLOAT);
	op->matchers.push_back(std::move(func));
	op->policy = SetMatcher::Policy::ORDERED;
	op->function = make_uniq<SpecificFunctionMatcher>("-");
	op->type = make_uniq<SpecificTypeMatcher>(LogicalType::FLOAT);

	root = std::move(op);
}

unique_ptr<Expression> CosineDistanceRule::Apply(LogicalOperator &op, vector<reference<Expression>> &bindings,
                                                 bool &changes_made, bool is_root) {
	// auto &root_expr = bindings[0].get().Cast<BoundFunctionExpression>();
	const auto &const_expr = bindings[1].get().Cast<BoundConstantExpression>();
	auto &similarity_expr = bindings[2].get().Cast<BoundFunctionExpression>();

	if (!const_expr.value.IsNull() && const_expr.value.GetValue<float>() == 1.0) {
		// Create the new array_cosine_distance function
		vector<unique_ptr<Expression>> args;
		vector<LogicalType> arg_types;
		arg_types.push_back(similarity_expr.children[0]->return_type);
		arg_types.push_back(similarity_expr.children[1]->return_type);
		args.push_back(std::move(similarity_expr.children[0]));
		args.push_back(std::move(similarity_expr.children[1]));

		auto &context = GetContext();
		auto func_entry = Catalog::GetEntry<ScalarFunctionCatalogEntry>(context, "", "", "array_cosine_distance",
		                                                                OnEntryNotFound::RETURN_NULL);

		if (!func_entry) {
			return nullptr;
		}

		changes_made = true;
		auto func = func_entry->functions.GetFunctionByArguments(context, arg_types);
		return make_uniq<BoundFunctionExpression>(similarity_expr.return_type, func, std::move(args), nullptr);
	}
	return nullptr;
}

//------------------------------------------------------------------------------
// Optimizer
//------------------------------------------------------------------------------
class HNSWExprOptimizer : public OptimizerExtension {
public:
	HNSWExprOptimizer() {
		optimize_function = Optimize;
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		ExpressionRewriter rewriter(input.context);
		rewriter.rules.push_back(make_uniq<CosineDistanceRule>(rewriter));
		rewriter.VisitOperator(*plan);
	}
};

void HNSWModule::RegisterExprOptimizer(DatabaseInstance &db) {
	// Register the TopKOptimizer
	db.config.optimizer_extensions.push_back(HNSWExprOptimizer());
}

} // namespace duckdb