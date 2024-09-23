#include "aggregate_function_matcher.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"

#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"
#include "hnsw/hnsw_index_scan.hpp"


namespace duckdb {

//------------------------------------------------------------------------------
// Physical operator
//------------------------------------------------------------------------------
class PhysicalHNSWIndexJoin final : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;
public:
	// TODO: Pass estimated cardinality
	PhysicalHNSWIndexJoin(vector<LogicalType> &types, DuckTableEntry &table_p, HNSWIndex &hnsw_index_p, idx_t limit_p,
		const vector<column_t>& table_column_ids_p, const vector<LogicalType> &table_types_p)
		: PhysicalOperator(PhysicalOperatorType::EXTENSION, types, 0), table(table_p),
	hnsw_index(hnsw_index_p), limit(limit_p), logical_column_ids(table_column_ids_p), table_types(table_types_p) {
	}

public:
	string GetName() const override { return "HNSW_INDEX_JOIN"; }
	bool ParallelOperator() const override { return false; }
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
						   GlobalOperatorState &gstate, OperatorState &state) const override;
public:
	DuckTableEntry &table;
	HNSWIndex &hnsw_index;

	// The projection expressions (the struct pack of the rows from the right table)
	unique_ptr<Expression> proj_expression;

	// The matching vector on the lhs side
	unique_ptr<Expression> group_expression;

	idx_t limit;
	vector<column_t> logical_column_ids;
	vector<LogicalType> table_types;
};

class HNSWIndexJoinState : public OperatorState {
public:
	explicit HNSWIndexJoinState(ClientContext &context) : executor(context) { }

	idx_t input_idx = 0;

	ColumnFetchState fetch_state;
	TableScanState local_storage_state;
	vector<storage_t> phyiscal_column_ids;

	// Index scan state
	unique_ptr<IndexScanState> index_state;
	Vector row_ids = Vector(LogicalType::ROW_TYPE);


	SelectionVector match_sel;
	DataChunk table_chunk;
	//DataChunk proj_chunk;

	ExpressionExecutor executor;
};

unique_ptr<OperatorState> PhysicalHNSWIndexJoin::GetOperatorState(ExecutionContext &context) const {
	auto result = make_uniq<HNSWIndexJoinState>(context.client);

	auto &local_storage = LocalStorage::Get(context.client, table.catalog);
	result->phyiscal_column_ids.reserve(logical_column_ids.size());

	// Figure out the storage column ids from the projection expression
	for(auto &id : logical_column_ids) {
		storage_t col_id = id;
		if(id != DConstants::INVALID_INDEX) {
			col_id = table.GetColumn(LogicalIndex(id)).StorageOid();
		}
		result->phyiscal_column_ids.push_back(col_id);
	}

	// Initialize the storage scan state
	result->local_storage_state.Initialize(result->phyiscal_column_ids, nullptr);
	local_storage.InitializeScan(table.GetStorage(), result->local_storage_state.local_state, nullptr);

	// Initialize the table chunk
	vector<LogicalType> source_types;
	source_types.insert(source_types.end(), table_types.begin(), table_types.end());
	source_types.push_back(group_expression->return_type);
	// TODO: Limit to 2048
	result->table_chunk.Initialize(context.client, source_types, limit);
	result->match_sel = SelectionVector(limit);

	result->executor.AddExpression(*proj_expression);

	return std::move(result);
}

OperatorResultType PhysicalHNSWIndexJoin::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
					   GlobalOperatorState &gstate, OperatorState &ostate) const {

	// Get a reference to the state and transaction
	auto &state = ostate.Cast<HNSWIndexJoinState>();
	auto &transcation = DuckTransaction::Get(context.client, table.catalog);

	// Get the vector from the input
	input.Flatten();
	auto &array_vector = ArrayVector::GetEntry(input.data[0]);
	const auto array_data = FlatVector::GetData<float>(array_vector);

	// TODO: reserve more
	ListVector::Reserve(chunk.data[0], limit);
	auto result_list_vector = ListVector::GetEntry(chunk.data[0]);
	auto result_list_entries = ListVector::GetData(chunk.data[0]);

	while(state.input_idx < input.size()) {
		const auto array_ptr = array_data + state.input_idx * limit;
		state.index_state = hnsw_index.InitializeScan(array_ptr, limit, context.client);

		// Scan the index for row id's
		const auto match_count = hnsw_index.Scan(*state.index_state, state.row_ids);
		if(match_count == 0) {
			result_list_entries[state.input_idx] = list_entry_t { 0, 0 };
			state.input_idx++;
			continue;
		}

		for(idx_t j = 0; j < match_count; j++) {
			state.match_sel.set_index(j, state.input_idx);
		}

		// Scan the table for the matching rows
		table.GetStorage().Fetch(transcation, state.table_chunk, state.phyiscal_column_ids, state.row_ids, match_count, state.fetch_state);

		// Reference the current search vector
		state.table_chunk.data[state.table_chunk.data.size() - 1].Slice(input.data[0], state.match_sel, match_count);

		// Set the cardinality
		state.table_chunk.SetCardinality(match_count);

		// execute the expression
		state.executor.ExecuteExpression(state.table_chunk, result_list_vector);

		result_list_entries[state.input_idx] = list_entry_t { 0, match_count };
		ConstantVector::Reference(chunk.data[1], input.data[0], state.input_idx, input.size());
		chunk.SetCardinality(1);

		ListVector::SetListSize(chunk.data[0], match_count);

		state.input_idx++;
		return OperatorResultType::HAVE_MORE_OUTPUT;
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

//------------------------------------------------------------------------------
// Logical Operator
//------------------------------------------------------------------------------

class LogicalHNSWIndexJoin final : public LogicalExtensionOperator {
public:
	LogicalHNSWIndexJoin(idx_t table_index_p, DuckTableEntry &table_p, HNSWIndex &hnsw_index_p, idx_t limit_p, const vector<column_t>& table_column_ids_p, const vector<LogicalType>& table_types_p)
	    : table_index(table_index_p), table(table_p), hnsw_index(hnsw_index_p), limit(limit_p), table_column_ids(table_column_ids_p), table_types(table_types_p) {
	}

public:
	void ResolveTypes() override;
	void ResolveColumnBindings(ColumnBindingResolver &res, vector<ColumnBinding> &bindings) override;
	vector<ColumnBinding> GetColumnBindings() override;

	string GetExtensionName() const override;
	unique_ptr<PhysicalOperator> CreatePlan(ClientContext &context, PhysicalPlanGenerator &generator) override;

	vector<idx_t> GetTableIndex() const override {
		return {0};
	}
public:
	idx_t table_index;
	DuckTableEntry &table;
	HNSWIndex &hnsw_index;
	unique_ptr<Expression> projection_expression;
	idx_t limit;

	vector<column_t> table_column_ids;
	vector<LogicalType> table_types;
};

//------------------------------------------------------------------------------
// Methods
//------------------------------------------------------------------------------

void LogicalHNSWIndexJoin::ResolveTypes() {
	types.push_back(LogicalType::LIST(projection_expression->return_type));
	D_ASSERT(expressions.size() == 1);
	types.push_back(expressions[0]->return_type);
}

vector<ColumnBinding> LogicalHNSWIndexJoin::GetColumnBindings() {
	return GenerateColumnBindings(table_index, types.size());
}

void LogicalHNSWIndexJoin::ResolveColumnBindings(ColumnBindingResolver &res, vector<ColumnBinding> &bindings) {
	for (auto &child : children) {
		res.VisitOperator(*child);
	}
	LogicalOperatorVisitor::EnumerateExpressions(*this, [&](unique_ptr<Expression> *child) { res.VisitExpression(child); });
	bindings = GetColumnBindings();
}

string LogicalHNSWIndexJoin::GetExtensionName() const {
	return "hnsw_index_join";
}

unique_ptr<PhysicalOperator> LogicalHNSWIndexJoin::CreatePlan(ClientContext &context, PhysicalPlanGenerator &generator) {
	auto result = make_uniq<PhysicalHNSWIndexJoin>(types, table, hnsw_index, limit, table_column_ids, table_types);
	result->children.push_back(generator.CreatePlan(std::move(children[0])));
	// Convert the projection expression

	result->proj_expression = projection_expression->Copy();
	D_ASSERT(expressions.size() == 1);
	result->group_expression = expressions[0]->Copy();
	return std::move(result);
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

	auto vector_matcher = make_uniq<ExpressionMatcher>();
	vector_matcher->type = make_uniq<SpecificTypeMatcher>(LogicalType::ARRAY(LogicalType::FLOAT, vector_size));
	distance_matcher->matchers.push_back(std::move(vector_matcher)); // The vector to match

	min_by_matcher.matchers.push_back(make_uniq<ExpressionMatcher>()); // Dont care about the column
	min_by_matcher.matchers.push_back(std::move(distance_matcher));
	min_by_matcher.matchers.push_back(make_uniq<ConstantExpressionMatcher>()); // The k value

	return min_by_matcher.Match(agg_expr, bindings);
}


struct ProjectionMerger final : LogicalOperatorVisitor {
	LogicalProjection *child_proj = nullptr;

	void VisitOperator(LogicalOperator &op) override {
		for(auto &child : op.children) {
			while(child->type == LogicalOperatorType::LOGICAL_PROJECTION) {
				child_proj = &child->Cast<LogicalProjection>();
				VisitOperatorExpressions(op);
				// Replace the child with its own child
				child = std::move(child->children.back());
			}
		}
		// Visit the children
		VisitOperatorChildren(op);
	}

	void VisitExpression(unique_ptr<Expression> *expression) override {
		if(expression->get()->type == ExpressionType::BOUND_COLUMN_REF) {
			const auto &ref = expression->get()->Cast<BoundColumnRefExpression>();
			if(ref.binding.table_index == child_proj->table_index) {
				// Replace the column reference with the expression
				*expression = child_proj->expressions[ref.binding.column_index]->Copy();
				return;
			}
		}

		VisitExpressionChildren(**expression);
	}

	static void MergeProjections(LogicalOperator &op) {
		ProjectionMerger inliner;
		inliner.VisitOperator(op);
	}
};

//------------------------------------------------------------------------------
// Main Optimizer
//------------------------------------------------------------------------------
// This optimizer rewrites
//
//	AGG(MIN_BY(t1.col1, distance_func(t1.col2, query_vector), k)) <- TABLE_SCAN(t1)
//  =>
//	AGG(LIST(col1 ORDER BY distance_func(col2, query_vector) ASC)) <- HNSW_INDEX_SCAN(t1, query_vector, k)
//

class HNSWJoinOptimizer : public OptimizerExtension {
public:
	HNSWJoinOptimizer() {
		optimize_function = Optimize;
	}

	static bool TryOptimize(Binder &binder, ClientContext &context, unique_ptr<LogicalOperator> &root, unique_ptr<LogicalOperator> &plan) {
		// Look for a Aggregate operator
		if(plan->children.empty()) {
			return false;
		}

		optional_idx child_idx = optional_idx::Invalid();
		for(idx_t i = 0; i < plan->children.size(); i++) {
			if(plan->children[i]->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
				child_idx = i;
			}
		}
		if(!child_idx.IsValid()) {
			return false;
		}
		// Look for a expression that is a distance expression
		auto &agg = plan->children[child_idx.GetIndex()]->Cast<LogicalAggregate>();
		if (agg.groups.size() != 1 || agg.expressions.size() != 1) {
			return false;
		}

		// we need the aggregate to be on top of a projection
		if (agg.children.size() != 1) {
			return false;
		}

		if(agg.children.back().get()->type != LogicalOperatorType::LOGICAL_PROJECTION) {
			return false;
		}

		// Save the aggregate bindings
		// This is going to be
		// 0. the grouping expression
		//	basically, the LHS vector column: s_vec
		// 1. the resulting expression
		//	basically, the result of the aggregate function: min_by(struct_pack(...), score, 3)
		auto old_agg_bindings = agg.GetColumnBindings();

		// What we need to return is: the same expression as passed in the min_by function.
		// This is going to be a struct of columns from the right table.
		// This is the same as bindings[1] we extract down below.
		// Except we also get the "score" distance metric.
		// But that will always be the last column.


		// Squash the projections
		ProjectionMerger::MergeProjections(agg);

		// There can be an arbitrary amount of chained projections here
		LogicalOperator* current = agg.children[0].get();
		while(current->type == LogicalOperatorType::LOGICAL_PROJECTION) {
			auto &proj = current->Cast<LogicalProjection>();
			if (proj.children.size() != 1) {
				return false;
			}
			current = proj.children[0].get();
		}

		if(current->type != LogicalOperatorType::LOGICAL_CROSS_PRODUCT || current->children.size() != 2) {
			return false;
		}
		auto &cp = current->Cast<LogicalCrossProduct>();
		// The cross product needs to be on top of a table scan

		if(cp.children[0]->type != LogicalOperatorType::LOGICAL_GET) {
			return false;
		}

		auto &table_scan_ptr = cp.children[0];
		auto &table_scan = table_scan_ptr->Cast<LogicalGet>();
		if(table_scan.function.name != "seq_scan") {
			return false;
		}

		if(cp.children[1]->type != LogicalOperatorType::LOGICAL_DELIM_GET) {
			return false;
		}
		auto &delim_scan_ptr = cp.children[1];
		auto &delim_scan = delim_scan_ptr->Cast<LogicalDelimGet>();


		// Get the table
		auto &table = *table_scan.GetTable();
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
			if (!hnsw_index.CanRewriteIndexExpression(table_scan, *index_expr)) {
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
			// bindings[4] = the matching vector column
			// bindings[5] = the k value

			const auto &distance_func = bindings[2].get().Cast<BoundFunctionExpression>();
			if (!hnsw_index.MatchesDistanceFunction(distance_func.function.name)) {
				return false;
			}

			const auto k_limit = bindings[5].get().Cast<BoundConstantExpression>().value.GetValue<int32_t>();
			bind_data = make_uniq<HNSWIndexScanBindData>(duck_table, hnsw_index, k_limit, nullptr);
			return true;
		});

		if (!bind_data) {
			// No index found
			return false;
		}


		// Ok, so here's whats going to happen:
		// We are going to remove the table scan entirely. The index join will take care of that.
		// To that end, we will only return two bindings: the group and the list of rows (structs) from the what used to
		// be the table scan. We can move in the expressions into the index join, and replace their bindings with physical bindings
		// into a temporary execution chunk that we populate right after scanning the index.

		// Before we create the index join, resolve the physical bindings of the agg projection
		// We will mimic the cross product layout within the HNSW index join
		const auto group_expr = bindings[4].get().Copy();

		ColumnBindingResolver resolver;
		resolver.VisitOperator(agg);

		auto index_join = make_uniq<LogicalHNSWIndexJoin>(binder.GenerateTableIndex(),
			duck_table, bind_data->index.Cast<HNSWIndex>(), bind_data->limit, table_scan.GetColumnIds(), table_scan.types);
		//index_join->children.push_back(std::move(table_scan_ptr));
		index_join->children.push_back(std::move(delim_scan_ptr));

		const auto &projection_expr = bindings[1].get();
		index_join->projection_expression = projection_expr.Copy();
		index_join->types.push_back(LogicalType::LIST(projection_expr.return_type));


		index_join->types.push_back(group_expr->return_type);
		index_join->expressions.push_back(group_expr->Copy());

		// Also, replace all the column bindings with our new bindings
		auto new_agg_bindings = index_join->GetColumnBindings();
		auto &new_types = index_join->types;

		// The aggregate will have first the groups, then the expressions, then the grouping functions.
		ColumnBindingReplacer replacer;
		replacer.replacement_bindings.emplace_back(old_agg_bindings[1], new_agg_bindings[0], new_types[0]);
		replacer.replacement_bindings.emplace_back(old_agg_bindings[0], new_agg_bindings[1], new_types[1]);

		// Replace this part of the plan with our index join
		plan->children[child_idx.GetIndex()] = std::move(index_join);

		// Make the plan consistent again, starting from the root
		// TODO: Setup a stop
		replacer.VisitOperator(*root);

		return true;
	}

	static void OptimizeRecursive(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &root, unique_ptr<LogicalOperator> &plan) {
		if (!TryOptimize(input.optimizer.binder, input.context, root, plan)) {
			// Recursively optimize the children
			for (auto &child : plan->children) {
				OptimizeRecursive(input, root, child);
			}
		}
	}

	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
		OptimizeRecursive(input, plan, plan);
	}
};

void HNSWModule::RegisterJoinOptimizer(DatabaseInstance &db) {
	// Register the JoinOptimizer
	db.config.optimizer_extensions.push_back(HNSWJoinOptimizer());
}

} // namespace duckdb