#include "aggregate_function_matcher.hpp"
#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_window.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"

#include <duckdb/planner/expression_iterator.hpp>

namespace duckdb {

//------------------------------------------------------------------------------
// Physical Operator
//------------------------------------------------------------------------------

class PhysicalHNSWIndexJoin final : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

	PhysicalHNSWIndexJoin(const vector<LogicalType> &types_p, const idx_t estimated_cardinality, DuckTableEntry &table_p, HNSWIndex &hnsw_index_p, const idx_t limit_p)
		: PhysicalOperator(TYPE, types_p, estimated_cardinality), table(table_p), hnsw_index(hnsw_index_p), limit(limit_p) {
	}
public:
	string GetName() const override;
	bool ParallelOperator() const override;
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
						   GlobalOperatorState &gstate, OperatorState &state) const override;

public:
	DuckTableEntry &table;
	HNSWIndex &hnsw_index;
	idx_t limit;

	vector<column_t> lhs_column_ids;
	vector<idx_t> lhs_projection_ids;
	idx_t lhs_vector_column;
	idx_t rhs_vector_column;
};

string PhysicalHNSWIndexJoin::GetName() const {
	return "HNSW_INDEX_JOIN";
}

bool PhysicalHNSWIndexJoin::ParallelOperator() const {
	return false;
}

// TODO: Assert at most k = SVS
class HNSWIndexJoinState final : public OperatorState {
public:
	idx_t input_idx = 0;

	ColumnFetchState fetch_state;
	TableScanState local_storage_state;
	vector<storage_t> phyiscal_column_ids;

	// Index scan state
	unique_ptr<IndexScanState> index_state;
	SelectionVector match_sel;
};

unique_ptr<OperatorState> PhysicalHNSWIndexJoin::GetOperatorState(ExecutionContext &context) const {
	auto result = make_uniq<HNSWIndexJoinState>();

	auto &local_storage = LocalStorage::Get(context.client, table.catalog);
	result->phyiscal_column_ids.reserve(lhs_column_ids.size());

	// Figure out the storage column ids from the projection expression
	for(auto &id : lhs_column_ids) {
		storage_t col_id = id;
		if(id != DConstants::INVALID_INDEX) {
			col_id = table.GetColumn(LogicalIndex(id)).StorageOid();
		}
		result->phyiscal_column_ids.push_back(col_id);
	}

	// Initialize selection vector
	result->match_sel.Initialize();

	// Initialize the storage scan state
	result->local_storage_state.Initialize(result->phyiscal_column_ids, nullptr);
	local_storage.InitializeScan(table.GetStorage(), result->local_storage_state.local_state, nullptr);

	// Initialize the index scan state
	result->index_state = hnsw_index.InitializeMultiScan(context.client);

	return std::move(result);
}

OperatorResultType PhysicalHNSWIndexJoin::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
												   GlobalOperatorState &gstate, OperatorState &ostate) const {
	auto &state = ostate.Cast<HNSWIndexJoinState>();
	auto &transcation = DuckTransaction::Get(context.client, table.catalog);

	// TODO: Assert that limit is at most 2048

	// TODO: dont flatten
	input.Flatten();

	auto &rhs_vector_vector = input.data[rhs_vector_column];
	auto &rhs_vector_child = ArrayVector::GetEntry(rhs_vector_vector);
	const auto rhs_vector_size = ArrayType::GetSize(rhs_vector_vector.GetType());
	const auto rhs_vector_ptr = FlatVector::GetData<float>(rhs_vector_child);

	hnsw_index.ResetMultiScan(*state.index_state);

	// How many batches are we going to process?
	const auto batch_count = MinValue(input.size() - state.input_idx, STANDARD_VECTOR_SIZE / limit);
	idx_t output_idx = 0;
	for(idx_t batch_idx = 0; batch_idx < batch_count; batch_idx++, state.input_idx++) {

		// Get the next batch
		const auto rhs_vector_data = rhs_vector_ptr + batch_idx * rhs_vector_size;

		// Scan the index for row ids
		const auto match_count = hnsw_index.ExecuteMultiScan(*state.index_state, rhs_vector_data, limit);
		for(idx_t i = 0; i < match_count; i++) {
			state.match_sel.set_index(output_idx++, batch_idx);
		}
	}

	const auto &row_ids = hnsw_index.GetMultiScanResult(*state.index_state);

	// Execute one big fetch for the LHS
	table.GetStorage().Fetch(transcation, chunk, state.phyiscal_column_ids, row_ids, output_idx, state.fetch_state);

	// Now slice the chunk so that we include the rhs too
	chunk.Slice(input, state.match_sel, output_idx, state.phyiscal_column_ids.size());

	// Set the cardinality
	chunk.SetCardinality(output_idx);

	if(state.input_idx == input.size()) {
		state.input_idx = 0;
		return OperatorResultType::NEED_MORE_INPUT;
	}

	return OperatorResultType::HAVE_MORE_OUTPUT;
}

//------------------------------------------------------------------------------
// Logical Operator
//------------------------------------------------------------------------------

class LogicalHNSWIndexJoin final : public LogicalExtensionOperator {
public:
	explicit LogicalHNSWIndexJoin(const idx_t table_index_p, DuckTableEntry &table_p, HNSWIndex &hnsw_index_p, const idx_t limit_p)
		: table_index(table_index_p), table(table_p), hnsw_index(hnsw_index_p), limit(limit_p) {
	}
public:
	string GetName() const override;
	void ResolveTypes() override;
	vector<ColumnBinding> GetColumnBindings() override;
	vector<ColumnBinding> GetLeftBindings();
	vector<ColumnBinding> GetRightBindings();
	unique_ptr<PhysicalOperator> CreatePlan(ClientContext &context, PhysicalPlanGenerator &generator) override;
public:
	idx_t table_index;

	DuckTableEntry &table;
	HNSWIndex &hnsw_index;
	idx_t limit;

	vector<column_t> lhs_column_ids;
	vector<idx_t> lhs_projection_ids;
	vector<LogicalType> lhs_returned_types;

	idx_t lhs_vector_column;
	idx_t rhs_vector_column;
};

string LogicalHNSWIndexJoin::GetName() const {
	return "HNSW_INDEX_JOIN";
}

void LogicalHNSWIndexJoin::ResolveTypes() {
	if(lhs_column_ids.empty()) {
		lhs_column_ids.push_back(COLUMN_IDENTIFIER_ROW_ID);
	}
	types.clear();

	if (lhs_projection_ids.empty()) {
		for (const auto &index : lhs_column_ids) {
			if (index == COLUMN_IDENTIFIER_ROW_ID) {
				types.emplace_back(LogicalType::ROW_TYPE);
			} else {
				types.push_back(lhs_returned_types[index]);
			}
		}
	} else {
		for (const auto &proj_index : lhs_projection_ids) {
			const auto &index = lhs_column_ids[proj_index];
			if (index == COLUMN_IDENTIFIER_ROW_ID) {
				types.emplace_back(LogicalType::ROW_TYPE);
			} else {
				types.push_back(lhs_returned_types[index]);
			}
		}
	}

	// Also add the types of the right hand side
	auto &right_types = children[0]->types;
	types.insert(types.end(), right_types.begin(), right_types.end());
}


vector<ColumnBinding> LogicalHNSWIndexJoin::GetLeftBindings() {
	vector<ColumnBinding> result;
	if(lhs_projection_ids.empty()) {
		for(idx_t col_idx = 0; col_idx < lhs_column_ids.size(); col_idx++) {
			result.emplace_back(table_index, col_idx);
		}
	} else {
		for(auto proj_id : lhs_projection_ids) {
			result.emplace_back(table_index, proj_id);
		}
	}
	return result;
}

vector<ColumnBinding> LogicalHNSWIndexJoin::GetRightBindings() {
	vector<ColumnBinding> result;
	for(auto &binding : children[0]->GetColumnBindings()) {
		result.push_back(binding);
	}
	return result;
}

vector<ColumnBinding> LogicalHNSWIndexJoin::GetColumnBindings() {
	vector<ColumnBinding> result;
	auto left_bindings = GetLeftBindings();
	auto right_bindings = GetRightBindings();
	result.insert(result.end(), left_bindings.begin(), left_bindings.end());
	result.insert(result.end(), right_bindings.begin(), right_bindings.end());
	return result;
}

unique_ptr<PhysicalOperator> LogicalHNSWIndexJoin::CreatePlan(ClientContext &context,
															  PhysicalPlanGenerator &generator) {
	auto result = make_uniq<PhysicalHNSWIndexJoin>(types, 0, table, hnsw_index, limit);
	result->limit = limit;
	result->lhs_column_ids = lhs_column_ids;
	result->lhs_projection_ids = lhs_projection_ids;
	result->lhs_vector_column = lhs_vector_column;
	result->rhs_vector_column = rhs_vector_column;

	// Plan the	child
	result->children.push_back(generator.CreatePlan(std::move(children[0])));

	return std::move(result);
}

//------------------------------------------------------------------------------
// Optimizer
//------------------------------------------------------------------------------

class HNSWIndexJoinOptimizer : public OptimizerExtension {
public:
	HNSWIndexJoinOptimizer();
	static bool TryOptimize(Binder &binder, ClientContext &context, unique_ptr<LogicalOperator> &root, unique_ptr<LogicalOperator> &plan);
	static void OptimizeRecursive(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &root, unique_ptr<LogicalOperator> &plan);
	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan);
};

HNSWIndexJoinOptimizer::HNSWIndexJoinOptimizer() {
	optimize_function = Optimize;
}

bool HNSWIndexJoinOptimizer::TryOptimize(Binder &binder, ClientContext &context, unique_ptr<LogicalOperator> &root, unique_ptr<LogicalOperator> &plan) {

	//------------------------------------------------------------------------------
	// Look for:
	// delim_join
	//	-> seq_scan (lhs)
	//  -> projection
	//		-> filter
	//			-> window
	//				-> projection
	//					-> cross_product
	//						-> delim_get
	//						-> seq_scan(rhs)
	//------------------------------------------------------------------------------

	//------------------------------------------------------------------------------
	// Match Operators
	//------------------------------------------------------------------------------
#define MATCH_OPERATOR(OP, TYPE, CHILD_COUNT) if(OP->type != LogicalOperatorType::TYPE || (OP->children.size() != CHILD_COUNT)) { return false; }
	MATCH_OPERATOR(plan, LOGICAL_PROJECTION, 1);
	auto &top_proj = plan->Cast<LogicalProjection>();

	MATCH_OPERATOR(top_proj.children.back(), LOGICAL_DELIM_JOIN, 2);
	auto &delim_join = top_proj.children.back()->Cast<LogicalJoin>();

	MATCH_OPERATOR(delim_join.children[0], LOGICAL_PROJECTION, 1)
	auto &filter_proj = delim_join.children[0]->Cast<LogicalProjection>();

	MATCH_OPERATOR(delim_join.children[1], LOGICAL_GET, 0);
	auto &lhs_get = delim_join.children[1]->Cast<LogicalGet>();
	if(lhs_get.function.name != "seq_scan") {
		return false;
	}

	MATCH_OPERATOR(filter_proj.children[0], LOGICAL_FILTER, 1);
	auto &filter = filter_proj.children[0]->Cast<LogicalFilter>();

	MATCH_OPERATOR(filter.children[0], LOGICAL_WINDOW, 1);
	auto &window = filter.children[0]->Cast<LogicalWindow>();

	MATCH_OPERATOR(window.children[0], LOGICAL_PROJECTION, 1);
	auto &distance_proj = window.children[0]->Cast<LogicalProjection>();

	MATCH_OPERATOR(distance_proj.children[0], LOGICAL_CROSS_PRODUCT, 2);
	auto &cross_product = distance_proj.children[0]->Cast<LogicalCrossProduct>();
#undef MATCH_OPERATOR

	// Extract the delim_get and the rhs_get
	unique_ptr<LogicalOperator>* delim_get_ptr;
	unique_ptr<LogicalOperator>* rhs_get_ptr;

	auto &cp_lhs = cross_product.children[0];
	auto &cp_rhs = cross_product.children[1];
	if(cp_lhs->type == LogicalOperatorType::LOGICAL_DELIM_GET && cp_rhs->type == LogicalOperatorType::LOGICAL_GET) {
		delim_get_ptr = &cp_lhs;;
		rhs_get_ptr = &cp_rhs;
	}
	else if(cp_rhs->type == LogicalOperatorType::LOGICAL_DELIM_GET && cp_lhs->type == LogicalOperatorType::LOGICAL_GET) {
		delim_get_ptr = &cp_rhs;
		rhs_get_ptr = &cp_lhs;
	}
	else {
		return false;
	}

	const auto &delim_get = (*delim_get_ptr)->Cast<LogicalDelimGet>();
	const auto &rhs_get = (*rhs_get_ptr)->Cast<LogicalGet>();
	if(rhs_get.function.name != "seq_scan") {
		return false;
	}

	//------------------------------------------------------------------------------
	// Match Expressions
	//------------------------------------------------------------------------------

	// Verify that the filter is filtering on the window row number
	if(filter.expressions.size() != 1) {
		return false;
	}
	if(filter.expressions.back()->type != ExpressionType::COMPARE_LESSTHANOREQUALTO) {
		return false;
	}
	auto &compare_expr = filter.expressions.back()->Cast<BoundComparisonExpression>();
	if(compare_expr.right->type != ExpressionType::VALUE_CONSTANT) {
		return false;
	}
	auto &constant_expr = compare_expr.right->Cast<BoundConstantExpression>();
	if(constant_expr.return_type != LogicalType::BIGINT) {
		return false;
	}
	auto k_value = constant_expr.value.GetValue<int64_t>();
	if(k_value < 0 || k_value >= STANDARD_VECTOR_SIZE) {
		// Can only optimize up to SVS
		return false;
	}
	if(compare_expr.left->type != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	auto &filter_ref_expr = compare_expr.left->Cast<BoundColumnRefExpression>();
	if(filter_ref_expr.binding.table_index != window.window_index) {
		return false;
	}
	if(filter_ref_expr.binding.column_index != 0) {
		return false;
	}

	// Verify that the window is ordering on a distance function
	if(window.expressions.size() != 1) {
		return false;
	}
	if(window.expressions.back()->type != ExpressionType::WINDOW_ROW_NUMBER) {
		return false;
	}
	auto &window_expr = window.expressions.back()->Cast<BoundWindowExpression>();
	if(window_expr.orders.size() != 1) {
		return false;
	}
	if(window_expr.orders.back().type != OrderType::ASCENDING) {
		return false;
	}
	if(window_expr.orders.back().expression->type != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	auto &distance_ref_expr = window_expr.orders.back().expression->Cast<BoundColumnRefExpression>();


	// Verify that this column ref references the distance expression in the projection
	if(distance_ref_expr.binding.table_index != distance_proj.table_index) {
		return false;
	}
	if(distance_ref_expr.binding.column_index >= distance_proj.expressions.size()) {
		return false;
	}
	auto &distance_expr_ptr = distance_proj.expressions[distance_ref_expr.binding.column_index];

	//------------------------------------------------------------------------------
	// Match the index
	//------------------------------------------------------------------------------
	auto &table = *lhs_get.GetTable();
	if (!table.IsDuckTable()) {
		// We can only replace the scan if the table is a duck table
		return false;
	}
	auto &duck_table = table.Cast<DuckTableEntry>();
	auto &table_info = *table.GetStorage().GetDataTableInfo();

	HNSWIndex *index_ptr = nullptr;
	vector<reference<Expression>> bindings;
	table_info.GetIndexes().BindAndScan<HNSWIndex>(context, table_info, [&](HNSWIndex &hnsw_index) {
		bindings.clear();
		if(!hnsw_index.TryMatchDistanceFunction(distance_expr_ptr, bindings)) {
			return false;
		}
		unique_ptr<Expression> bound_index_expr = nullptr;
		if(!hnsw_index.TryBindIndexExpression(lhs_get, bound_index_expr)) {
			return false;
		}

		// We also have to replace the table index here with the delim_get table index
		ExpressionIterator::EnumerateExpression(bound_index_expr, [&](Expression &child) {
			if(child.type == ExpressionType::BOUND_COLUMN_REF) {
				auto &bound_colref_expr = child.Cast<BoundColumnRefExpression>();
				if(bound_colref_expr.binding.table_index == lhs_get.table_index) {
					bound_colref_expr.binding.table_index = delim_get.table_index;
				}
			}
		});

		auto &lhs_dist_expr = bindings[1];
		auto &rhs_dist_expr = bindings[2];

		// Figure out which of the arguments to the distance function is the index expression (and move it to the lhs)
		// If the index expression is not part of this distance function, we can't optimize, return false.
		if(!lhs_dist_expr.get().Equals(*bound_index_expr)) {
			if(rhs_dist_expr.get().Equals(*bound_index_expr)) {
				std::swap(lhs_dist_expr, rhs_dist_expr);
			} else {
				return false;
			}
		}

		// Save the pointer to the index
		index_ptr = &hnsw_index;
		return true;
	});
	if(!index_ptr) {
		return false;
	}

	// Fuck it, for now dont allow expressions on the index
	if(bindings[1].get().type != ExpressionType::BOUND_COLUMN_REF) { return false; }
	if(bindings[2].get().type != ExpressionType::BOUND_COLUMN_REF) { return false; }
	const auto &lhs_ref_expr = bindings[1].get().Cast<BoundColumnRefExpression>();
	const auto &rhs_ref_expr = bindings[2].get().Cast<BoundColumnRefExpression>();

	if(rhs_ref_expr.binding.table_index != rhs_get.table_index) {
		// Well, we have to reference the rhs
		return false;
	}

	//------------------------------------------------------------------------------
	// Now create the HNSWIndexJoin operator
	//------------------------------------------------------------------------------

	auto index_join = make_uniq<LogicalHNSWIndexJoin>(binder.GenerateTableIndex(), duck_table, *index_ptr, k_value);
	index_join->lhs_column_ids = lhs_get.GetColumnIds();
	index_join->lhs_projection_ids = lhs_get.projection_ids;
	index_join->lhs_returned_types = lhs_get.returned_types;

	// TODO: this is kind of unsafe, column_index != physical index
	index_join->lhs_vector_column = lhs_ref_expr.binding.column_index;
	index_join->rhs_vector_column = rhs_ref_expr.binding.column_index;

	// All the expressions of the filter_proj will be bound column refs.
	// And they will reference columns from the distance_proj.
	ColumnBindingReplacer replacer;

	// Figure out which bindings we are pulling from the right hand side
	for(const auto &binding : filter_proj.GetColumnBindings()) {

		// The binding is a bound column ref to the filter
		const auto &filter_expr = filter_proj.expressions[binding.column_index];
		D_ASSERT(filter_expr->type == ExpressionType::BOUND_COLUMN_REF);
		const auto &filter_colref_expr = filter_expr->Cast<BoundColumnRefExpression>();

		// D_ASSERT(filter_colref_expr.binding.table_index == window.window_index);

		const auto &proj_expr = distance_proj.expressions[filter_colref_expr.binding.column_index];

		if(proj_expr->type != ExpressionType::BOUND_COLUMN_REF) {
			// TODO: Save these and push on top of the cross product later.
			throw NotImplementedException("Only bound column refs are supported in the filter expression");
		}

		const auto &proj_colref_expr = proj_expr->Cast<BoundColumnRefExpression>();
		if(proj_colref_expr.binding.table_index == delim_get.table_index) {
			ColumnBinding new_binding(index_join->table_index, proj_colref_expr.binding.column_index);
			replacer.replacement_bindings.emplace_back(binding, new_binding);
		} else if(proj_colref_expr.binding.table_index == rhs_get.table_index) {
			// Make it reference the RHS of the cross product (our new join) directly
			replacer.replacement_bindings.emplace_back(binding, proj_colref_expr.binding);
		}
	}

	// Figure out what we are pulling from the left hand side
	for(const auto &old_lhs_binding : lhs_get.GetColumnBindings()) {
		ColumnBinding new_binding(index_join->table_index, old_lhs_binding.column_index);
		replacer.replacement_bindings.emplace_back(old_lhs_binding, new_binding);
	}

	// Add the RHS of the cross product to the join
	index_join->children.emplace_back(std::move(*rhs_get_ptr));

	// Swap the plan
	top_proj.children[0] = std::move(index_join);

	// Replace the bindings
	replacer.VisitOperator(*root);

	return true;
}

void HNSWIndexJoinOptimizer::OptimizeRecursive(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &root, unique_ptr<LogicalOperator> &plan) {
	if (!TryOptimize(input.optimizer.binder, input.context, root, plan)) {
		// Recursively optimize the children
		for (auto &child : plan->children) {
			OptimizeRecursive(input, root, child);
		}
	}
}

void HNSWIndexJoinOptimizer::Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	OptimizeRecursive(input, plan, plan);
}

//------------------------------------------------------------------------------
// Register
//------------------------------------------------------------------------------

void HNSWModule::RegisterJoinOptimizer(DatabaseInstance &db) {
	// Register the JoinOptimizer
	db.config.optimizer_extensions.push_back(HNSWIndexJoinOptimizer());
}

} // namespace duckdb