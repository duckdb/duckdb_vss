#include "duckdb/catalog/catalog_entry/aggregate_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
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
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/transaction/duck_transaction.hpp"
#include "duckdb/storage/storage_index.hpp"

#include "hnsw/hnsw.hpp"
#include "hnsw/hnsw_index.hpp"

namespace duckdb {

//------------------------------------------------------------------------------
// Physical Operator
//------------------------------------------------------------------------------

class PhysicalHNSWIndexJoin final : public PhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

	PhysicalHNSWIndexJoin(const vector<LogicalType> &types_p, const idx_t estimated_cardinality,
	                      DuckTableEntry &table_p, HNSWIndex &hnsw_index_p, const idx_t limit_p)
	    : PhysicalOperator(TYPE, types_p, estimated_cardinality), table(table_p), hnsw_index(hnsw_index_p),
	      limit(limit_p) {
	}

public:
	string GetName() const override;
	bool ParallelOperator() const override;
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override;
	OperatorResultType Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
	                           GlobalOperatorState &gstate, OperatorState &state) const override;
	InsertionOrderPreservingMap<string> ParamsToString() const override;

public:
	DuckTableEntry &table;
	HNSWIndex &hnsw_index;
	idx_t limit;

	vector<column_t> inner_column_ids;
	vector<idx_t> inner_projection_ids;

	idx_t outer_vector_column;
	idx_t inner_vector_column;
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
	vector<StorageIndex> physical_column_ids;

	// Index scan state
	unique_ptr<IndexScanState> index_state;
	SelectionVector match_sel;
};

unique_ptr<OperatorState> PhysicalHNSWIndexJoin::GetOperatorState(ExecutionContext &context) const {
	auto result = make_uniq<HNSWIndexJoinState>();

	auto &local_storage = LocalStorage::Get(context.client, table.catalog);
	result->physical_column_ids.reserve(inner_column_ids.size());

	// Figure out the storage column ids from the projection expression
	for (auto &id : inner_column_ids) {
		storage_t col_id = id;
		if (id != DConstants::INVALID_INDEX) {
			col_id = table.GetColumn(LogicalIndex(id)).StorageOid();
		}
		result->physical_column_ids.emplace_back(col_id);
	}

	// Initialize selection vector
	result->match_sel.Initialize();

	// Initialize the storage scan state
	result->local_storage_state.Initialize(result->physical_column_ids, nullptr);
	local_storage.InitializeScan(table.GetStorage(), result->local_storage_state.local_state, nullptr);

	// Initialize the index scan state
	result->index_state = hnsw_index.InitializeMultiScan(context.client);

	return std::move(result);
}

OperatorResultType PhysicalHNSWIndexJoin::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                  GlobalOperatorState &gstate, OperatorState &ostate) const {
	auto &state = ostate.Cast<HNSWIndexJoinState>();
	auto &transcation = DuckTransaction::Get(context.client, table.catalog);

	input.Flatten();

	// The first 0..inner_column_ids.size() columns are the inner table columns
	const auto MATCH_COLUMN_OFFSET = inner_column_ids.size();
	// The next column is the row number
	const auto OUTER_COLUMN_OFFSET = MATCH_COLUMN_OFFSET + 1;
	// The rest of the columns are the outer table columns

	auto &rhs_vector_vector = input.data[outer_vector_column];
	auto &rhs_vector_child = ArrayVector::GetEntry(rhs_vector_vector);
	const auto rhs_vector_size = ArrayType::GetSize(rhs_vector_vector.GetType());
	const auto rhs_vector_ptr = FlatVector::GetData<float>(rhs_vector_child);

	// We mimic the window row_number() operator here and output the row number in each batch, basically.
	const auto row_number_vector = FlatVector::GetData<idx_t>(chunk.data[MATCH_COLUMN_OFFSET]);

	hnsw_index.ResetMultiScan(*state.index_state);

	// How many batches are we going to process?
	const auto batch_count = MinValue(input.size() - state.input_idx, STANDARD_VECTOR_SIZE / limit);
	idx_t output_idx = 0;
	for (idx_t batch_idx = 0; batch_idx < batch_count; batch_idx++, state.input_idx++) {

		// Get the next batch
		const auto rhs_vector_data = rhs_vector_ptr + state.input_idx * rhs_vector_size;

		// Scan the index for row ids
		const auto match_count = hnsw_index.ExecuteMultiScan(*state.index_state, rhs_vector_data, limit);
		for (idx_t i = 0; i < match_count; i++) {
			state.match_sel.set_index(output_idx, state.input_idx);
			row_number_vector[output_idx] = i + 1; // Note: 1-indexed!
			output_idx++;
		}
	}

	const auto &row_ids = hnsw_index.GetMultiScanResult(*state.index_state);

	// Execute one big fetch for the LHS
	table.GetStorage().Fetch(transcation, chunk, state.physical_column_ids, row_ids, output_idx, state.fetch_state);

	// Now slice the chunk so that we include the rhs too
	chunk.Slice(input, state.match_sel, output_idx, OUTER_COLUMN_OFFSET);

	// Set the cardinality
	chunk.SetCardinality(output_idx);

	if (state.input_idx == input.size()) {
		state.input_idx = 0;
		return OperatorResultType::NEED_MORE_INPUT;
	}

	return OperatorResultType::HAVE_MORE_OUTPUT;
}

InsertionOrderPreservingMap<string> PhysicalHNSWIndexJoin::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	auto table_name = table.name;
	auto index_name = hnsw_index.name;
	result.insert("table", table_name);
	result.insert("index", index_name);
	result.insert("limit", to_string(limit));
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

//------------------------------------------------------------------------------
// Logical Operator
//------------------------------------------------------------------------------

class LogicalHNSWIndexJoin final : public LogicalExtensionOperator {
public:
	explicit LogicalHNSWIndexJoin(const idx_t table_index_p, DuckTableEntry &table_p, HNSWIndex &hnsw_index_p,
	                              const idx_t limit_p)
	    : table_index(table_index_p), table(table_p), hnsw_index(hnsw_index_p), limit(limit_p) {
	}

public:
	string GetName() const override;
	void ResolveTypes() override;
	vector<ColumnBinding> GetColumnBindings() override;
	vector<ColumnBinding> GetLeftBindings();
	vector<ColumnBinding> GetRightBindings();
	unique_ptr<PhysicalOperator> CreatePlan(ClientContext &context, PhysicalPlanGenerator &generator) override;
	idx_t EstimateCardinality(ClientContext &context) override;

public:
	idx_t table_index;

	DuckTableEntry &table;
	HNSWIndex &hnsw_index;
	idx_t limit;

	vector<column_t> inner_column_ids;
	vector<idx_t> inner_projection_ids;
	vector<LogicalType> inner_returned_types;

	idx_t outer_vector_column;
	idx_t inner_vector_column;
};

string LogicalHNSWIndexJoin::GetName() const {
	return "HNSW_INDEX_JOIN";
}

void LogicalHNSWIndexJoin::ResolveTypes() {
	if (inner_column_ids.empty()) {
		inner_column_ids.push_back(COLUMN_IDENTIFIER_ROW_ID);
	}
	types.clear();

	if (inner_projection_ids.empty()) {
		for (const auto &index : inner_column_ids) {
			if (index == COLUMN_IDENTIFIER_ROW_ID) {
				types.emplace_back(LogicalType::ROW_TYPE);
			} else {
				types.push_back(inner_returned_types[index]);
			}
		}
	} else {
		for (const auto &proj_index : inner_projection_ids) {
			const auto &index = inner_column_ids[proj_index];
			if (index == COLUMN_IDENTIFIER_ROW_ID) {
				types.emplace_back(LogicalType::ROW_TYPE);
			} else {
				types.push_back(inner_returned_types[index]);
			}
		}
	}

	// Always add the row_number after the inner columns
	types.emplace_back(LogicalType::BIGINT);

	// Also add the types of the right hand side
	auto &right_types = children[0]->types;
	types.insert(types.end(), right_types.begin(), right_types.end());
}

vector<ColumnBinding> LogicalHNSWIndexJoin::GetLeftBindings() {
	vector<ColumnBinding> result;
	if (inner_projection_ids.empty()) {
		for (idx_t col_idx = 0; col_idx < inner_column_ids.size(); col_idx++) {
			result.emplace_back(table_index, col_idx);
		}
	} else {
		for (auto proj_id : inner_projection_ids) {
			result.emplace_back(table_index, proj_id);
		}
	}

	// Always add the row number last
	result.emplace_back(table_index, inner_column_ids.size());

	return result;
}

vector<ColumnBinding> LogicalHNSWIndexJoin::GetRightBindings() {
	vector<ColumnBinding> result;
	for (auto &binding : children[0]->GetColumnBindings()) {
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

	auto result = make_uniq<PhysicalHNSWIndexJoin>(types, estimated_cardinality, table, hnsw_index, limit);
	result->limit = limit;
	result->inner_column_ids = inner_column_ids;
	result->inner_projection_ids = inner_projection_ids;
	result->outer_vector_column = outer_vector_column;
	result->inner_vector_column = inner_vector_column;

	// Plan the	child
	result->children.push_back(generator.CreatePlan(std::move(children[0])));

	return std::move(result);
}

idx_t LogicalHNSWIndexJoin::EstimateCardinality(ClientContext &context) {
	// The cardinality of the HNSW index join is the cardinality of the outer table
	if (has_estimated_cardinality) {
		return estimated_cardinality;
	}

	const auto child_cardinality = children[0]->EstimateCardinality(context);
	estimated_cardinality = child_cardinality * limit;
	has_estimated_cardinality = true;

	return estimated_cardinality;
}

//------------------------------------------------------------------------------
// Optimizer
//------------------------------------------------------------------------------

class HNSWIndexJoinOptimizer : public OptimizerExtension {
public:
	HNSWIndexJoinOptimizer();
	static bool TryOptimize(Binder &binder, ClientContext &context, unique_ptr<LogicalOperator> &root,
	                        unique_ptr<LogicalOperator> &plan);
	static void OptimizeRecursive(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &root,
	                              unique_ptr<LogicalOperator> &plan);
	static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan);
};

HNSWIndexJoinOptimizer::HNSWIndexJoinOptimizer() {
	optimize_function = Optimize;
}

class CardinalityResetter final : public LogicalOperatorVisitor {
public:
	ClientContext &context;

	explicit CardinalityResetter(ClientContext &context_p) : context(context_p) {
	}

	void VisitOperator(LogicalOperator &op) override {
		op.has_estimated_cardinality = false;
		VisitOperatorChildren(op);
		op.EstimateCardinality(context);
	}
};

bool HNSWIndexJoinOptimizer::TryOptimize(Binder &binder, ClientContext &context, unique_ptr<LogicalOperator> &root,
                                         unique_ptr<LogicalOperator> &plan) {

	//------------------------------------------------------------------------------
	// Look for:
	// delim_join
	//	-> seq_scan (lhs)
	//  -> projection "outer" (optional, only there if the distance expression is unused)
	//		-> filter
	//			-> window
	//				-> projection "inner"
	//					-> cross_product
	//						-> delim_get
	//						-> seq_scan(rhs)
	//------------------------------------------------------------------------------

	//------------------------------------------------------------------------------
	// Match Operators
	//------------------------------------------------------------------------------
#define MATCH_OPERATOR(OP, TYPE, CHILD_COUNT)                                                                          \
	if (OP->type != LogicalOperatorType::TYPE || (OP->children.size() != CHILD_COUNT)) {                               \
		return false;                                                                                                  \
	}

	MATCH_OPERATOR(plan, LOGICAL_DELIM_JOIN, 2);
	auto &delim_join = plan->Cast<LogicalJoin>();

	// branch
	MATCH_OPERATOR(delim_join.children[1], LOGICAL_GET, 0);
	auto outer_get_ptr = &delim_join.children[1];
	auto &outer_get = (*outer_get_ptr)->Cast<LogicalGet>();

	// branch
	// There might not be a projection here if we keep the distance function.

	const unique_ptr<LogicalOperator> *filter_ptr = nullptr;
	const unique_ptr<LogicalOperator> *outer_proj_ptr = nullptr;

	auto &delim_child = delim_join.children[0];
	if (delim_child->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		// The distance expressions is projected out.
		auto &filter_proj = delim_child->Cast<LogicalProjection>();
		if (filter_proj.children.back()->type != LogicalOperatorType::LOGICAL_FILTER) {
			return false;
		}
		outer_proj_ptr = &delim_child;
		filter_ptr = &filter_proj.children.back();
	} else if (delim_child->type == LogicalOperatorType::LOGICAL_FILTER) {
		// The distance function is kept.
		filter_ptr = &delim_child;
	} else {
		return false;
	}

	auto &filter = (*filter_ptr)->Cast<LogicalFilter>();

	MATCH_OPERATOR(filter.children.back(), LOGICAL_WINDOW, 1);
	auto &window = filter.children.back()->Cast<LogicalWindow>();

	MATCH_OPERATOR(window.children[0], LOGICAL_PROJECTION, 1);
	auto &inner_proj = window.children[0]->Cast<LogicalProjection>();

	MATCH_OPERATOR(inner_proj.children[0], LOGICAL_CROSS_PRODUCT, 2);
	auto &cross_product = inner_proj.children[0]->Cast<LogicalCrossProduct>();
#undef MATCH_OPERATOR

	// Extract the delim_get and the rhs_get
	unique_ptr<LogicalOperator> *delim_get_ptr;
	unique_ptr<LogicalOperator> *inner_get_ptr;

	auto &cp_lhs = cross_product.children[0];
	auto &cp_rhs = cross_product.children[1];
	if (cp_lhs->type == LogicalOperatorType::LOGICAL_DELIM_GET && cp_rhs->type == LogicalOperatorType::LOGICAL_GET) {
		delim_get_ptr = &cp_lhs;
		inner_get_ptr = &cp_rhs;
	} else if (cp_rhs->type == LogicalOperatorType::LOGICAL_DELIM_GET &&
	           cp_lhs->type == LogicalOperatorType::LOGICAL_GET) {
		delim_get_ptr = &cp_rhs;
		inner_get_ptr = &cp_lhs;
	} else {
		return false;
	}

	auto &delim_get = (*delim_get_ptr)->Cast<LogicalDelimGet>();
	auto &inner_get = (*inner_get_ptr)->Cast<LogicalGet>();
	if (inner_get.function.name != "seq_scan") {
		return false;
	}

	//------------------------------------------------------------------------------
	// Match Expressions
	//------------------------------------------------------------------------------

	// Verify that the filter is filtering on the window row number
	if (filter.expressions.size() != 1) {
		return false;
	}
	if (filter.expressions.back()->type != ExpressionType::COMPARE_LESSTHANOREQUALTO) {
		return false;
	}
	const auto &compare_expr = filter.expressions.back()->Cast<BoundComparisonExpression>();
	if (compare_expr.right->type != ExpressionType::VALUE_CONSTANT) {
		return false;
	}
	const auto &constant_expr = compare_expr.right->Cast<BoundConstantExpression>();
	if (constant_expr.return_type != LogicalType::BIGINT) {
		return false;
	}
	auto k_value = constant_expr.value.GetValue<int64_t>();
	if (k_value < 0 || k_value >= STANDARD_VECTOR_SIZE) {
		// Can only optimize up to SVS
		return false;
	}
	if (compare_expr.left->type != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	auto &filter_ref_expr = compare_expr.left->Cast<BoundColumnRefExpression>();
	if (filter_ref_expr.binding.table_index != window.window_index) {
		return false;
	}
	if (filter_ref_expr.binding.column_index != 0) {
		return false;
	}

	// Verify that the window is ordering on a distance function
	if (window.expressions.size() != 1) {
		return false;
	}
	if (window.expressions.back()->type != ExpressionType::WINDOW_ROW_NUMBER) {
		return false;
	}
	auto &window_expr = window.expressions.back()->Cast<BoundWindowExpression>();
	if (window_expr.orders.size() != 1) {
		return false;
	}
	if (window_expr.orders.back().type != OrderType::ASCENDING) {
		return false;
	}
	if (window_expr.orders.back().expression->type != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	const auto &distance_ref_expr = window_expr.orders.back().expression->Cast<BoundColumnRefExpression>();
	// Verify that this column ref references the distance expression in the projection
	if (distance_ref_expr.binding.table_index != inner_proj.table_index) {
		return false;
	}
	if (distance_ref_expr.binding.column_index >= inner_proj.expressions.size()) {
		return false;
	}
	const auto &distance_expr_ptr = inner_proj.expressions[distance_ref_expr.binding.column_index];

	//------------------------------------------------------------------------------
	// Match the index
	//------------------------------------------------------------------------------
	auto &table = *inner_get.GetTable();
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
		if (!hnsw_index.TryMatchDistanceFunction(distance_expr_ptr, bindings)) {
			return false;
		}
		unique_ptr<Expression> bound_index_expr = nullptr;
		if (!hnsw_index.TryBindIndexExpression(inner_get, bound_index_expr)) {
			return false;
		}

		// We also have to replace the outer table index here with the delim_get table index
		ExpressionIterator::EnumerateExpression(bound_index_expr, [&](Expression &child) {
			if (child.type == ExpressionType::BOUND_COLUMN_REF) {
				auto &bound_colref_expr = child.Cast<BoundColumnRefExpression>();
				if (bound_colref_expr.binding.table_index == outer_get.table_index) {
					bound_colref_expr.binding.table_index = delim_get.table_index;
				}
			}
		});

		auto &lhs_dist_expr = bindings[1];
		auto &rhs_dist_expr = bindings[2];

		// Figure out which of the arguments to the distance function is the index expression (and move it to the rhs)
		// If the index expression is not part of this distance function, we can't optimize, return false.
		if (lhs_dist_expr.get().Equals(*bound_index_expr)) {
			if (!rhs_dist_expr.get().Equals(*bound_index_expr)) {
				std::swap(lhs_dist_expr, rhs_dist_expr);
			} else {
				return false;
			}
		}

		// Save the pointer to the index
		index_ptr = &hnsw_index;
		return true;
	});
	if (!index_ptr) {
		return false;
	}

	// Fuck it, for now dont allow expressions on the index
	if (bindings[1].get().type != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	if (bindings[2].get().type != ExpressionType::BOUND_COLUMN_REF) {
		return false;
	}
	const auto &outer_ref_expr = bindings[1].get().Cast<BoundColumnRefExpression>();
	const auto &inner_ref_expr = bindings[2].get().Cast<BoundColumnRefExpression>();

	// Sanity check
	if (inner_ref_expr.binding.table_index != inner_get.table_index) {
		// Well, we have to reference the rhs
		return false;
	}

	//------------------------------------------------------------------------------
	// Now create the HNSWIndexJoin operator
	//------------------------------------------------------------------------------

	auto index_join = make_uniq<LogicalHNSWIndexJoin>(binder.GenerateTableIndex(), duck_table, *index_ptr, k_value);
	for(auto &column_id : inner_get.GetColumnIds()) {
		index_join->inner_column_ids.emplace_back(column_id.GetPrimaryIndex());
	}
	index_join->inner_projection_ids = inner_get.projection_ids;
	index_join->inner_returned_types = inner_get.returned_types;

	// TODO: this is kind of unsafe, column_index != physical index
	index_join->outer_vector_column = outer_ref_expr.binding.column_index;
	index_join->inner_vector_column = inner_ref_expr.binding.column_index;

	ColumnBindingReplacer replacer;

	// Start by creating an new projection from the delim join
	// This projection will be the top projection of the new plan
	vector<unique_ptr<Expression>> projection_expressions;
	auto projection_table_index = binder.GenerateTableIndex();
	auto delim_bindings = delim_join.GetColumnBindings();
	auto delim_types = delim_join.types;

	idx_t new_binding_idx = 0;
	for (idx_t i = 0; i < delim_bindings.size(); i++) {
		auto &old_binding = delim_bindings[i];

		if (old_binding.table_index == window.window_index) {
			// The window expression is never used past the filter. We can just skip it.
			// I think...?
			continue;
		}

		auto &old_type = delim_types[i];
		projection_expressions.push_back(make_uniq<BoundColumnRefExpression>(old_type, old_binding));
		replacer.replacement_bindings.emplace_back(old_binding,
		                                           ColumnBinding(projection_table_index, new_binding_idx++));
	}

	// Also add the window expression to the projection last. We will replace this with a reference to the index join
	// in the next inlining step
	ColumnBinding window_binding(window.window_index, 0);
	projection_expressions.push_back(make_uniq<BoundColumnRefExpression>(LogicalType::BIGINT, window_binding));
	replacer.replacement_bindings.emplace_back(window_binding,
	                                           ColumnBinding(projection_table_index, new_binding_idx++));

	auto new_projection = make_uniq<LogicalProjection>(projection_table_index, std::move(projection_expressions));

	// Replace all previous references with our new projection
	replacer.VisitOperator(*root);
	replacer.replacement_bindings.clear();

	// Start inlining our expressions
	for (auto &expr : new_projection->expressions) {
		auto &ref = expr->Cast<BoundColumnRefExpression>();

		// If this references the inner table scan, reference the index join instead
		if (ref.binding.table_index == inner_get.table_index) {
			ref.binding.table_index = index_join->table_index;
		}

		// If this references the outer proj, replace it with the inner proj
		if (outer_proj_ptr) {
			auto &outer_proj = outer_proj_ptr->get()->Cast<LogicalProjection>();
			if (ref.binding.table_index == outer_proj.table_index) {
				// assert that this can only be bound column ref
				const auto &outer_expr = outer_proj.expressions[ref.binding.column_index];
				const auto &outer_ref = outer_expr->Cast<BoundColumnRefExpression>();

				ref.binding = outer_ref.binding;
			}
		}

		// If this references the inner proj, replace it with the actual expression
		if (ref.binding.table_index == inner_proj.table_index) {
			const auto &inner_expr = inner_proj.expressions[ref.binding.column_index];
			expr = inner_expr->Copy();
			// These can still reference the delim_get, but we replace them in the next step.
		}

		// Special case: the window row number expression. Forward this to the index join
		else if (ref.binding.table_index == window.window_index) {
			// The special "row_number" expression is always the last column of the index_join itself
			ColumnBinding index_row_number_binding(index_join->table_index, index_join->inner_column_ids.size());
			expr = make_uniq<BoundColumnRefExpression>(LogicalType::BIGINT, index_row_number_binding);
		}
	}

	// Everything that used to reference the delim get now just reference the outer get
	for (const auto &old_binding : delim_get.GetColumnBindings()) {
		auto new_binding = ColumnBinding(outer_get.table_index, old_binding.column_index);
		replacer.replacement_bindings.emplace_back(old_binding, new_binding);
	}

	// Everything that used to reference the inner get now just reference the index join
	for (const auto &old_binding : inner_get.GetColumnBindings()) {
		auto new_binding = ColumnBinding(index_join->table_index, old_binding.column_index);
		replacer.replacement_bindings.emplace_back(old_binding, new_binding);
	}

	replacer.VisitOperator(*new_projection);

	// We are done!

	// Add the outer get (LHS) of the cross product to the join
	index_join->children.emplace_back(std::move(*outer_get_ptr));

	// Add the new projection on top of the join
	new_projection->children.emplace_back(std::move(index_join));
	new_projection->EstimateCardinality(context);

	// Swap the plan
	plan = std::move(new_projection);

	CardinalityResetter cardinality_resetter(context);
	cardinality_resetter.VisitOperator(*root);

	return true;
}

void HNSWIndexJoinOptimizer::OptimizeRecursive(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &root,
                                               unique_ptr<LogicalOperator> &plan) {
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