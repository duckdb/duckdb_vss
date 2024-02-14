#pragma once

#include <unordered_map>
#include <cfloat>

#define USEARCH_USE_SIMSIMD 0
#define USEARCH_USE_FP16LIB 1
#define USEARCH_USE_OPENMP  0

#include "usearch/index.hpp"
#include "usearch/index_dense.hpp"
#include "usearch/index_plugins.hpp"