#pragma once

// The stats layer's plumbing: the field projections a structured fold's
// result passes through, and the bin arithmetic Histogram shares with its
// edges. Named objects, not lambdas — map<f> takes an entity, and a
// closure's op_type fails silently.

#include "../Math.h"
#include "Expr.h"

#include <cstddef>

namespace tensor::detail {

// Welford's result carries both moments; a caller who asked only for the
// variance gets it through an ordinary elementwise node above the fold
// (the epilogue), so Var stays one pass and stays an expression.
struct VarField {
    constexpr auto operator()(auto s) const { return s.var; }
};
inline constexpr VarField var_field{};

// A histogram's range: numpy's rule for the degenerate case, so a constant
// tensor still produces a grid rather than a division by zero.
template <typename T> constexpr void widen_degenerate(T &lo, T &hi) {
    if (!(lo < hi)) {
        lo -= T(0.5);
        hi += T(0.5);
    }
}

} // namespace tensor::detail
