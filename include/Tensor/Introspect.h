#pragma once

// Compile-time introspection of expression TYPES:
//
//   formula<E>()    render E as a formula, e.g. "(in0[i] + (in1[i] * s0))"
//   slots_of<E>()   census of E's leaves: tensor views and scalars
//
// Definitions live in Introspect/, included at the bottom.

#include "Core.h"

#include <cstddef>
#include <string_view>

namespace tensor {

struct ExprSlots {
    size_t views = 0;   // TensorView leaves, numbered in0, in1, … in DFS order
    size_t scalars = 0; // arithmetic leaves, numbered s0, s1, …
};

// Render an expression type as a formula string. Purely compile-time.
template <AnyExpr E> consteval std::string_view formula();

// Count an expression type's tensor and scalar leaves.
template <AnyExpr E> consteval ExprSlots slots_of();

} // namespace tensor

// The definitions.
#include "Introspect/Introspect.h" // IWYU pragma: export
