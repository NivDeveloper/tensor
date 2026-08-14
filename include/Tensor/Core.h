#pragma once

// The shared vocabulary — start reading here.

#include "detail/Core.h"

#include <array>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace tensor {

// A multidimensional index, e.g. Index<2>{row, col}.
template <size_t N> using Index = std::array<size_t, N>;

// A plain shaped expression — positional shape rules, no free indices.
template <typename E>
concept TensorExpr = requires {
    typename std::remove_cvref_t<E>::type;
    typename std::remove_cvref_t<E>::extents_type;
    { std::remove_cvref_t<E>::rank } -> std::convertible_to<size_t>;
} && !detail::has_free_index_v<std::remove_cvref_t<E>>;

// An index-bearing expression — a function of its free indices; its shape
// is the free-index space, and it evaluates like any other expression.
template <typename E>
concept IndexedExpr = detail::has_free_index_v<std::remove_cvref_t<E>>;

// Any evaluable expression, plain or index-bearing.
template <typename E>
concept AnyExpr = TensorExpr<E> || IndexedExpr<E>;

// An expression, a scalar (arithmetic / complex) to broadcast, or a
// shapeless generator — which broadcasts like a scalar but varies per cell.
// Anything else fails overload resolution.
template <typename E>
concept Operand =
    TensorExpr<E> || IndexedExpr<E> ||
    detail::is_broadcast_scalar_v<std::remove_cvref_t<E>> ||
    detail::is_shapeless_generator_v<std::remove_cvref_t<E>>;

// At least one child an expression, so plain scalar arithmetic never
// routes through the library. A shapeless generator counts: it is not a
// scalar, and letting the node build is what gets a miscombination the
// diagnostic in Expr rather than "no match for operator*".
template <typename... Cs>
concept Operands =
    (Operand<Cs> && ...) &&
    ((TensorExpr<Cs> || IndexedExpr<Cs> ||
      detail::is_shapeless_generator_v<std::remove_cvref_t<Cs>>) ||
     ...);

} // namespace tensor
