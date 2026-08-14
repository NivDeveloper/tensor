#pragma once

// The one fold: consume listed indices with a reducible op —
//
//   fold<j>(A[i,j] * x[j])            // contraction (Add is the default)
//   fold<ops::Max, j>(A[i,j] + c[j])  // op-generic indexed fold
//   fold<ops::Max>(E * K * f)         // whole-expression fold → the scalar
//   fold<ops::Add, 0, 2>(t)           // axis numbers with a PLAIN operand
//
// Placeholders take an index-bearing operand, axis numbers a plain one
// (subscripted internally, so there is ONE node protocol); the op declares
// identity<T>() and the fold order is unspecified.

#include "../Expr.h"

#include <cstddef>
#include <limits>
#include <utility>

namespace tensor {

// clang-format off: annotations predate clang-format's parser
namespace ops {

// Folds only; the elementwise pair is math::Fmax/Fmin (C's NaN rule,
// where this op is a plain a < b).
struct [[=detail::sym("max")]] Max {
    static constexpr auto operator()(auto a, auto b) { return a < b ? b : a; }
    template <typename T> static constexpr T identity() {
        return std::numeric_limits<T>::lowest();
    }
};
struct [[=detail::sym("min")]] Min {
    static constexpr auto operator()(auto a, auto b) { return b < a ? b : a; }
    template <typename T> static constexpr T identity() {
        return std::numeric_limits<T>::max();
    }
};

// The folded ids are named `summed` after the Add default; walkers detect
// the protocol by this member, never by the op's name.
template <typename Op, size_t... Summed> struct [[=detail::sym("fold")]] Fold {
    using op = Op;
    static constexpr std::array<size_t, sizeof...(Summed)> summed{Summed...};
    static constexpr auto operator()(auto a) { return a; }
};

} // namespace ops
// clang-format on

template <typename Op, auto... Ids, AnyExpr S> constexpr auto fold(S &&s) {
    return detail::fold_dispatch<Op, Ids...>(std::forward<S>(s));
}

template <auto... Ids, AnyExpr S> constexpr auto fold(S &&s) {
    return detail::fold_dispatch<ops::Add, Ids...>(std::forward<S>(s));
}

} // namespace tensor
