#pragma once

// The one scatter: a fold whose output cell is chosen by DATA —
//
//   scatter<i>(wrap<C>(cell[i]), 1.0f)                     // counts per cell
//   scatter<i>(clamp<C>(cell[i]), q[i])                    // deposit q
//   scatter<ops::Max, i>(wrap<C>(cell[i]), q[i])           // per-cell maximum
//   scatter<i>(wrap<C>(bx[i]), wrap<C>(by[i]), v[i])       // a 2-D grid
//
// Every argument but the last is a destination and carries a write policy
// naming its extent; the last is the value. Ids are consumed exactly as
// fold consumes them, and any that survive stay axes of the result, after
// the destinations.

#include "../Expr.h"

#include <array>
#include <cstddef>
#include <utility>

namespace tensor {

// clang-format off: annotations predate clang-format's parser
namespace ops {

// Same protocol as ops::Fold — `summed` plus the `op` alias — so every
// whole-tree rule keyed on it (one per tree, no nesting, the epilogue)
// covers a scatter unchanged. `placed` is the only thing that marks it out,
// and walkers detect it by that member, never by the op's name.
template <typename Op, size_t... Ids> struct [[=detail::sym("scatter")]] Scatter {
    using op = Op;
    static constexpr std::array<size_t, sizeof...(Ids)> summed{Ids...};
    static constexpr bool placed = true;
    // Children are [dest…, value]; the node's element type is the VALUE's —
    // through finish∘lift for a structured op. Never executed: the type
    // oracle for Expr's invoke_result_t.
    static constexpr auto operator()(auto... cs) {
        using V = decltype(cs...[sizeof...(cs) - 1]);
        if constexpr (detail::StructuredIndexed<Op, V>)
            return Op::finish(Op::lift(cs...[sizeof...(cs) - 1], size_t{}));
        else if constexpr (detail::Structured<Op, V>)
            return Op::finish(Op::lift(cs...[sizeof...(cs) - 1]));
        else
            return cs...[sizeof...(cs) - 1];
    }
};

} // namespace ops
// clang-format on

template <typename Op, auto... Ids, typename... Cs>
constexpr auto scatter(Cs &&...cs) {
    return detail::scatter_dispatch<Op, Ids...>(std::forward<Cs>(cs)...);
}

template <auto... Ids, typename... Cs> constexpr auto scatter(Cs &&...cs) {
    return detail::scatter_dispatch<ops::Add, Ids...>(std::forward<Cs>(cs)...);
}

} // namespace tensor
