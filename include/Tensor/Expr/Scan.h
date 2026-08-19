#pragma once

// The one scan: a running op along ONE index, which it KEEPS —
//
//   scan<m>(h[j, m])              // the running sum (Add is the default)
//   scan<ops::Max, m>(x[i, m])    // the running maximum
//   scan<ops::Add, 1>(t)          // an axis number, with a PLAIN operand
//
// `fold` consumes an index and `scatter` places one; a scan preserves it, so
// the result has the operand's shape and its last entry along the axis is
// what `fold` over that axis would have returned. The op declares
// identity<T>(), as a fold's does.
//
// Evaluation order is SPECIFIED, unlike a fold's: the prefix is taken in
// ascending order with one accumulator per row, so serial and threaded
// results agree bit for bit. Exclusive scan needs no builder — read the
// materialized result one place back, where pad supplies the identity:
//
//   auto s = eval(scan<ops::Add, m>(x));   s[j, pad(m - 1_c, 0.0f)]

#include "../Expr.h"

#include <cstddef>
#include <utility>

namespace tensor {

// clang-format off: annotations predate clang-format's parser
namespace ops {

// The scanned id is named `scanned`, and walkers detect the protocol by that
// member rather than by the op's name. There is deliberately no `summed`: a
// scan consumes nothing, which is exactly what makes node_extents give it
// its child's shape and free_ids_v pass its child's indices through.
template <typename Op, size_t Id> struct [[=detail::sym("scan")]] Scan {
    using op = Op;
    static constexpr size_t scanned = Id;
    static constexpr auto operator()(auto a) { return a; }
};

} // namespace ops
// clang-format on

template <typename Op, auto Id, AnyExpr S> constexpr auto scan(S &&s) {
    return detail::scan_dispatch<Op, Id>(std::forward<S>(s));
}

template <auto Id, AnyExpr S> constexpr auto scan(S &&s) {
    return detail::scan_dispatch<ops::Add, Id>(std::forward<S>(s));
}

template <typename Op, auto Id, detail::RangedExpr S>
constexpr auto scan(S &&s) {
    return scan<Op, Id>(std::forward<S>(s).c);
}

template <auto Id, detail::RangedExpr S> constexpr auto scan(S &&s) {
    return scan<Id>(std::forward<S>(s).c);
}

} // namespace tensor
