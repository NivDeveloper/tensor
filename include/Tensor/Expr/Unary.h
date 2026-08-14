#pragma once

// The unary operators: - + ~ !. Named functions are map<f> / math::.

#include "../Expr.h"

#include <utility>

namespace tensor {

// clang-format off: annotations predate clang-format's parser
namespace ops {

struct [[=detail::sym("-")]] Neg {
    static constexpr auto operator()(auto a) { return -a; }
    static constexpr bool zero_stable = true;
};
struct [[=detail::sym("+")]] Pos {
    static constexpr auto operator()(auto a) { return +a; }
    static constexpr bool zero_stable = true;
};
struct [[=detail::sym("~")]] BitNot {
    static constexpr auto operator()(auto a) { return ~a; }
};
struct [[=detail::sym("!")]] Not {
    static constexpr auto operator()(auto a) { return !a; }
};

} // namespace ops
// clang-format on

template <typename E>
    requires Operands<E>
constexpr auto operator-(E &&e) {
    return detail::make_expr<ops::Neg>(std::forward<E>(e));
}
template <typename E>
    requires Operands<E>
constexpr auto operator+(E &&e) {
    return detail::make_expr<ops::Pos>(std::forward<E>(e));
}
template <typename E>
    requires Operands<E>
constexpr auto operator~(E &&e) {
    return detail::make_expr<ops::BitNot>(std::forward<E>(e));
}
template <typename E>
    requires Operands<E>
constexpr auto operator!(E &&e) {
    return detail::make_expr<ops::Not>(std::forward<E>(e));
}

} // namespace tensor
