#pragma once

// The binary operators — valarray's set; named functions go through map<f>.
// auto-generic, so validity follows the element types; comparisons produce
// bool elements, no short-circuit. Declaring identity<T>() makes an op a
// usable fold; declaring absorber<T>() asserts the value annihilates the
// op from either side; declaring zero_stable asserts op(0, 0) is 0, which
// lets a guard shared by both operands rise past the op.

#include "../Expr.h"

#include <utility>

namespace tensor {

// clang-format off: annotations predate clang-format's parser
namespace ops {

struct [[=detail::sym("+"), =detail::atomic("InterlockedAdd")]] Add {
    static constexpr auto operator()(auto a, auto b) { return a + b; }
    template <typename T> static constexpr T identity() { return T{}; }
    static constexpr bool zero_stable = true;
};
struct [[=detail::sym("-")]] Sub {
    static constexpr auto operator()(auto a, auto b) { return a - b; }
    static constexpr bool zero_stable = true;
};
struct [[=detail::sym("*")]] Mul {
    static constexpr auto operator()(auto a, auto b) { return a * b; }
    template <typename T> static constexpr T identity() { return T{1}; }
    template <typename T> static constexpr T absorber() { return T{}; }
};
struct [[=detail::sym("/")]] Div {
    static constexpr auto operator()(auto a, auto b) { return a / b; }
};
struct [[=detail::sym("%")]] Mod {
    static constexpr auto operator()(auto a, auto b) { return a % b; }
};
struct [[=detail::sym("&"), =detail::atomic("InterlockedAnd")]] BitAnd {
    static constexpr auto operator()(auto a, auto b) { return a & b; }
    template <typename T> static constexpr T identity() { return T(~T{}); }
    template <typename T> static constexpr T absorber() { return T{}; }
};
struct [[=detail::sym("|"), =detail::atomic("InterlockedOr")]] BitOr {
    static constexpr auto operator()(auto a, auto b) { return a | b; }
    template <typename T> static constexpr T identity() { return T{}; }
};
struct [[=detail::sym("^"), =detail::atomic("InterlockedXor")]] BitXor {
    static constexpr auto operator()(auto a, auto b) { return a ^ b; }
    template <typename T> static constexpr T identity() { return T{}; }
};
struct [[=detail::sym("<<")]] Shl {
    static constexpr auto operator()(auto a, auto b) { return a << b; }
};
struct [[=detail::sym(">>")]] Shr {
    static constexpr auto operator()(auto a, auto b) { return a >> b; }
};
struct [[=detail::sym("==")]] Eq {
    static constexpr auto operator()(auto a, auto b) { return a == b; }
};
struct [[=detail::sym("!=")]] Ne {
    static constexpr auto operator()(auto a, auto b) { return a != b; }
};
struct [[=detail::sym("<")]] Lt {
    static constexpr auto operator()(auto a, auto b) { return a < b; }
};
struct [[=detail::sym(">")]] Gt {
    static constexpr auto operator()(auto a, auto b) { return a > b; }
};
struct [[=detail::sym("<=")]] Le {
    static constexpr auto operator()(auto a, auto b) { return a <= b; }
};
struct [[=detail::sym(">=")]] Ge {
    static constexpr auto operator()(auto a, auto b) { return a >= b; }
};
struct [[=detail::sym("&&")]] And {
    static constexpr auto operator()(auto a, auto b) { return a && b; }
    template <typename T> static constexpr T identity() { return T{true}; }
    template <typename T> static constexpr T absorber() { return T{}; }
};
struct [[=detail::sym("||")]] Or {
    static constexpr auto operator()(auto a, auto b) { return a || b; }
    template <typename T> static constexpr T identity() { return T{}; }
};

} // namespace ops
// clang-format on

template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator+(L &&l, R &&r) {
    return detail::make_expr<ops::Add>(std::forward<L>(l), std::forward<R>(r));
}
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator-(L &&l, R &&r) {
    return detail::make_expr<ops::Sub>(std::forward<L>(l), std::forward<R>(r));
}
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator*(L &&l, R &&r) {
    return detail::make_expr<ops::Mul>(std::forward<L>(l), std::forward<R>(r));
}
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator/(L &&l, R &&r) {
    return detail::make_expr<ops::Div>(std::forward<L>(l), std::forward<R>(r));
}
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator%(L &&l, R &&r) {
    return detail::make_expr<ops::Mod>(std::forward<L>(l), std::forward<R>(r));
}
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator&(L &&l, R &&r) {
    return detail::make_expr<ops::BitAnd>(std::forward<L>(l),
                                          std::forward<R>(r));
}
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator|(L &&l, R &&r) {
    return detail::make_expr<ops::BitOr>(std::forward<L>(l),
                                         std::forward<R>(r));
}
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator^(L &&l, R &&r) {
    return detail::make_expr<ops::BitXor>(std::forward<L>(l),
                                          std::forward<R>(r));
}
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator<<(L &&l, R &&r) {
    return detail::make_expr<ops::Shl>(std::forward<L>(l), std::forward<R>(r));
}
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator>>(L &&l, R &&r) {
    return detail::make_expr<ops::Shr>(std::forward<L>(l), std::forward<R>(r));
}
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator==(L &&l, R &&r) {
    return detail::make_expr<ops::Eq>(std::forward<L>(l), std::forward<R>(r));
}
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator!=(L &&l, R &&r) {
    return detail::make_expr<ops::Ne>(std::forward<L>(l), std::forward<R>(r));
}
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator<(L &&l, R &&r) {
    return detail::make_expr<ops::Lt>(std::forward<L>(l), std::forward<R>(r));
}
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator>(L &&l, R &&r) {
    return detail::make_expr<ops::Gt>(std::forward<L>(l), std::forward<R>(r));
}
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator<=(L &&l, R &&r) {
    return detail::make_expr<ops::Le>(std::forward<L>(l), std::forward<R>(r));
}
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator>=(L &&l, R &&r) {
    return detail::make_expr<ops::Ge>(std::forward<L>(l), std::forward<R>(r));
}
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator&&(L &&l, R &&r) {
    return detail::make_expr<ops::And>(std::forward<L>(l), std::forward<R>(r));
}
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator||(L &&l, R &&r) {
    return detail::make_expr<ops::Or>(std::forward<L>(l), std::forward<R>(r));
}

} // namespace tensor
