#pragma once

// The annotation vocabulary: an op carries its spelling as an annotation —
//
//   struct [[=detail::sym("+")]] Add { … };
//
// — and everything queries it; no parallel symbol tables. A forgotten
// annotation fails the build by name (the lint in detail/Ops.h).

#include <meta>
#include <string_view>

namespace tensor::detail {

struct Symbol {
    const char *str;
};

// A pointer into a string literal is not a permitted annotation constant.
consteval Symbol sym(std::string_view s) {
    return {std::define_static_string(s)};
}

// Marks a reduction op that returns one of its operands UNCHANGED, so a
// wider accumulator cannot change its answer — and would only cost, since
// the narrow compare vectorizes where the widened one does not.
struct Selective {};

// Marks an op with no Slang intrinsic; eval(dev, …) rejects it by name.
struct CpuOnly {};

// A reducible op's Slang atomic, for the scatter kernel that deposits into
// the output. Consulted only for integral element types — see
// scatter_is_atomic, which owns that rule.
struct Atomic {
    const char *str;
};

consteval Atomic atomic(std::string_view s) {
    return {std::define_static_string(s)};
}

} // namespace tensor::detail
