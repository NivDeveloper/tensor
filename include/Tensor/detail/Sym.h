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

// Marks an op with no Slang intrinsic; eval(dev, …) rejects it by name.
struct CpuOnly {};

} // namespace tensor::detail
