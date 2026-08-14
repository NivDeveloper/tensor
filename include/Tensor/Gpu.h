#pragma once

// gpu_source<E>(): the complete Slang compute shader for expression type E,
// generated at compile time. The ABI is positional, in for_each_leaf's DFS
// order, so host code fills the push constants with the same walk.
// Definition in Gpu/Source.h, included at the bottom.

#include "Core.h"

#include <string_view>

namespace tensor {

// The gates fire wherever a program is generated — direct calls included.
template <AnyExpr E, auto... Order> consteval std::string_view gpu_source();

} // namespace tensor

// The definition.
#include "Gpu/Source.h" // IWYU pragma: export
