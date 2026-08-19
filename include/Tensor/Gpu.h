#pragma once

// The GPU surface. gpu_source<E>(): the complete Slang compute shader for
// expression type E, generated at compile time — always available, and free
// of any device dependency. The ABI is positional, in for_each_leaf's DFS
// order, so host code fills the push constants with the same walk.
//
// eval(dev, e) — running that shader — arrives with the TENSOR_ENABLE_GPU
// opt-in, the only part that needs the device layer:
//
//   auto r = eval(*dev, a + b * 2.0f);   // same expression as eval(a + …)
//
// Definitions in Gpu/Source.h and Gpu/Eval.h, included at the bottom.

#include "Core.h"

#include <string_view>

namespace tensor {

// The gates fire wherever a program is generated — direct calls included.
template <AnyExpr E, auto... Order> consteval std::string_view gpu_source();

// A whole-tensor fold past one workgroup's budget SPLITS: gpu_source is then
// the partial pass, writing one accumulator per group, and this is the
// combine pass that reduces them to the answer. Empty when the fold fits one
// group, which is how eval knows whether to issue the second dispatch.
template <AnyExpr E> consteval std::string_view gpu_combine_source();

// A range-tagged coordinate generates the program of the coordinate it
// wraps: the tag never entered a tree, so there is nothing else to emit.
template <detail::RangedExpr E, auto... Order>
consteval std::string_view gpu_source();
template <detail::RangedExpr E> consteval std::string_view
gpu_combine_source();

#ifndef TENSOR_GPU_ENABLED
// Without the opt-in there is no device type to overload on, so the spelling
// is caught structurally: any two-argument eval names the flag it is missing
// rather than failing as an unknown function.
template <typename Device, AnyExpr E>
auto eval(Device &, const E &) = delete(
    "GPU evaluation is opt-in: configure with -DTENSOR_ENABLE_GPU=ON and "
    "link tensor::gpu");
template <typename Device, detail::RangedExpr E>
auto eval(Device &, const E &) = delete(
    "GPU evaluation is opt-in: configure with -DTENSOR_ENABLE_GPU=ON and "
    "link tensor::gpu");
#endif

} // namespace tensor

// The definitions.
#include "Gpu/Source.h" // IWYU pragma: export
#ifdef TENSOR_GPU_ENABLED
#include "Gpu/Eval.h" // IWYU pragma: export
#endif
