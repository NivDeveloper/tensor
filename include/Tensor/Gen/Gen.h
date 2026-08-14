#pragma once

// The generator builders. Each is one aggregate initialisation — the value
// algebra lives on Generator itself (Expr/Node.h) so the CPU and the
// emitted shader read from one definition.

#include "../Gen.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tensor {

template <size_t... Extents, typename T>
    requires detail::is_broadcast_scalar_v<T>
constexpr Generator<detail::GenKind::Fill, T, Extents...> fill(T v) {
    return {v, T{}};
}

template <size_t... Extents, typename T>
    requires detail::is_broadcast_scalar_v<T>
constexpr Generator<detail::GenKind::Iota, T, Extents...> iota(T start) {
    return {start, T{}};
}

template <size_t N, typename T>
    requires(N >= 1 && std::is_floating_point_v<T>)
constexpr Generator<detail::GenKind::LinSpace, T, N> linspace(T a, T b) {
    return {a, b};
}

inline void seed(std::uint64_t s) {
    detail::seed_slot().store(s, std::memory_order_relaxed);
    detail::draw_slot().store(0, std::memory_order_relaxed);
}

template <typename T, size_t... Extents>
    requires std::is_floating_point_v<T>
Generator<detail::GenKind::Uniform, T, Extents...> uniform() {
    return {T{}, T{}, detail::claim_stream()};
}

template <typename T, size_t... Extents>
    requires std::is_floating_point_v<T>
Generator<detail::GenKind::Normal, T, Extents...> normal() {
    return {T{}, T{}, detail::claim_stream()};
}

} // namespace tensor
