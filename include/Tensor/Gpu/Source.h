#pragma once

// gpu_source's body: the four capability gates, then the generated program.

#include "../Gpu.h"
#include "../detail/Gpu.h"

#include <meta>
#include <string_view>
#include <type_traits>

namespace tensor {

template <AnyExpr E, auto... Order> consteval std::string_view gpu_source() {
    using D = std::remove_cvref_t<E>;
    // Every gate reads the memoized census, and fires through a DISCARDED
    // branch: a static_assert's message is constant-evaluated even when the
    // assertion holds, and these messages walk the tree and render it a
    // second time. Eagerly, the six cost more than the program they guard.
    constexpr auto gates = detail::gpu_gates<D>;
    if constexpr (gates.cpu_only)
        static_assert(false, detail::gpu_cpu_only_error(^^D));
    if constexpr (!gates.streamless)
        static_assert(false, detail::gpu_sample_error(^^D));
    if constexpr (!gates.emissible)
        static_assert(false, detail::gpu_map_error(^^D));
    if constexpr (gates.bad_type)
        static_assert(false, detail::gpu_type_error(^^D));
    if constexpr (!gates.fold_fits)
        static_assert(false, detail::gpu_fold_size_error(^^D));
    if constexpr (!gates.scan_fits)
        static_assert(false, detail::gpu_scan_rows_error(^^D));
    // A permuted layout is a distinct program — a distinct memoization key.
    return std::define_static_string(detail::gpu_program(
        ^^D, detail::fold_identity<E>(), {detail::order_id<Order>()...}));
}

} // namespace tensor
