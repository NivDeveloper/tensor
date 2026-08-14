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
    static_assert(detail::first_cpu_only_op(^^D) == std::meta::info{},
                  detail::gpu_cpu_only_error(^^D));
    static_assert(detail::tree_gpu_streamless(^^D),
                  detail::gpu_sample_error(^^D));
    static_assert(detail::tree_gpu_emissible(^^D), detail::gpu_map_error(^^D));
    static_assert(detail::first_unmappable_type(^^D) == std::meta::info{},
                  detail::gpu_type_error(^^D));
    static_assert(detail::fold_fits_one_group<E>(),
                  detail::gpu_fold_size_error(^^D));
    // A permuted layout is a distinct program — a distinct memoization key.
    return std::define_static_string(detail::gpu_program(
        ^^D, detail::fold_identity<E>(), {detail::order_id<Order>()...}));
}

} // namespace tensor
