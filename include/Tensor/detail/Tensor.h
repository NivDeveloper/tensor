#pragma once

// Plumbing behind Tensor.h — not for direct use.

#include "../Core.h"
#include "Diagnostics.h"
#include "Tree.h"

#include <cstddef>
#include <memory>
#include <meta>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensor::detail {

// A Tensor's residency state, reached from its views by pointer. Presence
// of storage == the device copy is valid; !host_valid ⇒ storage non-null
// (a stale host always has a device copy to sync from).
struct ShadowSlot {
    std::unique_ptr<DeviceStorage> storage;
    bool host_valid = true;
};

// The index-fill constraint: invocable with one size_t per axis (the pack
// carries the rank).
template <size_t> using index_arg = size_t;

template <typename F, typename T, typename Seq>
inline constexpr bool index_fill_for = false;

template <typename F, typename T, size_t... Is>
inline constexpr bool index_fill_for<F, T, std::index_sequence<Is...>> =
    std::is_invocable_r_v<T, F &, index_arg<Is>...>;

template <typename F, typename T, size_t Rank>
concept IndexFill = index_fill_for<F, T, std::make_index_sequence<Rank>>;

// The owning Tensor an expression materializes into: the extents' args with
// the index-type slot swapped for the element type. An index-bearing tree's
// extents are its free-index space (a lone indexed leaf carries no alias),
// permuted by eval's Order when one is named; a malformed Order falls back
// to the default so eval's own asserts get to name the offender.
template <typename E, auto... Order>
consteval std::meta::info eval_result_of() {
    using D = std::remove_cvref_t<E>;
    auto ext = [] {
        // A scatter's axes are its destinations first, so the free-index
        // space below would drop them: its own extents are authoritative.
        if constexpr (scatter_count_v<D> == 1) {
            return std::meta::dealias(^^typename D::extents_type);
        } else if constexpr (index_bearing_v<D> && sizeof...(Order) > 0) {
            const auto p = free_plan(std::meta::dealias(^^D));
            const std::vector<size_t> ids{order_id<Order>()...};
            if (order_mismatch(p, ids) != index_slots)
                return free_extents_of(id_census(std::meta::dealias(^^D)));
            const auto q = ordered_plan(p, ids);
            std::vector<std::meta::info> args{^^size_t};
            for (size_t a = 0; a < q.n; ++a)
                args.push_back(std::meta::reflect_constant(q.ext[a]));
            return std::meta::substitute(^^std::extents, args);
        } else if constexpr (index_bearing_v<D>) {
            return free_extents_of(id_census(std::meta::dealias(^^D)));
        } else {
            return std::meta::dealias(^^typename D::extents_type);
        }
    }();
    auto args = std::meta::template_arguments_of(ext);
    args[0] = ^^std::remove_cvref_t<typename D::type>;
    return std::meta::substitute(^^Tensor, args);
}

} // namespace tensor::detail
