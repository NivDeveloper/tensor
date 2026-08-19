#pragma once

// The grouping maps, composed purely from existing ops the way Gen/Dist.h
// composes the distributions. bins is the world→grid affine map — a
// subtraction, one scalar multiply, math::Floor; scalar bounds are the
// point: the scale NB / (hi - lo) is computed once, here, and travels as
// ONE scalar slot. cuts is the edge-list quantizer — an indicator sum, so
// its k+1 edges travel as k+1 scalar slots and the comparisons reach the
// device on the operator vocabulary's own terms.

#include "../Math.h"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace tensor {

// By const&, not value: a Tensor handed as a bound (move-only) must reach
// the static_assert below, not die on its deleted copy constructor.
template <size_t NB, Operand X, typename T>
constexpr auto bins(X &&x, const T &lo, const T &hi) {
    if constexpr (!std::is_floating_point_v<T>)
        static_assert(false, detail::bins_bounds_error(^^T));
    else if constexpr (NB == 0)
        static_assert(false, detail::bins_zero_error());
    else {
        const T s = T(NB) / (hi - lo);
        auto q = math::Floor((std::forward<X>(x) - lo) * s);
        // Tagged, so a policy can take its extent from the count spelled
        // here. A scalar operand never built a node and takes no tag.
        if constexpr (AnyExpr<decltype(q)>)
            return detail::Ranged<NB, decltype(q)>{q};
        else
            return q;
    }
}

template <size_t NB, size_t N> constexpr auto bins(Ix<N>) {
    if constexpr (NB == 0)
        static_assert(false, detail::bins_zero_error());
    else
        // A token: the extent being divided is the id's pinned one, which
        // only the census knows. scatter_dispatch resolves it.
        return detail::IxBins<NB, N>{};
}

template <size_t... Es, size_t N> constexpr auto cuts(Ix<N>) {
    if constexpr (sizeof...(Es) < 2)
        static_assert(false, detail::cuts_count_error());
    else if constexpr (![] {
                           constexpr std::array<size_t, sizeof...(Es)> e{Es...};
                           for (size_t q = 1; q < e.size(); ++q)
                               if (e[q] <= e[q - 1])
                                   return false;
                           return true;
                       }())
        static_assert(false, detail::ix_cuts_unsorted_error());
    else
        // A token: whether the cuts cover the axis is a question about the
        // extent, which only the census knows. scatter_dispatch answers it.
        return detail::IxCuts<N, Es...>{};
}

template <Operand X, typename... Ts>
constexpr auto cuts(X &&x, const Ts &...es) {
    if constexpr (sizeof...(Ts) < 2)
        static_assert(false, detail::cuts_count_error());
    else if constexpr (!(std::is_arithmetic_v<Ts> && ...)) {
        constexpr std::meta::info bad = [] {
            std::meta::info r = ^^void;
            auto probe = [&]<typename U>(std::type_identity<U>) {
                if (!std::is_arithmetic_v<U> && r == ^^void)
                    r = ^^U;
            };
            (probe(std::type_identity<Ts>{}), ...);
            return r;
        }();
        static_assert(false, detail::cuts_edge_error(bad));
    } else {
        using T = std::common_type_t<Ts...>;
        // By reference, spent k+1 times: each comparison takes its own
        // copy of x, so forward would move from all but the first.
        const auto &xr = x;
        auto q = (... + (xr >= T(es))) - T(1);
        // k+1 cut points bound k bins — the tag a policy reads.
        if constexpr (AnyExpr<decltype(q)>)
            return detail::Ranged<sizeof...(Ts) - 1, decltype(q)>{q};
        else
            return q;
    }
}

} // namespace tensor
