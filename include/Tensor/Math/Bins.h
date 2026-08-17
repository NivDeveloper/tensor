#pragma once

// bins: the world→grid affine map, composed purely from existing ops the
// way Gen/Dist.h composes the distributions — a subtraction, one scalar
// multiply, math::Floor. Scalar bounds are the point: the scale
// NB / (hi - lo) is computed once, here, and travels as ONE scalar slot.

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
        return math::Floor((std::forward<X>(x) - lo) * s);
    }
}

} // namespace tensor
