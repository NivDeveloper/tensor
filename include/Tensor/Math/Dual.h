#pragma once

// Dual's dispatch: a scalar argument list is a plain std call, a tensor
// operand builds the node with the annotated op type.

#include "../Math.h"

#include <utility>

namespace tensor::math {

template <typename Op>
template <typename... Cs>
constexpr auto Dual<Op>::operator()(Cs &&...cs) {
    if constexpr (Operands<Cs...>)
        return detail::make_expr<Op>(std::forward<Cs>(cs)...);
    else
        return Op{}(std::forward<Cs>(cs)...);
}

} // namespace tensor::math
