#pragma once

// formula and slots_of bodies — one render walk each;
// define_static_string persists the consteval std::string into the binary.

#include "../Introspect.h"
#include "../detail/Tree.h"

#include <meta>
#include <string_view>
#include <type_traits>

namespace tensor {

template <AnyExpr E> consteval std::string_view formula() {
    ExprSlots s{};
    return std::define_static_string(detail::render(^^std::remove_cvref_t<E>,
                                                    detail::formula_style, "i",
                                                    s.views, s.scalars));
}

template <AnyExpr E> consteval ExprSlots slots_of() {
    ExprSlots s{};
    (void)detail::render(^^std::remove_cvref_t<E>, detail::formula_style, "i",
                         s.views, s.scalars);
    return s;
}

template <detail::RangedExpr E> consteval std::string_view formula() {
    return formula<typename std::remove_cvref_t<E>::coord_type>();
}

template <detail::RangedExpr E> consteval ExprSlots slots_of() {
    return slots_of<typename std::remove_cvref_t<E>::coord_type>();
}

} // namespace tensor
