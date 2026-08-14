#pragma once

// The affine index algebra behind A[zero(i + j - k)] and 2_c * i.

#include "../Expr.h"

#include <initializer_list>
#include <type_traits>

namespace tensor {

template <detail::IndexTerm A, detail::IndexTerm B>
constexpr auto operator+(A, B) {
    return IxForm<detail::add_lin(detail::term_lin<A>(), detail::term_lin<B>(),
                                  1)>{};
}
template <detail::IndexTerm A, detail::IndexTerm B>
constexpr auto operator-(A, B) {
    return IxForm<detail::add_lin(detail::term_lin<A>(), detail::term_lin<B>(),
                                  -1)>{};
}
template <detail::IndexTerm A> constexpr auto operator-(A) {
    return IxForm<detail::scale_lin(detail::term_lin<A>(), -1)>{};
}
template <detail::IndexTerm A, detail::IndexTerm B>
constexpr auto operator*(A, B) {
    constexpr detail::Lin a = detail::term_lin<A>(), b = detail::term_lin<B>();
    static_assert(detail::lin_const(a) || detail::lin_const(b),
                  "an index expression is affine: a placeholder cannot "
                  "multiply a placeholder — one side must be a _c constant");
    if constexpr (detail::lin_const(a))
        return IxForm<detail::scale_lin(b, a.off)>{};
    else
        return IxForm<detail::scale_lin(a, b.off)>{};
}

template <detail::DecoratedTerm A, detail::IndexTerm B>
constexpr auto operator+(A, B) {
    return IxDec<detail::displace(std::remove_cvref_t<A>::map,
                                  detail::term_lin<B>(), 1)>{};
}
template <detail::IndexTerm A, detail::DecoratedTerm B>
constexpr auto operator+(A, B) {
    return IxDec<detail::displace(std::remove_cvref_t<B>::map,
                                  detail::term_lin<A>(), 1)>{};
}
template <detail::DecoratedTerm A, detail::IndexTerm B>
constexpr auto operator-(A, B) {
    return IxDec<detail::displace(std::remove_cvref_t<A>::map,
                                  detail::term_lin<B>(), -1)>{};
}

namespace indices {

template <detail::SubscriptTerm T> consteval auto wrap(T) {
    return IxDec<detail::decorate(detail::term_map<T>(),
                                  detail::Policy::Wrap)>{};
}
template <detail::SubscriptTerm T> consteval auto clamp(T) {
    return IxDec<detail::decorate(detail::term_map<T>(),
                                  detail::Policy::Clamp)>{};
}
template <detail::SubscriptTerm T> consteval auto zero(T) {
    return IxDec<detail::decorate(detail::term_map<T>(),
                                  detail::Policy::Zero)>{};
}
template <detail::SubscriptTerm T, typename V> constexpr auto pad(T, V v) {
    return IxPad<detail::decorate(detail::term_map<T>(), detail::Policy::Pad),
                 V>{v};
}

template <char... Cs> constexpr auto operator""_c() {
    constexpr int v = [] {
        int r = 0;
        for (char ch : {Cs...})
            if (ch != '\'')
                r = 10 * r + (ch - '0');
        return r;
    }();
    return IxForm<detail::const_lin(v)>{};
}

} // namespace indices

} // namespace tensor
