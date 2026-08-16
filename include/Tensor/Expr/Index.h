#pragma once

// The affine index algebra behind A[zero(i + j - k)] and 2_c * i, and the
// write-side policies behind scatter's wrap<8>(cell[i]).

#include "../Expr.h"

#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <utility>

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

template <detail::AffineTerm T> consteval auto wrap(T) {
    return IxDec<detail::decorate(detail::term_map<T>(),
                                  detail::Policy::Wrap)>{};
}
template <detail::AffineTerm T> consteval auto clamp(T) {
    return IxDec<detail::decorate(detail::term_map<T>(),
                                  detail::Policy::Clamp)>{};
}
template <detail::AffineTerm T> consteval auto zero(T) {
    return IxDec<detail::decorate(detail::term_map<T>(),
                                  detail::Policy::Zero)>{};
}
template <detail::AffineTerm T, typename V> constexpr auto pad(T, V v) {
    return IxPad<detail::decorate(detail::term_map<T>(), detail::Policy::Pad),
                 V>{v};
}

// The read side on a GATHERED coordinate. constexpr, not consteval: the
// coordinate is a runtime expression and only its policy is compile-time.
// The extent still comes from the operand's own axis, unlike the write side
// below, which has no operand to ask.
template <IndexedExpr C> constexpr auto wrap(C &&c) {
    return IxData<detail::Policy::Wrap, std::remove_cvref_t<C>>{
        std::forward<C>(c)};
}
template <IndexedExpr C> constexpr auto clamp(C &&c) {
    return IxData<detail::Policy::Clamp, std::remove_cvref_t<C>>{
        std::forward<C>(c)};
}
template <IndexedExpr C> constexpr auto zero(C &&c) {
    return IxData<detail::Policy::Zero, std::remove_cvref_t<C>>{
        std::forward<C>(c)};
}
template <IndexedExpr C, typename V> constexpr auto pad(C &&c, V v) {
    return IxDataPad<detail::Policy::Pad, std::remove_cvref_t<C>, V>{
        std::forward<C>(c), v};
}

// The write side. constexpr, not consteval: the coordinate is a runtime
// expression, and only its policy and extent are compile-time. The extent
// IS compile-time, though, which is what lets a destination too large for
// its coordinate's type be refused here rather than discovered as wrong
// numbers later (ACC-L4).
#define TENSOR_PLACE_CAPACITY(E, C)                                            \
    static_assert(                                                             \
        E <= detail::index_capacity<                                           \
                 std::remove_cvref_t<typename std::remove_cvref_t<C>::type>>(), \
        detail::index_capacity_error(                                          \
            ^^std::remove_cvref_t<typename std::remove_cvref_t<C>::type>, E,   \
            detail::index_capacity<                                            \
                std::remove_cvref_t<typename std::remove_cvref_t<C>::type>>()))
template <size_t E, IndexedExpr C> constexpr auto wrap(C &&c) {
    TENSOR_PLACE_CAPACITY(E, C);
    return detail::Placed<detail::Place::Wrap, E, std::remove_cvref_t<C>>{
        std::forward<C>(c)};
}
template <size_t E, IndexedExpr C> constexpr auto clamp(C &&c) {
    TENSOR_PLACE_CAPACITY(E, C);
    return detail::Placed<detail::Place::Clamp, E, std::remove_cvref_t<C>>{
        std::forward<C>(c)};
}
template <size_t E, IndexedExpr C> constexpr auto drop(C &&c) {
    TENSOR_PLACE_CAPACITY(E, C);
    return detail::Placed<detail::Place::Drop, E, std::remove_cvref_t<C>>{
        std::forward<C>(c)};
}

#undef TENSOR_PLACE_CAPACITY

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
