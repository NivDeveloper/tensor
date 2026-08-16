#pragma once

// TensorView, Generator and Expr member bodies, the tuple protocol behind
// the structured-binding walkers, and for_each_leaf.

#include "../Expr.h"
#include "../detail/Rng.h"

#include <cstddef>
#include <mdspan>
#include <tuple>
#include <type_traits>

namespace tensor {

template <typename T, size_t... Extents>
constexpr T TensorView<T, Extents...>::operator[](Index<rank> idx) const {
    return std::mdspan<T, extents_type>(data)[idx];
}
template <typename T, size_t... Extents>
constexpr T TensorView<T, Extents...>::operator[](size_t flat) const {
    return data[flat];
}

template <typename T, size_t... Extents>
template <detail::SubscriptArg... Sub>
    requires(detail::SubscriptTerm<Sub> || ...)
constexpr auto TensorView<T, Extents...>::operator[](Sub... sub) const {
    return detail::make_indexed<TensorView, extents_type, Sub...>(*this, sub...);
}

// The generator value, defined once against the flat index — the emitted
// shader spells this same algebra inline. linspace interpolates from BOTH
// ends rather than stepping from one, so the first and last cells land on
// a and b exactly. A sampler reads its cell's own counter-based stream, so
// the value depends on nothing but (key, cell) — not on thread count, chunk
// boundaries, or which device ran it.
template <detail::GenKind K, typename T, size_t... Extents>
constexpr T Generator<K, T, Extents...>::operator[](size_t flat) const {
    if constexpr (K == detail::GenKind::Fill)
        return a;
    else if constexpr (K == detail::GenKind::Iota)
        return T(a + T(flat));
    else if constexpr (K == detail::GenKind::Uniform) {
        const auto r = detail::philox_cell(key, flat);
        return detail::uniform01<T>(r.x, r.y);
    } else if constexpr (K == detail::GenKind::Normal)
        return detail::normal01<T>(detail::philox_cell(key, flat));
    else if constexpr (K == detail::GenKind::Stream)
        return T{key, flat}; // this cell's own private stream
    else if constexpr (count <= 1)
        return a;
    else {
        const T t = T(flat) / T(count - 1);
        return a * (T(1) - t) + b * t;
    }
}

template <detail::GenKind K, typename T, size_t... Extents>
constexpr T Generator<K, T, Extents...>::operator[](Index<rank> idx) const {
    size_t flat = 0;
    for (size_t d = 0; d < rank; ++d)
        flat = flat * extents_type::static_extent(d) + idx[d];
    return (*this)[flat];
}

// The shapeless form has only the flat spelling: with no extents of its
// own there is no multi-index to decompose, and the cell it draws for is
// whichever one the surrounding loop is computing.
template <detail::GenKind K, typename T>
constexpr T Generator<K, T>::operator[](size_t flat) const {
    if constexpr (K == detail::GenKind::Uniform) {
        const auto r = detail::philox_cell(key, flat);
        return detail::uniform01<T>(r.x, r.y);
    } else if constexpr (K == detail::GenKind::Normal)
        return detail::normal01<T>(detail::philox_cell(key, flat));
    else if constexpr (K == detail::GenKind::Stream)
        return T{key, flat};
    else
        return a;
}

// A fold node keeps element access — its cells are defined (survivors in
// first-appearance order) and the subscript stays the strict single-chain
// spelling; other index-bearing trees evaluate first.
// Flattened here rather than passed down: every operand of a plain tree
// shares the node's extents, so the flat index is the same cell — and a
// leaf with no extents of its own (a shapeless generator) has nothing to
// decompose a multi-index with.
template <typename Op, typename... Cs>
constexpr typename Expr<Op, Cs...>::type
Expr<Op, Cs...>::operator[](Index<rank> idx) const {
    static_assert(detail::scatter_count_v<Expr> == 0,
                  detail::scatter_element_error());
    static_assert(!detail::index_bearing_v<Expr> || detail::FoldOnlyNode<Expr>,
                  detail::indexed_element_error());
    detail::sync_leaf_hosts(*this);
    size_t flat = 0;
    for (size_t d = 0; d < rank; ++d)
        flat = flat * extents_type::static_extent(d) + idx[d];
    return detail::eval_node(*this, flat);
}
template <typename Op, typename... Cs>
constexpr typename Expr<Op, Cs...>::type
Expr<Op, Cs...>::operator[](size_t flat) const {
    static_assert(detail::scatter_count_v<Expr> == 0,
                  detail::scatter_element_error());
    static_assert(!detail::index_bearing_v<Expr> || detail::FoldOnlyNode<Expr>,
                  detail::indexed_element_error());
    detail::sync_leaf_hosts(*this);
    return detail::eval_node(*this, flat);
}

template <typename Op, typename... Cs>
template <detail::SubscriptArg... Sub>
    requires(detail::SubscriptTerm<Sub> || ...)
constexpr auto Expr<Op, Cs...>::operator[](Sub... sub) const {
    // A scan is index-bearing, so the assert below already refuses it; this
    // one goes first so the message names the fix rather than the symptom.
    static_assert(detail::scan_count_v<Expr> == 0,
                  detail::scan_below_subscript_error());
    static_assert(!detail::has_free_index_v<Expr>,
                  "an index-bearing expression cannot be subscripted again");
    static_assert(!detail::tree_has_contraction(^^Expr),
                  detail::fold_below_subscript_error());
    if constexpr (!detail::has_free_index_v<Expr>)
        return detail::make_indexed<Expr, extents_type, Sub...>(*this, sub...);
}

// The tuple protocol: const auto& [...children] = node;
template <size_t I, typename Op, typename... Cs>
constexpr const auto &get(const Expr<Op, Cs...> &e) {
    return detail::slot_get<I>(e.children);
}

} // namespace tensor

template <typename Op, typename... Cs>
struct std::tuple_size<tensor::Expr<Op, Cs...>>
    : std::integral_constant<size_t, sizeof...(Cs)> {};
template <size_t I, typename Op, typename... Cs>
struct std::tuple_element<I, tensor::Expr<Op, Cs...>> {
    using type = Cs...[I]; // pack indexing: no std::tuple in the protocol
};

namespace tensor {

template <typename Node, typename F>
constexpr void for_each_leaf(const Node &n, F &&f) {
    if constexpr (detail::ExprNode<std::remove_cvref_t<Node>>) {
        const auto &[... children] = n;
        (for_each_leaf(children, f), ...);
    } else if constexpr (detail::is_indexed_v<std::remove_cvref_t<Node>>) {
        // The fill is a scalar leaf of its own, ahead of the operand's —
        // the order the emitter's census and the host packing both walk.
        // The order the emitter's census and the host packing both walk:
        // the fill's scalar slot, then each gathered axis's coordinate in
        // axis order, then the operand's leaves.
        if constexpr (std::remove_cvref_t<Node>::padded)
            f(n.fill);
        detail::for_each_slot(n.d, [&](const auto &c) { for_each_leaf(c, f); });
        for_each_leaf(n.e, f);
    } else if constexpr (std::is_same_v<std::remove_cvref_t<Node>,
                                        detail::NoCoord>) {
        // an affine axis carries no coordinate
    } else if constexpr (detail::is_placed_v<std::remove_cvref_t<Node>>) {
        for_each_leaf(n.c, f); // a destination is its coordinate's leaves
    } else {
        f(n);
    }
}

} // namespace tensor
