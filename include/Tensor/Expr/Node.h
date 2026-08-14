#pragma once

// TensorView and Expr member bodies, the tuple protocol behind the
// structured-binding walkers, and for_each_leaf.

#include "../Expr.h"

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

// A fold node keeps element access — its cells are defined (survivors in
// first-appearance order) and the subscript stays the strict single-chain
// spelling; other index-bearing trees evaluate first.
template <typename Op, typename... Cs>
constexpr typename Expr<Op, Cs...>::type
Expr<Op, Cs...>::operator[](Index<rank> idx) const {
    static_assert(!detail::index_bearing_v<Expr> || detail::ContractNode<Expr>,
                  detail::indexed_element_error());
    detail::sync_leaf_hosts(*this);
    return detail::eval_node(*this, idx);
}
template <typename Op, typename... Cs>
constexpr typename Expr<Op, Cs...>::type
Expr<Op, Cs...>::operator[](size_t flat) const {
    static_assert(!detail::index_bearing_v<Expr> || detail::ContractNode<Expr>,
                  detail::indexed_element_error());
    detail::sync_leaf_hosts(*this);
    return detail::eval_node(*this, flat);
}

template <typename Op, typename... Cs>
template <detail::SubscriptArg... Sub>
    requires(detail::SubscriptTerm<Sub> || ...)
constexpr auto Expr<Op, Cs...>::operator[](Sub... sub) const {
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
        if constexpr (std::remove_cvref_t<Node>::padded)
            f(n.fill);
        for_each_leaf(n.e, f);
    } else {
        f(n);
    }
}

} // namespace tensor
