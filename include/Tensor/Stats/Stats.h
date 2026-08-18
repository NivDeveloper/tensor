#pragma once

// The stats layer's definitions. Every one-pass member is a composition of
// the fold vocabulary — no new node, no new op — so each fuses and reaches
// the device on the same terms its pieces do.

#include "../Stats.h"

#include <cstddef>
#include <utility>

namespace tensor::stats {

template <AnyExpr E> constexpr auto Mean(E &&e) {
    using T = detail::element_t<std::remove_cvref_t<E>>;
    constexpr size_t n = eval_result_t<std::remove_cvref_t<E>>::element_count;
    return fold(std::forward<E>(e)) * (T(1) / T(n));
}

template <AnyExpr E> constexpr auto MeanVar(E &&e) {
    return fold<ops::Welford>(std::forward<E>(e));
}

template <AnyExpr E> constexpr auto Var(E &&e) {
    return map<detail::var_field>(fold<ops::Welford>(std::forward<E>(e)));
}

template <AnyExpr E> constexpr auto Std(E &&e) {
    return math::Sqrt(Var(std::forward<E>(e)));
}

// ── the program ─────────────────────────────────────────────────────────────

template <size_t B, typename T> constexpr auto Edges(T lo, T hi) {
    return lo + gen::Iota<B + 1>(T(0)) * ((hi - lo) / T(B));
}

// The device twin. A program takes its device explicitly because it IS a
// sequence — two dispatches with a dependency between them — where an
// expression only has to be handed to the eval the caller chose.
template <size_t B, typename Dev, AnyExpr E> auto Histogram(Dev &dev, E &&e) {
    using namespace tensor::indices;
    using T = detail::element_t<std::remove_cvref_t<E>>;
    auto x = eval(dev, std::forward<E>(e));
    auto [lo, hi] = eval(dev, fold<ops::MinMax>(x));
    detail::widen_degenerate(lo, hi);
    auto counts = eval(dev, scatter<i>(clamp<B>(bins<B>(x[i], lo, hi)), 1u));
    return HistogramOf<T, B>{std::move(counts), eval(dev, Edges<B>(lo, hi))};
}

template <size_t B, AnyExpr E> auto Histogram(E &&e) {
    using namespace tensor::indices;
    using T = detail::element_t<std::remove_cvref_t<E>>;
    auto x = eval(std::forward<E>(e));
    // Pass one: both ends together. Pass two: the deposit. A parameter the
    // data supplies cannot be known before the data is read, so two is the
    // floor — and clamp keeps the maximum, which sits exactly on hi.
    auto [lo, hi] = eval(fold<ops::MinMax>(x));
    detail::widen_degenerate(lo, hi);
    auto counts = eval(scatter<i>(clamp<B>(bins<B>(x[i], lo, hi)), 1u));
    return HistogramOf<T, B>{std::move(counts), eval(Edges<B>(lo, hi))};
}

template <size_t B, AnyExpr E> auto Mode(E &&e) {
    auto h = Histogram<B>(std::forward<E>(e));
    auto peak = eval(fold<ops::ArgMax>(h.counts)); // ties: the lowest bin
    return (h.edges[peak.at] + h.edges[peak.at + 1]) / 2;
}

template <size_t B, typename Dev, AnyExpr E> auto Mode(Dev &dev, E &&e) {
    auto h = Histogram<B>(dev, std::forward<E>(e));
    // The histogram is the pass worth putting on the device; picking the
    // tallest of B counts is not, and the counts are integral, which a
    // structured op does not lower (its state is float). Host it.
    auto peak = eval(fold<ops::ArgMax>(h.counts));
    return (h.edges[peak.at] + h.edges[peak.at + 1]) / 2;
}

} // namespace tensor::stats
