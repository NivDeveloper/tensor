#pragma once

// The closed-form distributions. Each is its inverse CDF written over a
// uniform draw, so each is an ordinary expression — no new node, no new
// leaf, and the GPU lowers it because every piece already lowers.
//
// Two things every formula here is careful about. It reads (1 - u) rather
// than u wherever a log follows, because uniform draws [0, 1) and the log
// would otherwise see 0; and where the support is unbounded on both sides
// it draws from the OPEN interval, so no cell can return an infinity.

#include "../Gen.h"
#include "../Math.h"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace tensor::rng {

template <size_t... Extents, Operand R> constexpr auto Exponential(R &&rate) {
    using T = detail::element_t<std::remove_cvref_t<R>>;
    return -math::Log(T(1) - Uniform<T, Extents...>()) / std::forward<R>(rate);
}

template <size_t... Extents, Operand S> constexpr auto Rayleigh(S &&sigma) {
    using T = detail::element_t<std::remove_cvref_t<S>>;
    return std::forward<S>(sigma) *
           math::Sqrt(T(-2) * math::Log(T(1) - Uniform<T, Extents...>()));
}

template <size_t... Extents, Operand A, Operand K>
constexpr auto Weibull(A &&scale, K &&shape) {
    using T = detail::element_t<std::remove_cvref_t<A>>;
    return std::forward<A>(scale) *
           math::Pow(-math::Log(T(1) - Uniform<T, Extents...>()),
                     T(1) / std::forward<K>(shape));
}

template <size_t... Extents, Operand X, Operand A>
constexpr auto Pareto(X &&xm, A &&alpha) {
    using T = detail::element_t<std::remove_cvref_t<X>>;
    return std::forward<X>(xm) * math::Pow(T(1) - Uniform<T, Extents...>(),
                                           T(-1) / std::forward<A>(alpha));
}

template <size_t... Extents, Operand M, Operand S>
constexpr auto LogNormal(M &&mu, S &&sigma) {
    using T = detail::element_t<std::remove_cvref_t<M>>;
    return math::Exp(std::forward<M>(mu) +
                     std::forward<S>(sigma) * Normal<T, Extents...>());
}

template <size_t... Extents, Operand X, Operand G>
constexpr auto Cauchy(X &&x0, G &&gamma) {
    using T = detail::element_t<std::remove_cvref_t<X>>;
    const auto u = Uniform<T, Extents...>() * detail::open_scale<T>() +
                   detail::open_shift<T>();
    return std::forward<X>(x0) +
           std::forward<G>(gamma) *
               math::Tan(T(3.14159265358979323846) * (u - T(0.5)));
}

template <size_t... Extents, Operand M, Operand B>
constexpr auto Gumbel(M &&mu, B &&beta) {
    using T = detail::element_t<std::remove_cvref_t<M>>;
    const auto u = Uniform<T, Extents...>() * detail::open_scale<T>() +
                   detail::open_shift<T>();
    return std::forward<M>(mu) -
           std::forward<B>(beta) * math::Log(-math::Log(u));
}

template <size_t... Extents, Operand M, Operand S>
constexpr auto Logistic(M &&mu, S &&s) {
    using T = detail::element_t<std::remove_cvref_t<M>>;
    const auto u = Uniform<T, Extents...>() * detail::open_scale<T>() +
                   detail::open_shift<T>();
    return std::forward<M>(mu) +
           std::forward<S>(s) * math::Log(u / (T(1) - u));
}

template <size_t... Extents, Operand M, Operand B>
constexpr auto Laplace(M &&mu, B &&b) {
    using T = detail::element_t<std::remove_cvref_t<M>>;
    // Two draws, so two call sites and two independent streams.
    return std::forward<M>(mu) +
           std::forward<B>(b) *
               (-math::Log(T(1) - Uniform<T, Extents...>()) +
                math::Log(T(1) - Uniform<T, Extents...>()));
}

} // namespace tensor::rng
