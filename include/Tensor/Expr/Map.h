#pragma once

// map<f>: lift a plain function of any arity into an expression node —
//
//   constexpr float halve(float x) { return x / 2; }
//   auto e = map<halve>(t);            // formula: "halve(in0[i])"
//
// Plain functions only: the math vocabulary (Math.h) is spelt
// directly — math::Sqrt(t), never map<math::Sqrt>(t). Mapped functions
// are CPU-only unless the kernel-translation opt-in lowers them. Under
// use_threads(n) a mapped function must be thread-safe.

#include "../Expr.h"
#include "../detail/Tree.h"

#include <meta>
#include <type_traits>
#include <utility>

namespace tensor {

namespace math {
template <typename Op> struct Dual; // Math.h — the vocabulary objects
}

// clang-format off: annotations predate clang-format's parser
namespace ops {

template <std::meta::info F> struct [[=detail::fn_symbol(F)]] Fn {
    static constexpr auto operator()(auto... a) { return [:F:](a...); }
};

} // namespace ops
// clang-format on

template <auto &F, typename... Cs>
    requires Operands<Cs...>
constexpr auto map(Cs &&...cs) {
    using Op = std::remove_cvref_t<decltype(F)>;
    constexpr bool vocabulary =
        detail::is_specialization_of(^^Op, ^^math::Dual) ||
        detail::carries_symbol(^^Op);
    constexpr auto vocab_op = [] {
        if constexpr (detail::is_specialization_of(^^Op, ^^math::Dual))
            return std::meta::dealias(^^typename Op::op);
        else
            return ^^Op;
    }();
    static_assert(!vocabulary, detail::map_vocabulary_error(vocab_op));
    if constexpr (!vocabulary)
        return detail::make_expr<ops::Fn<detail::entity_of<F>()>>(
            std::forward<Cs>(cs)...);
}

} // namespace tensor
