#pragma once

// sample<f>: lift a plain sampling function into an expression node, with a
// per-cell stream prepended to its arguments. The node is an ordinary
// Expr — what makes it a sampler is its first child, a Stream generator.

#include "../Gen.h"
#include "../detail/Meta.h"
#include "../detail/Rng.h"

#include <cstddef>
#include <meta>
#include <utility>

namespace tensor {

// clang-format off: annotations predate clang-format's parser
namespace ops {

template <std::meta::info F> struct [[=detail::fn_symbol(F)]] Sampled {
    static constexpr auto operator()(auto r, auto... a) { return [:F:](r, a...); }
};

} // namespace ops
// clang-format on

template <auto &F, size_t... Extents, typename... Cs>
constexpr auto sample(Cs &&...cs) {
    return detail::make_expr<ops::Sampled<detail::entity_of<F>()>>(
        Generator<detail::GenKind::Stream, detail::Rng, Extents...>{
            {}, {}, detail::claim_stream()},
        std::forward<Cs>(cs)...);
}

} // namespace tensor
