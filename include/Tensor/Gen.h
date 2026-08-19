#pragma once

// Two vocabularies of storage-free expression leaves, spelt like the math
// one — a namespace and a capital:
//
//   gen::Fill<3, 4>(0.0f)          every cell the same value
//   gen::Iota<10>(1)               1, 2, … 10
//   gen::LinSpace<128>(0.0f, 1.0f) 128 points, both endpoints exact
//
//   rng::Seed(2026)                pin the run; random otherwise
//   rng::Normal<float, 64, 64>()   N(0, 1), and Uniform for [0, 1)
//   rng::Exponential<1024>(rate)   and eight more distributions
//   rng::Choice<int, 1024>({-1, 1})  one of the values listed
//   rng::Sample<f, 1024>(args…)    anything needing rejection
//
// Extents are explicit, the element type comes from the argument. These are
// expressions, so they fuse — eval(math::Sin(gen::LinSpace<64>(0.0f, tau)))
// is one pass storing no ramp, and on the GPU a generator costs a parameter
// or two instead of a buffer, an upload and an ABI slot.
//
// Log spacing is the composition: eval(math::Exp(gen::LinSpace<N>(la, lb))).
//
// Definitions live in Gen/*.h, included at the bottom.

#include "Core.h"
#include "Expr.h"
#include "detail/Gen.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tensor {

// ── gen:: the deterministic shapes ──────────────────────────────────────────
namespace gen {

// Every cell the same value.
template <size_t... Extents, typename T>
    requires detail::is_broadcast_scalar_v<T>
constexpr Generator<detail::GenKind::Fill, T, Extents...> Fill(T v);

// start, start + 1, … in row-major visiting order — the coordinate itself
// as a value, which is also what stratified sampling wants.
template <size_t... Extents, typename T>
    requires detail::is_broadcast_scalar_v<T>
constexpr Generator<detail::GenKind::Iota, T, Extents...> Iota(T start);

// N points from a to b inclusive, interpolated from BOTH ends so that
// x[0] == a and x[N-1] == b hold exactly; N == 1 yields {a}. There is
// deliberately no floating-point `arange`: its length is ceil((b-a)/step),
// which floating point makes unpredictable, and here the length IS the type.
template <size_t N, typename T>
    requires(N >= 1 && std::is_floating_point_v<T>)
constexpr Generator<detail::GenKind::LinSpace, T, N> LinSpace(T a, T b);

} // namespace gen

// ── rng:: sampling ──────────────────────────────────────────────────────────
// A sample is a pure function of (seed, stream, coordinate): identical at
// any thread count and on any device, and never disturbed by what a
// neighbouring cell drew. Each call site takes the next stream, so two
// samplers in one expression are independent with nothing to spell.
//
//   auto n = eval(mu + sigma * rng::Normal<float, 64, 64>());
//
// The draw happens where the sampler is WRITTEN, which keeps eval a pure
// function of its expression: `auto e = rng::Normal<float,8>(); eval(e);`
// twice gives the same numbers, while a sampler written inside a loop body
// draws afresh on every pass.
namespace rng {

// One cell's private stream — the argument a sampling function takes.
using Stream = detail::Rng;

// Pin the run. Absent this the seed is drawn from entropy, so a program is
// reproducible only when it asks to be. Resets the stream counter, so the
// draws that follow are the same ones every time. Externally synchronised
// with expression building, like use_threads is with eval.
void Seed(std::uint64_t s);

// [0, 1) — never 1.0, so log(1 - u) is finite for every draw.
template <typename T, size_t... Extents>
    requires std::is_floating_point_v<T>
Generator<detail::GenKind::Uniform, T, Extents...> Uniform();

// N(0, 1). Shift and scale with ordinary arithmetic — mu + sigma * Normal
// is that distribution, and mu and sigma may be tensors.
template <typename T, size_t... Extents>
    requires std::is_floating_point_v<T>
Generator<detail::GenKind::Normal, T, Extents...> Normal();

// ── the closed-form distributions ───────────────────────────────────────
// Each is its inverse CDF over a uniform draw, so each is an ordinary
// expression: it fuses, and it reaches the GPU with no extra machinery.
// Parameters broadcast — scalar or per cell — and every one is total: no
// draw returns an infinity.
//
// Extents are always spelt here, unlike Uniform/Normal: these compose with
// scalar constants, and a node of nothing but scalars pins no shape.

// Mean 1/rate.
template <size_t... Extents, Operand R> constexpr auto Exponential(R &&rate);

// Scale sigma; the magnitude of a 2-D standard normal vector.
template <size_t... Extents, Operand S> constexpr auto Rayleigh(S &&sigma);

// scale * (-log U)^(1/shape).
template <size_t... Extents, Operand A, Operand K>
constexpr auto Weibull(A &&scale, K &&shape);

// Power law with minimum xm and exponent alpha.
template <size_t... Extents, Operand X, Operand A>
constexpr auto Pareto(X &&xm, A &&alpha);

// exp(mu + sigma * N(0,1)) — mu and sigma are of the LOG, as std has them.
template <size_t... Extents, Operand M, Operand S>
constexpr auto LogNormal(M &&mu, S &&sigma);

// Location x0, half-width gamma. No mean and no variance — heavy tails.
template <size_t... Extents, Operand X, Operand G>
constexpr auto Cauchy(X &&x0, G &&gamma);

// Extreme-value type I: location mu, scale beta.
template <size_t... Extents, Operand M, Operand B>
constexpr auto Gumbel(M &&mu, B &&beta);

// Location mu, scale s.
template <size_t... Extents, Operand M, Operand S>
constexpr auto Logistic(M &&mu, S &&s);

// Two-sided exponential, built as the difference of two — which is what it
// is, and cheaper than the signed inverse CDF.
template <size_t... Extents, Operand M, Operand B>
constexpr auto Laplace(M &&mu, B &&b);

// ── the discrete draw ───────────────────────────────────────────────────
// One of the values listed, per cell. The element type is spelt first, as
// Uniform and Normal spell theirs, and the values give their own count:
//
//   rng::Choice<int, 64, 64>({-1, 1})          fair
//   rng::Choice<int, 64, 64>({-1, 1}, {1, 2})  one-to-two
//
// Weights need not be normalized and may be integers, so {1, 2} is exactly
// one-to-two. Composed like the families above — the cumulative weights
// are the cut points of an indicator sum over a single uniform draw — so
// it fuses and lowers to the device with nothing new.
template <typename T, size_t... Extents, size_t K>
    requires std::is_arithmetic_v<T>
constexpr auto Choice(const T (&vals)[K]);

template <typename T, size_t... Extents, size_t K, typename W, size_t KW>
    requires(std::is_arithmetic_v<T> && std::is_arithmetic_v<W>)
constexpr auto Choice(const T (&vals)[K], const W (&wts)[KW]);

// ── Sample<f>: any distribution at all ──────────────────────────────────
// The open end of the vocabulary, as map<f> is for functions. Write a plain
// function taking a Stream — rejection loops included, since each cell
// draws from its own and a variable number of draws in one cell cannot
// disturb another:
//
//   float gamma(rng::Stream &r, float a) { for (;;) { … r.uniform() … } }
//   auto g = eval(rng::Sample<gamma, 1024>(2.5f));
//
// Extra arguments are ordinary operands, so they broadcast. CPU-only: the
// shader would need both the function body and the stream plumbing.
template <auto &F, size_t... Extents, typename... Cs>
constexpr auto Sample(Cs &&...cs);

} // namespace rng

namespace ops {
template <std::meta::info F> struct Sampled; // rng::Sample's node
}

// Materializing a shapeless sampler on its own has no answer — how many
// cells? The deleted overload says so instead of failing as an unknown
// function, since a shapeless generator is deliberately not an AnyExpr.
template <typename G>
    requires detail::is_shapeless_generator_v<std::remove_cvref_t<G>>
auto eval(const G &) = delete(
    "a shapeless sampler has no extent of its own and nothing pins one — "
    "give it extents, rng::Uniform<float, 64>(), or combine it with an "
    "operand that has a shape");

} // namespace tensor

#include "Gen/Gen.h"    // IWYU pragma: export
#include "Gen/Dist.h"   // IWYU pragma: export
#include "Gen/Sample.h" // IWYU pragma: export

// The ops lint, re-swept because this surface defines one of its own.
#include "detail/OpsCheck.h"
