#pragma once

// The descriptive layer — the few statistics whose spelling hides a
// decision worth hiding:
//
//   float mu = eval(stats::Mean(x));       // an EXPRESSION: N from the type
//   float sd = eval(stats::Std(x - mu));   // one pass, via ops::Welford
//   auto  h  = stats::Histogram(x);        // {counts, edges}, two passes
//
// It is deliberately small. A name belongs here only if it removes a
// decision a caller would otherwise get wrong — the element count, the
// choice between the cancellation-prone E[x²]-E[x]² and Welford, a
// histogram's range and edge arithmetic. Everything else is already said
// by the fold and scatter vocabulary: a grouped sum is `scatter<i>(dest,
// v)`, both range ends are `fold<ops::MinMax>(x)`, an extremum and its
// location are `fold<ops::ArgMax>(x)`. Those spellings name the op that
// runs, so they stay the spelling.
//
// Most members return expressions and fuse into whatever uses them. The
// exception is the one whose PARAMETER comes from the data — a histogram
// cannot deposit before it knows its range — and it returns the answer
// directly, in one pass more than the information allows.
//
// Var and Std are the SAMPLE statistics, dividing by n-1: Mathematica's
// and Julia's convention, NOT numpy's 1/n default (numpy spells this
// var(ddof=1)). A single element therefore has NaN variance, the same
// answer an empty group gives.
//
// Definitions live in Stats/Stats.h, included at the bottom.

#include "Gen.h"
#include "Math.h"
#include "Tensor.h"
#include "detail/Stats.h"

#include <cstddef>
#include <type_traits>

namespace tensor {

// A histogram is two arrays that belong together, and naming its fields
// makes `npy::savez(path, h)` write a .npz Python reads with no schema:
// d["counts"], d["edges"] — matplotlib's stairs() takes exactly this pair.
template <typename T, size_t B> struct HistogramOf {
    Tensor<unsigned, B> counts;
    Tensor<T, B + 1> edges;
};

namespace stats {

// ── one pass, as expressions ────────────────────────────────────────────────

// The arithmetic mean. The element count comes from the TYPE, so there is
// no N to pass and none to get wrong.
template <AnyExpr E> constexpr auto Mean(E &&e);

// Mean and variance together — one traversal, through ops::Welford, so
// asking for both costs what asking for one does.
template <AnyExpr E> constexpr auto MeanVar(E &&e);

// The SAMPLE variance and its root (divisor n-1; see the header note).
// Spelt through ops::Welford, never through E[x²]-E[x]², which cancels.
// Both project a field of MeanVar's result with map<f>, so on a device
// they need the mapped-function opt-in that every map<f> needs; MeanVar
// itself is the bare fold and lowers with no opt-in at all.
template <AnyExpr E> constexpr auto Var(E &&e);
template <AnyExpr E> constexpr auto Std(E &&e);

// ── the program: a parameter the data has to supply first ───────────────────

// B equal bins over [min, max], nothing dropped: the range is one
// ops::MinMax pass and the deposit is the second, which is as few as the
// information allows. Edges are B+1 points, numpy's convention.
template <size_t B = 64, AnyExpr E> auto Histogram(E &&e);

// The same on a device. A PROGRAM takes its device, because it is a
// sequence — a range pass, then a deposit that needs the range — where an
// expression is handed to whichever eval the caller picked.
template <size_t B = 64, typename Dev, AnyExpr E>
auto Histogram(Dev &dev, E &&e);

// The B+1 bin edges of a named grid — bins<B>'s inverse, so a hand-written
// deposit can label its axis without redoing the spacing arithmetic.
template <size_t B = 64, typename T> constexpr auto Edges(T lo, T hi);

// The centre of the most populous bin; ties take the lowest bin.
template <size_t B = 64, AnyExpr E> auto Mode(E &&e);
template <size_t B = 64, typename Dev, AnyExpr E> auto Mode(Dev &dev, E &&e);

} // namespace stats
} // namespace tensor

// The definitions.
#include "Stats/Stats.h" // IWYU pragma: export
