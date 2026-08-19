#pragma once

// The one fold: consume listed indices with a reducible op —
//
//   fold<j>(A[i,j] * x[j])            // contraction (Add is the default)
//   fold<ops::Max, j>(A[i,j] + c[j])  // op-generic indexed fold
//   fold<ops::Max>(E * K * f)         // whole-expression fold → the scalar
//   fold<ops::Add, 0, 2>(t)           // axis numbers with a PLAIN operand
//
// Placeholders take an index-bearing operand, axis numbers a plain one
// (subscripted internally, so there is ONE node protocol); the op declares
// identity<T>() and the fold order is unspecified.

#include "../Expr.h"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace tensor {

// clang-format off: annotations predate clang-format's parser
namespace ops {

// Folds only; the elementwise pair is math::Fmax/Fmin (C's NaN rule,
// where this op is a plain a < b).
struct [[=detail::sym("max"), =detail::atomic("InterlockedMax"),
         =detail::Selective{}]] Max {
    static constexpr auto operator()(auto a, auto b) { return a < b ? b : a; }
    template <typename T> static constexpr T identity() {
        return std::numeric_limits<T>::lowest();
    }
};
struct [[=detail::sym("min"), =detail::atomic("InterlockedMin"),
         =detail::Selective{}]] Min {
    static constexpr auto operator()(auto a, auto b) { return b < a ? b : a; }
    template <typename T> static constexpr T identity() {
        return std::numeric_limits<T>::max();
    }
};

// ── structured accumulators ─────────────────────────────────────────────────
// The carrier is the op's own state ⟨M, lift, merge, finish⟩, (M, merge, ε)
// a commutative monoid; declaring state<T> is what marks the family
// (detail::Structured). Precision is the state's own choice — Welford
// widens its fields (ACC-G9 moved inside M), the comparison ops do not
// (a comparison is exact). Structured folds demand in-range reads, like
// every non-Add fold.

// Both range ends in one pass. finish on an empty fiber: a NaN pair for
// floating point (ACC-L12); integral states keep the identity ends.
struct [[=detail::sym("minmax"), =detail::slang_body(R"(struct minmax_state { float lo; float hi; };

minmax_state minmax_identity() {
  minmax_state s; s.lo = 3.402823466e+38f; s.hi = -3.402823466e+38f; return s;
}

minmax_state minmax_lift(float x, uint c) {
  minmax_state s; s.lo = x; s.hi = x; return s;
}

minmax_state minmax_merge(minmax_state a, minmax_state b) {
  minmax_state s; s.lo = min(a.lo, b.lo); s.hi = max(a.hi, b.hi); return s;
}

minmax_state minmax_finish(minmax_state s) {
  if (s.hi < s.lo) { s.lo = 0.0f / 0.0f; s.hi = 0.0f / 0.0f; }
  return s;
}
)")]] MinMax {
    template <typename T> struct [[=detail::sym("minmax_state")]] state {
        T lo, hi;
    };
    template <typename T> static constexpr state<T> identity() {
        return {std::numeric_limits<T>::max(),
                std::numeric_limits<T>::lowest()};
    }
    template <typename T> static constexpr state<T> lift(T x) {
        return {x, x};
    }
    template <typename T>
    static constexpr state<T> merge(state<T> a, state<T> b) {
        return {b.lo < a.lo ? b.lo : a.lo, a.hi < b.hi ? b.hi : a.hi};
    }
    template <typename T> static constexpr state<T> finish(state<T> s) {
        if constexpr (std::numeric_limits<T>::has_quiet_NaN)
            if (s.hi < s.lo)
                return {std::numeric_limits<T>::quiet_NaN(),
                        std::numeric_limits<T>::quiet_NaN()};
        return s;
    }
};

// Mean AND variance in one pass (Chan's exact pairwise combination).
// SAMPLE variance, M2/(n-1) — Mathematica's and Julia's convention, NOT
// numpy's default — so n <= 1 gives NaN, which also answers the empty
// fiber (ACC-L12). Floating-point elements only: integer variance would
// truncate silently.
struct [[=detail::sym("welford"), =detail::slang_body(R"(struct welford_state { float mean; float m2; uint n; };
struct welford_result { float mean; float var; };

welford_state welford_identity() {
  welford_state s; s.mean = 0.0f; s.m2 = 0.0f; s.n = 0u; return s;
}

welford_state welford_lift(float x, uint c) {
  welford_state s; s.mean = x; s.m2 = 0.0f; s.n = 1u; return s;
}

welford_state welford_merge(welford_state a, welford_state b) {
  if (a.n == 0u) return b;
  if (b.n == 0u) return a;
  welford_state s;
  s.n = a.n + b.n;
  float d = b.mean - a.mean;
  float fn = float(s.n);
  s.mean = a.mean + d * (float(b.n) / fn);
  s.m2 = a.m2 + b.m2 + d * d * (float(a.n) * float(b.n) / fn);
  return s;
}

welford_result welford_finish(welford_state s) {
  welford_result r;
  r.mean = s.n == 0u ? 0.0f / 0.0f : s.mean;
  r.var = s.n <= 1u ? 0.0f / 0.0f : s.m2 / float(s.n - 1u);
  return r;
}
)")]] Welford {
    template <std::floating_point T> struct [[=detail::sym("welford_state")]] state {
        detail::accumulator_t<T> mean{}, m2{}; // widening INSIDE the state
        size_t n = 0;
    };
    template <typename T> struct [[=detail::sym("welford_result")]] result {
        T mean, var;
    };
    template <std::floating_point T> static constexpr state<T> identity() {
        return {};
    }
    template <std::floating_point T> static constexpr state<T> lift(T x) {
        return {x, {}, 1};
    }
    template <std::floating_point T>
    static constexpr state<T> merge(state<T> a, state<T> b) {
        if (a.n == 0)
            return b; // ε is a defining clause, not a formula limit
        if (b.n == 0)
            return a;
        using A = detail::accumulator_t<T>;
        const size_t n = a.n + b.n;
        const A d = b.mean - a.mean;
        return {a.mean + d * (A(b.n) / A(n)),
                a.m2 + b.m2 + d * d * (A(a.n) * A(b.n) / A(n)), n};
    }
    template <std::floating_point T>
    static constexpr result<T> finish(state<T> s) {
        constexpr T nan = std::numeric_limits<T>::quiet_NaN();
        return {s.n == 0 ? nan : T(s.mean),
                s.n <= 1 ? nan
                         : T(s.m2 / detail::accumulator_t<T>(s.n - 1))};
    }
};

// A value AND its location in one pass. lift observes the bound coordinate
// (the flat row-major position in the summed space, listed-id order); ties
// take the FIRST occurrence — a lexicographic max over (value, -index), so
// the answer is order-independent, not merely order-fixed. NaN loses, as
// in ops::Max.
struct [[=detail::sym("argmax"), =detail::slang_body(R"(struct argmax_state { float value; uint at; };

argmax_state argmax_identity() {
  argmax_state s; s.value = -3.402823466e+38f; s.at = 0xffffffffu; return s;
}

argmax_state argmax_lift(float x, uint c) {
  argmax_state s; s.value = x; s.at = c; return s;
}

argmax_state argmax_merge(argmax_state a, argmax_state b) {
  if (a.value < b.value) return b;
  if (b.value < a.value) return a;
  return b.at < a.at ? b : a;
}

argmax_state argmax_finish(argmax_state s) {
  if (s.at == 0xffffffffu) s.value = 0.0f / 0.0f;
  return s;
}
)")]] ArgMax {
    template <typename T> struct [[=detail::sym("argmax_state")]] state {
        T value;
        std::uint32_t at;
    };
    template <typename T> static constexpr state<T> identity() {
        return {std::numeric_limits<T>::lowest(),
                std::numeric_limits<std::uint32_t>::max()};
    }
    template <typename T> static constexpr state<T> lift(T x, size_t c) {
        return {x, std::uint32_t(c)};
    }
    template <typename T>
    static constexpr state<T> merge(state<T> a, state<T> b) {
        if (a.value < b.value)
            return b;
        if (b.value < a.value)
            return a;
        return b.at < a.at ? b : a;
    }
    template <typename T> static constexpr state<T> finish(state<T> s) {
        // The identity's index cannot be a real one (a fold is capped at
        // 2^32-1 elements), so it is the empty fiber's signature.
        if constexpr (std::numeric_limits<T>::has_quiet_NaN)
            if (s.at == std::numeric_limits<std::uint32_t>::max())
                return {std::numeric_limits<T>::quiet_NaN(), s.at};
        return s;
    }
};

struct [[=detail::sym("argmin"), =detail::slang_body(R"(struct argmin_state { float value; uint at; };

argmin_state argmin_identity() {
  argmin_state s; s.value = 3.402823466e+38f; s.at = 0xffffffffu; return s;
}

argmin_state argmin_lift(float x, uint c) {
  argmin_state s; s.value = x; s.at = c; return s;
}

argmin_state argmin_merge(argmin_state a, argmin_state b) {
  if (b.value < a.value) return b;
  if (a.value < b.value) return a;
  return b.at < a.at ? b : a;
}

argmin_state argmin_finish(argmin_state s) {
  if (s.at == 0xffffffffu) s.value = 0.0f / 0.0f;
  return s;
}
)")]] ArgMin {
    template <typename T> struct [[=detail::sym("argmin_state")]] state {
        T value;
        std::uint32_t at;
    };
    template <typename T> static constexpr state<T> identity() {
        return {std::numeric_limits<T>::max(),
                std::numeric_limits<std::uint32_t>::max()};
    }
    template <typename T> static constexpr state<T> lift(T x, size_t c) {
        return {x, std::uint32_t(c)};
    }
    template <typename T>
    static constexpr state<T> merge(state<T> a, state<T> b) {
        if (b.value < a.value)
            return b;
        if (a.value < b.value)
            return a;
        return b.at < a.at ? b : a;
    }
    template <typename T> static constexpr state<T> finish(state<T> s) {
        // The identity's index cannot be a real one (a fold is capped at
        // 2^32-1 elements), so it is the empty fiber's signature.
        if constexpr (std::numeric_limits<T>::has_quiet_NaN)
            if (s.at == std::numeric_limits<std::uint32_t>::max())
                return {std::numeric_limits<T>::quiet_NaN(), s.at};
        return s;
    }
};

// The folded ids are named `summed` after the Add default; walkers detect
// the protocol by this member, never by the op's name.
template <typename Op, size_t... Summed> struct [[=detail::sym("fold")]] Fold {
    using op = Op;
    static constexpr std::array<size_t, sizeof...(Summed)> summed{Summed...};
    // The node's element is finish∘lift of the summand's; the degenerate
    // case (M = R = T) returns it unchanged. Never executed — the type
    // oracle for Expr's invoke_result_t.
    static constexpr auto operator()(auto a) {
        if constexpr (detail::StructuredIndexed<Op, decltype(a)>)
            return Op::finish(Op::lift(a, size_t{}));
        else if constexpr (detail::Structured<Op, decltype(a)>)
            return Op::finish(Op::lift(a));
        else
            return a;
    }
};

} // namespace ops
// clang-format on

template <typename Op, auto... Ids, AnyExpr S> constexpr auto fold(S &&s) {
    return detail::fold_dispatch<Op, Ids...>(std::forward<S>(s));
}

template <auto... Ids, AnyExpr S> constexpr auto fold(S &&s) {
    return detail::fold_dispatch<ops::Add, Ids...>(std::forward<S>(s));
}

template <typename Op, auto... Ids, detail::RangedExpr S>
constexpr auto fold(S &&s) {
    return fold<Op, Ids...>(std::forward<S>(s).c);
}

template <auto... Ids, detail::RangedExpr S> constexpr auto fold(S &&s) {
    return fold<Ids...>(std::forward<S>(s).c);
}

} // namespace tensor
