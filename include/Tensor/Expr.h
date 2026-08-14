#pragma once

// Building and evaluating lazy expressions — the declaration surface:
//
//   TensorView<T, Es…>   non-owning terminal node over external storage
//   Expr<Op, Cs…>        THE node — unary, binary, n-ary are all this type
//   operators            - + ~ ! and + - * / % & | ^ << >>
//                                    == != < > <= >= && ||
//   map<f>(cs…)          any plain function (any arity)
//   fold<Op, Ids…>(s)    THE fold: fold<j>(A[i,j] * x[j]), fold<ops::Max>(e)
//   indices              the subscript vocabulary i…n, 2_c, wrap/clamp/zero
//   for_each_leaf(e, f)  runtime DFS over an expression's leaves
//
// Definitions live in Expr/*.h, included at the bottom.

#include "Core.h"
#include "detail/Expr.h"

#include <cstddef>
#include <mdspan>
#include <meta>
#include <tuple>
#include <type_traits>

namespace tensor {

// ── the symbolic index vocabulary ───────────────────────────────────────────
// Subscript shapes travel in the TYPE (parameters are never constant
// expressions) — hence the _c literal (A[0_c, i], 2_c * i).

// A placeholder. tensor::indices predeclares i…n as Ix<0>…Ix<5>.
template <size_t N> struct Ix {
    static_assert(N < detail::index_slots, "index placeholder beyond Ix<7>");
    static constexpr size_t id = N;
};

// An affine combination of placeholders (i + j - k, 2_c * i, 1_c).
template <detail::Lin L> struct IxForm {
    static constexpr detail::Lin lin = L;
};

// A decorated index: an affine form plus a per-read boundary pipeline —
// wrap(i + 1_c), clamp(j), zero(i - 1_c) — chains resolve innermost-first.
template <detail::DecMap M> struct IxDec {
    static constexpr detail::DecMap map = M;
};

// A decorated form carrying the value a missed read yields.
template <detail::DecMap M, typename V> struct IxPad {
    static constexpr detail::DecMap map = M;
    V value;
};

// Only + - and constant scaling: a non-affine index is unspellable.
template <detail::IndexTerm A, detail::IndexTerm B>
constexpr auto operator+(A, B);
template <detail::IndexTerm A, detail::IndexTerm B>
constexpr auto operator-(A, B);
template <detail::IndexTerm A> constexpr auto operator-(A);
template <detail::IndexTerm A, detail::IndexTerm B>
constexpr auto operator*(A, B);

// A decorated form displaces further (wrap(i+1_c) - 1_c); it cannot be
// scaled or negated — those would act on the resolved coordinate.
template <detail::DecoratedTerm A, detail::IndexTerm B>
constexpr auto operator+(A, B);
template <detail::IndexTerm A, detail::DecoratedTerm B>
constexpr auto operator+(A, B);
template <detail::DecoratedTerm A, detail::IndexTerm B>
constexpr auto operator-(A, B);

// using namespace tensor::indices → the placeholders i…n, the _c literal,
// and the boundary decorations.
namespace indices {

inline constexpr Ix<0> i{};
inline constexpr Ix<1> j{};
inline constexpr Ix<2> k{};
inline constexpr Ix<3> l{};
inline constexpr Ix<4> m{};
inline constexpr Ix<5> n{};

template <char... Cs> constexpr auto operator""_c();

// How this read resolves an out-of-range coordinate: periodic, edge-held,
// absent (the element type's zero), or absent with a value of your own —
// pad's value is an ordinary runtime scalar, so a wall temperature or a
// fold's identity can be a parameter. One pad per read.
template <detail::SubscriptTerm T> consteval auto wrap(T);
template <detail::SubscriptTerm T> consteval auto clamp(T);
template <detail::SubscriptTerm T> consteval auto zero(T);
template <detail::SubscriptTerm T, typename V> constexpr auto pad(T, V v);

} // namespace indices

// ── the op vocabulary ───────────────────────────────────────────────────────
// The closed set, forward-declared: the names usable as template arguments
// (fold<ops::Max, j>(…), fold<ops::Add, 0>(…)). Definitions — with their
// [[=detail::sym("…")]] annotations — live in the Expr/*.h files.
namespace ops {

struct Neg;    // -
struct Pos;    // +
struct BitNot; // ~
struct Not;    // !

struct Add;    // +
struct Sub;    // -
struct Mul;    // *
struct Div;    // /
struct Mod;    // %
struct BitAnd; // &
struct BitOr;  // |
struct BitXor; // ^
struct Shl;    // <<
struct Shr;    // >>
struct Eq;     // ==
struct Ne;     // !=
struct Lt;     // <
struct Gt;     // >
struct Le;     // <=
struct Ge;     // >=
struct And;    // &&
struct Or;     // ||

struct Max; // folds only — the elementwise pair is math::Fmax/Fmin
struct Min;

template <typename Op, size_t... Summed> struct Fold;
template <std::meta::info F> struct Fn;

} // namespace ops

// ── the nodes ───────────────────────────────────────────────────────────────

template <typename T, size_t... Extents> struct TensorView {
    using type = T;
    using extents_type = std::extents<size_t, Extents...>;
    static constexpr size_t rank = sizeof...(Extents);

    T *data;
    // The owning Tensor's residency state; null for a raw view.
    detail::ShadowSlot *shadow = nullptr;

    constexpr T operator[](Index<rank> idx) const;
    constexpr T operator[](size_t flat) const;

    // Symbolic subscript, v[wrap(i + 1_c)]: an indexed read for the expression grammar.
    template <detail::SubscriptArg... Sub>
        requires(detail::SubscriptTerm<Sub> || ...)
    constexpr auto operator[](Sub... sub) const;
};

// THE node — every arity is sizeof...(Cs), children stored by value; all
// behaviour is generic over "children + op_type".
template <typename Op, typename... Cs> struct Expr {
    using op_type = Op;

    detail::Kids<Cs...> children;

    // Every op keeps its shape carrier's extents; a fold decides its own,
    // and an index-bearing node's shape is its free-index space.
    using extents_type = typename[:detail::node_extents<Op, Cs...>():];
    static constexpr size_t rank = extents_type::rank();
    using type = std::invoke_result_t<Op, detail::element_t<Cs>...>;

    constexpr type operator[](Index<rank> idx) const;
    constexpr type operator[](size_t flat) const;

    // Symbolic subscript, (1.0 + f)[j]: elementwise work fuses into the
    // enclosing indexed expression.
    template <detail::SubscriptArg... Sub>
        requires(detail::SubscriptTerm<Sub> || ...)
    constexpr auto operator[](Sub... sub) const;
};

// DFS over the leaves; an indexed leaf is transparent.
template <typename Node, typename F>
constexpr void for_each_leaf(const Node &n, F &&f);

// ── the elementwise operators ───────────────────────────────────────────────
// auto-generic, so validity follows the element types; scalars broadcast.

template <typename E>
    requires Operands<E>
constexpr auto operator-(E &&e);
template <typename E>
    requires Operands<E>
constexpr auto operator+(E &&e);
template <typename E>
    requires Operands<E>
constexpr auto operator~(E &&e);
template <typename E>
    requires Operands<E>
constexpr auto operator!(E &&e);

template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator+(L &&l, R &&r);
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator-(L &&l, R &&r);
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator*(L &&l, R &&r);
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator/(L &&l, R &&r);
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator%(L &&l, R &&r);
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator&(L &&l, R &&r);
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator|(L &&l, R &&r);
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator^(L &&l, R &&r);
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator<<(L &&l, R &&r);
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator>>(L &&l, R &&r);
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator==(L &&l, R &&r);
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator!=(L &&l, R &&r);
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator<(L &&l, R &&r);
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator>(L &&l, R &&r);
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator<=(L &&l, R &&r);
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator>=(L &&l, R &&r);
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator&&(L &&l, R &&r);
template <typename L, typename R>
    requires Operands<L, R>
constexpr auto operator||(L &&l, R &&r);

// ── the builders ────────────────────────────────────────────────────────────

// map<f>: lift a plain function of any arity — map<halve>(t). The math
// vocabulary is spelt directly (math::Sqrt(t)), never through map.
template <auto &F, typename... Cs>
    requires Operands<Cs...>
constexpr auto map(Cs &&...cs);

// THE fold: consume listed indices with a reducible op (Add default) —
// fold<j>(A[i,j] * x[j]) is the contraction, fold<ops::Max, j>(…) the
// op-generic form. Placeholders take an index-bearing operand, axis
// numbers a plain one; none folds everything. One fold per tree, with
// only elementwise ops above it (the epilogue).
template <typename Op, auto... Ids, AnyExpr S> constexpr auto fold(S &&s);
template <auto... Ids, AnyExpr S> constexpr auto fold(S &&s);

} // namespace tensor

// The definitions (see the index above).
#include "Expr/Node.h"     // IWYU pragma: export
#include "Expr/Index.h"    // IWYU pragma: export
#include "Expr/Unary.h"    // IWYU pragma: export
#include "Expr/Binary.h"   // IWYU pragma: export
#include "Expr/Map.h"      // IWYU pragma: export
#include "Expr/Fold.h"     // IWYU pragma: export

// The ops lint sweeps the complete ops:: namespace — keep it last.
#include "detail/OpsCheck.h"
