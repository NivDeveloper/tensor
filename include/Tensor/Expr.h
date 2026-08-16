#pragma once

// Building and evaluating lazy expressions — the declaration surface:
//
//   TensorView<T, Es…>   non-owning terminal node over external storage
//   Expr<Op, Cs…>        THE node — unary, binary, n-ary are all this type
//   operators            - + ~ ! and + - * / % & | ^ << >>
//                                    == != < > <= >= && ||
//   map<f>(cs…)          any plain function (any arity)
//   fold<Op, Ids…>(s)    THE fold: fold<j>(A[i,j] * x[j]), fold<ops::Max>(e)
//   scan<Op, Id>(s)      THE scan: the running op along one index, kept
//   indices              the subscript vocabulary i…n, 2_c, wrap/clamp/zero
//   for_each_leaf(e, f)  runtime DFS over an expression's leaves
//
// Definitions live in Expr/*.h, included at the bottom.

#include "Core.h"
#include "detail/Expr.h"

#include <cstddef>
#include <cstdint>
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

// A gathered coordinate: the policy is compile-time, the coordinate is an
// expression evaluated per output cell. IxDataPad adds pad's fill.
template <detail::Policy P, typename C> struct IxData {
    static constexpr detail::DecMap map = detail::data_map(P);
    C c;
};
template <detail::Policy P, typename C, typename V> struct IxDataPad {
    static constexpr detail::DecMap map = detail::data_map(P);
    C c;
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

// A GATHERED coordinate takes no index arithmetic. There is no affine form
// to displace — the coordinate is a value — so the shift belongs in the
// expression, where it is ordinary elementwise work.
template <detail::DataTerm A, detail::IndexTerm B>
constexpr auto operator+(A, B) =
    delete("a data subscript takes no index arithmetic — put the shift in "
           "the expression: clamp(cell[i] - 1), not clamp(cell[i]) - 1_c");
template <detail::IndexTerm A, detail::DataTerm B>
constexpr auto operator+(A, B) =
    delete("a data subscript takes no index arithmetic — put the shift in "
           "the expression: clamp(cell[i] - 1), not clamp(cell[i]) - 1_c");
template <detail::DataTerm A, detail::IndexTerm B>
constexpr auto operator-(A, B) =
    delete("a data subscript takes no index arithmetic — put the shift in "
           "the expression: clamp(cell[i] - 1), not clamp(cell[i]) - 1_c");

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
template <detail::AffineTerm T> consteval auto wrap(T);
template <detail::AffineTerm T> consteval auto clamp(T);
template <detail::AffineTerm T> consteval auto zero(T);
template <detail::AffineTerm T, typename V> constexpr auto pad(T, V v);

// The same four on a GATHERED coordinate — a subscript computed at run time
// rather than an affine form: mu[clamp(cell[i])]. The coordinate is any
// arithmetic expression; a floating-point one is floored before the policy
// applies. Its own free indices join the tree's, and the axis it subscripts
// is consumed.
template <IndexedExpr C> constexpr auto wrap(C &&c);
template <IndexedExpr C> constexpr auto clamp(C &&c);
template <IndexedExpr C> constexpr auto zero(C &&c);
template <IndexedExpr C, typename V> constexpr auto pad(C &&c, V v);

// The same question on the WRITE side, for a scatter destination. A read
// resolves against the operand's own axis; a scatter writes into an axis of
// nothing, so the extent is named — wrap<8>(cell[i]). drop is zero's
// counterpart: it removes the contribution rather than substituting a
// value, which is every op's identity, so it is legal under any of them.
template <size_t E, IndexedExpr C> constexpr auto wrap(C &&c);
template <size_t E, IndexedExpr C> constexpr auto clamp(C &&c);
template <size_t E, IndexedExpr C> constexpr auto drop(C &&c);

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
template <typename Op, size_t Id> struct Scan;
template <typename Op, size_t... Summed> struct Scatter;
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

// A storage-free terminal: its value at a coordinate is a closed-form
// function of that coordinate, so it costs a parameter or two and no
// buffer, upload or ABI slot. Shaped exactly like TensorView, so every
// walker that only reads leaves treats it as one. Spell them through
// Gen.h — fill, iota, linspace — never by naming this type.
template <detail::GenKind K, typename T, size_t... Extents> struct Generator {
    using type = T;
    using extents_type = std::extents<size_t, Extents...>;
    static constexpr size_t rank = sizeof...(Extents);
    static constexpr detail::GenKind kind = K;
    static constexpr size_t count = (size_t{1} * ... * Extents);

    T a{}, b{};            // the deterministic kinds' parameters
    std::uint64_t key = 0; // a sampler's stream: the seed folded with it

    constexpr T operator[](Index<rank> idx) const;
    constexpr T operator[](size_t flat) const;
};

// The SHAPELESS generator — no extents_type, so it is not a TensorExpr and
// pins no shape, broadcasting like a scalar while still varying per cell.
// Only the samplers spell it (noise has no extent of its own the way a ramp
// does); it reads the coordinate of whatever cell is being computed, which
// is why permuting the output order permutes which cell draws what.
template <detail::GenKind K, typename T> struct Generator<K, T> {
    using type = T;
    static constexpr detail::GenKind kind = K;

    T a{}, b{};
    std::uint64_t key = 0;

    constexpr T operator[](size_t flat) const;
};

// THE node — every arity is sizeof...(Cs), children stored by value; all
// behaviour is generic over "children + op_type".
template <typename Op, typename... Cs> struct Expr {
    using op_type = Op;

    detail::Kids<Cs...> children;

    // A shapeless sampler broadcasts, so some operand in its node has to
    // say how many cells there are; scalars cannot, they broadcast too.
    static_assert(!(detail::is_shapeless_generator_v<Cs> || ...) ||
                      ((TensorExpr<Cs> || IndexedExpr<Cs>) || ...),
                  detail::gen_unpinned_error());

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

// THE scan: a running op along ONE index, which it KEEPS — scan<m>(h[j,m])
// is the running sum whose last entry is fold<m>'s answer. fold consumes an
// index, scatter places one; a scan preserves it, so the result has the
// operand's shape. Evaluation order is SPECIFIED (ascending, one
// accumulator per row), so serial and threaded agree bit for bit. Exclusive
// needs no builder: read the result at pad(m - 1_c, identity).
template <typename Op, auto Id, AnyExpr S> constexpr auto scan(S &&s);
template <auto Id, AnyExpr S> constexpr auto scan(S &&s);

// THE scatter: the same fold, into a cell chosen by DATA rather than by a
// free index — scatter<i>(wrap<C>(cell[i]), q[i]) deposits each particle's
// q into its own cell. Every argument but the last is a destination and
// carries a write policy naming its extent; the last is the value. Ids are
// consumed exactly as fold consumes them, and any that survive stay axes of
// the result, after the destinations.
template <typename Op, auto... Ids, typename... Cs>
constexpr auto scatter(Cs &&...cs);
template <auto... Ids, typename... Cs> constexpr auto scatter(Cs &&...cs);

} // namespace tensor

// The definitions (see the index above).
#include "Expr/Node.h"     // IWYU pragma: export
#include "Expr/Index.h"    // IWYU pragma: export
#include "Expr/Unary.h"    // IWYU pragma: export
#include "Expr/Binary.h"   // IWYU pragma: export
#include "Expr/Map.h"      // IWYU pragma: export
#include "Expr/Fold.h"     // IWYU pragma: export
#include "Expr/Scan.h"     // IWYU pragma: export
#include "Expr/Scatter.h"  // IWYU pragma: export

// The ops lint sweeps the complete ops:: namespace — keep it last.
#include "detail/OpsCheck.h"
