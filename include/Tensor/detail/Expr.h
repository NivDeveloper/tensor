#pragma once

// The value-level node protocol: pack structured bindings, never member
// names or template-argument positions.

#include "../Core.h"
#include "Diagnostics.h"
#include "Meta.h"
#include "Pool.h"
#include "Tree.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <mdspan>
#include <meta>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensor::detail {

// Tensors join as views, everything else stores by value. Building an
// expression is a READ: the const view, so every tensor leaf is
// TensorView<const T> and the const and non-const spellings of one
// expression are one type — one formula, one kernel.
template <typename E> constexpr auto as_child(E &&e) {
    using D = std::remove_cvref_t<E>;
    if constexpr (is_specialization_of(^^D, ^^Tensor))
        return std::as_const(e).view();
    else
        return D(std::forward<E>(e));
}

// The first TensorExpr child donates rank and extents.
template <typename... Cs> consteval size_t shape_carrier_index() {
    constexpr bool is_expr[] = {TensorExpr<Cs>...};
    for (size_t i = 0; i < sizeof...(Cs); ++i)
        if (is_expr[i])
            return i;
    return 0;
}

template <typename... Cs>
using shape_carrier_t =
    std::remove_cvref_t<Cs...[shape_carrier_index<Cs...>()]>;

// ── the node protocols, value level ─────────────────────────────────────────
// Type-level twin in Tree.h: is_contraction (the fold protocol).

template <typename D>
concept ExprNode = requires { typename D::op_type; };

// Residency: bring every leaf's HOST copy current before a CPU walk reads
// it — eval and the element subscripts call this; device eval binds
// resident buffers instead. The const_cast is sound: a slot-bearing view
// only ever aliases a Tensor's own mutable array.
template <typename Node> constexpr void sync_leaf_hosts(const Node &n) {
    using D = std::remove_cvref_t<Node>;
    if constexpr (ExprNode<D>) {
        const auto &[... children] = n;
        (sync_leaf_hosts(children), ...);
    } else if constexpr (is_indexed_v<D>) {
        // The coordinates too: a gather's index tensor is as much a leaf as
        // the operand, and a stale one reads garbage silently.
        for_each_slot(n.d, [](const auto &c) { sync_leaf_hosts(c); });
        sync_leaf_hosts(n.e);
    } else if constexpr (std::is_same_v<D, NoCoord>) {
        // an affine axis carries no coordinate
    } else if constexpr (is_placed_v<D>) {
        sync_leaf_hosts(n.c);
    } else if constexpr (requires { n.shadow; }) {
        if (n.shadow && n.shadow->storage && !n.shadow->host_valid) {
            constexpr size_t bytes = [] {
                size_t count = 1;
                for (size_t r = 0; r < D::rank; ++r)
                    count *= D::extents_type::static_extent(r);
                return count * sizeof(typename D::type);
            }();
            n.shadow->storage->download(
                const_cast<void *>(static_cast<const void *>(n.data)), bytes);
            n.shadow->host_valid = true;
        }
    } else if constexpr (requires { n.view(); }) {
        sync_leaf_hosts(n.view()); // a bare Tensor as the whole expression
    }
}


// The fold: consumes its indexed summand's `summed` placeholders with
// its `op`; the survivors stay visible.
template <typename Op>
concept ContractOp = requires {
    Op::summed;
    typename Op::op;
};

template <typename D>
concept ContractNode = ExprNode<D> && ContractOp<typename D::op_type>;

// The scatter: the same protocol with the destination marker. It IS a
// contraction, so every whole-tree rule covers it; what differs is that its
// output cell comes from data, so it can never be computed per output cell
// and every per-cell walker has to route around it.
template <typename D>
concept ScatterNode = ExprNode<D> && ScatterOp<typename D::op_type>;

template <typename D>
concept FoldOnlyNode = ContractNode<D> && !ScatterNode<D>;

// What a reduction accumulates IN (ACC-G9). A float32 chain loses three of
// seven significant digits by 4e6 terms through absorption; accumulating in
// float64 costs nothing measurable and buys four orders. It changes the
// PRECISION of each step, never the ORDER, so a reduction is still exactly
// what ACC-G8 predicts and every path still agrees with every other
// (ACC-G4) — which holds only while all three widen together.
template <typename T> struct accumulator {
    using type = T;
};
template <> struct accumulator<float> {
    using type = double;
};
template <typename T> struct accumulator<std::complex<T>> {
    using type = std::complex<typename accumulator<T>::type>;
};
template <typename T> using accumulator_t = typename accumulator<T>::type;

// ... but only where rounding ACCUMULATES. A Selective op (ops::Max, Min)
// returns an operand unchanged, so a wider accumulator cannot change its
// answer — and it is not free: 6.6% on a cache-resident float max fold,
// nothing measurable once memory-bound.
consteval bool op_is_selective(std::meta::info op) {
    return !std::meta::annotations_of_with_type(op, ^^Selective).empty();
}
template <typename Op, typename T>
using fold_accumulator_t =
    std::conditional_t<op_is_selective(^^Op), T, accumulator_t<T>>;

// Declaring identity<T>() is the op's assertion that it is associative and
// commutative; reduction order is unspecified.
template <typename Op, typename T>
concept Reducible = requires { Op::template identity<T>(); };

// Declaring absorber<T>() is the op's assertion that the value annihilates
// it from either side — what lets a guarded zero absorb a product term.
template <typename Op, typename T>
concept Absorbing = requires { Op::template absorber<T>(); };

// Declaring zero_stable is the op's assertion that op(0, …, 0) is 0 — what
// lets a guard EVERY operand carries rise past the op.
template <typename Op>
concept ZeroStable = requires { requires Op::zero_stable; };

// Nested, not &&: the absorber call must not be instantiated for an op
// that declares none.
template <typename Op, typename T> consteval bool absorbs_zero() {
    if constexpr (Absorbing<Op, T>)
        return Op::template absorber<T>() == T{};
    else
        return false;
}

// A scatter's axes are its DESTINATIONS, in argument order, then whatever
// ids survive it — the one shape in the library that is part positional and
// part free-index.
template <typename Op, typename... Cs>
consteval std::meta::info scatter_extents() {
    std::vector<std::meta::info> args{^^size_t};
    [&]<size_t... K>(std::index_sequence<K...>) {
        (args.push_back(std::meta::reflect_constant(Cs...[K]::ext)), ...);
    }(std::make_index_sequence<sizeof...(Cs) - 1>{});
    const auto s = drop_summed(
        children_census<Cs...>(),
        std::vector<size_t>(Op::summed.begin(), Op::summed.end()));
    for (size_t id : s.order)
        args.push_back(std::meta::reflect_constant(s.pinned(id)));
    return std::meta::substitute(^^std::extents, args);
}

// The carrier's extents — except a fold binds a fresh index space.
template <typename Op, typename Carrier>
consteval std::meta::info result_extents_of() {
    if constexpr (ContractOp<Op>) {
        const std::vector<size_t> summed(Op::summed.begin(), Op::summed.end());
        return contracted_extents(^^Carrier, summed);
    } else if constexpr (requires { typename Carrier::extents_type; }) {
        return ^^typename Carrier::extents_type;
    } else {
        return ^^std::extents<size_t>;
    }
}

// The hook Expr's extents_type splices: an index-bearing node's extents
// are its free-index space (first-appearance order across the children);
// everything else keeps result_extents_of's answer.
template <typename Op, typename... Cs>
consteval std::meta::info node_extents() {
    if constexpr (ScatterOp<Op>)
        return scatter_extents<Op, std::remove_cvref_t<Cs>...>();
    // Nothing here carries a shape (a shapeless sampler among scalars):
    // rank 0 so instantiation reaches Expr's assert, which names the fix.
    else if constexpr (!((TensorExpr<Cs> || IndexedExpr<Cs>) || ...))
        return ^^std::extents<size_t>;
    else if constexpr ((index_bearing_v<std::remove_cvref_t<Cs>> || ...) &&
                       !ContractOp<Op>)
        return free_extents_of(children_census<Cs...>());
    else
        return result_extents_of<Op, shape_carrier_t<Cs...>>();
}

// ── evaluation ──────────────────────────────────────────────────────────────

// A flat index decomposes row-major; Index<rank> passes through.
template <typename Extents, typename At> constexpr auto to_multi_index(At at) {
    if constexpr (std::is_integral_v<At>) {
        Index<Extents::rank()> idx{};
        for (size_t k = Extents::rank(); k-- > 0;) {
            idx[k] = at % Extents::static_extent(k);
            at /= Extents::static_extent(k);
        }
        return idx;
    } else {
        return at;
    }
}

template <typename Node, typename At>
constexpr auto eval_node(const Node &n, At at);

// The summand walker: each indexed leaf turns the placeholder values into
// a multi-index; out of range yields zero WITHOUT evaluating the operand.
using IxEnv = std::array<std::ptrdiff_t, index_slots>;

// A guarded affine read: the map plus the axis extent it must stay inside.
struct GuardMap {
    Lin lin{};
    size_t ext = 0;
};
// One entry per map at its smallest extent (which implies the larger ones);
// a full set drops the rest — an uncollected guard only stays a runtime one.
struct GuardSet {
    std::array<GuardMap, 2 * index_slots> m{};
    size_t n = 0;
    consteval void add(GuardMap g) {
        for (size_t i = 0; i < n; ++i)
            if (m[i].lin == g.lin) {
                m[i].ext = g.ext < m[i].ext ? g.ext : m[i].ext;
                return;
            }
        if (n < m.size())
            m[n++] = g;
    }
    consteval void append(const GuardSet &o) {
        for (size_t i = 0; i < o.n; ++i)
            add(o.m[i]);
    }
    // Guards both sides carry, at the WIDER extent: the node is zero only
    // outside every operand's range, so the tighter one does not hold.
    consteval void intersect(const GuardSet &o) {
        GuardSet r;
        for (size_t i = 0; i < n; ++i)
            for (size_t q = 0; q < o.n; ++q)
                if (m[i].lin == o.m[q].lin)
                    r.m[r.n++] = {m[i].lin,
                                  m[i].ext > o.m[q].ext ? m[i].ext : o.m[q].ext};
        *this = r;
    }
};

// Does the set pin this map inside [0, ext) already?
consteval bool covered(const GuardSet &g, Lin m, size_t ext) {
    for (size_t i = 0; i < g.n; ++i)
        if (g.m[i].ext <= ext && g.m[i].lin == m)
            return true;
    return false;
}

// A gathered coordinate as a signed index. A floating-point one is FLOORED,
// and saturated first: converting an out-of-range float is UB, and inside
// consteval a hard "not a constant expression". Saturation is invisible for
// any coordinate that is not already absurd, and for one that is, every
// policy answers edge-or-absent — which is right.
inline constexpr double coord_saturate = 1.0e9;

template <typename V> constexpr std::ptrdiff_t coord_value(V v) {
    if constexpr (std::is_floating_point_v<V>) {
        const double d = double(v);
        const double f = d < -coord_saturate  ? -coord_saturate
                         : d > coord_saturate ? coord_saturate
                                              : d;
        return std::ptrdiff_t(f < 0 && f != std::ptrdiff_t(f)
                                  ? std::ptrdiff_t(f) - 1
                                  : std::ptrdiff_t(f));
    } else {
        return std::ptrdiff_t(v);
    }
}

template <GuardSet Proven = GuardSet{}, typename Node>
constexpr auto eval_indexed(const Node &n, const IxEnv &env) {
    using D = std::remove_cvref_t<Node>;
    if constexpr (is_broadcast_scalar_v<D>) {
        return n;
    } else if constexpr (is_indexed_v<D>) {
        // unrolled so zero coefficients and bare-slot bounds checks vanish;
        // a displaced axis the loop clamps already prove skips its check
        using E = std::remove_cvref_t<typename D::operand_type>;
        constexpr size_t r = E::rank;
        Index<r> at{};
        bool in = true;
        [&]<size_t... K>(std::index_sequence<K...>) {
            (
                [&] {
                    constexpr DecMap dm = D::maps[K];
                    constexpr size_t ext = E::extents_type::static_extent(K);
                    if constexpr (map_data(dm)) {
                        // A gathered axis: the coordinate is a value, so it
                        // is evaluated here and then run through the same
                        // policy tail the chain arm uses. An empty Proven —
                        // the coordinate's reads are unrelated to whatever
                        // the enclosing loop clamped.
                        std::ptrdiff_t v = coord_value(
                            eval_indexed<GuardSet{}>(slot_get<K>(n.d), env));
                        if constexpr (dm.s[0].pol == Policy::Wrap) {
                            v %= std::ptrdiff_t(ext);
                            if (v < 0)
                                v += std::ptrdiff_t(ext);
                            at[K] = size_t(v);
                        } else if constexpr (dm.s[0].pol == Policy::Clamp) {
                            at[K] = size_t(v < 0 ? 0
                                           : v >= std::ptrdiff_t(ext)
                                               ? std::ptrdiff_t(ext) - 1
                                               : v);
                        } else if (v < 0 || v >= std::ptrdiff_t(ext)) {
                            in = false; // None/Zero/Pad guard
                        } else {
                            at[K] = size_t(v);
                        }
                    } else if constexpr (constexpr int b = map_bare_slot(dm);
                                         b >= 0) {
                        at[K] = size_t(env[size_t(b)]);
                    } else if constexpr (map_affine(dm)) {
                        constexpr Lin mk = dm.s[0].lin;
                        std::ptrdiff_t v = mk.off;
                        [&]<size_t... P>(std::index_sequence<P...>) {
                            (
                                [&] {
                                    if constexpr (mk.c[P] != 0)
                                        v += mk.c[P] * env[P];
                                }(),
                                ...);
                        }(std::make_index_sequence<index_slots>{});
                        if constexpr (covered(Proven, mk, ext)) {
                            at[K] = size_t(v);
                        } else if (v < 0 || v >= std::ptrdiff_t(ext)) {
                            in = false;
                        } else {
                            at[K] = size_t(v);
                        }
                    } else {
                        // The decoration chain, stages innermost-first: each
                        // adds its affine part, then applies its policy —
                        // None/Zero guard, Wrap/Clamp transform.
                        std::ptrdiff_t v = 0;
                        bool ok = true;
                        [&]<size_t... St>(std::index_sequence<St...>) {
                            (
                                [&] {
                                    if constexpr (St < dm.n) {
                                        constexpr Lin sl = dm.s[St].lin;
                                        v += sl.off;
                                        [&]<size_t... P>(
                                            std::index_sequence<P...>) {
                                            (
                                                [&] {
                                                    if constexpr (sl.c[P] != 0)
                                                        v += sl.c[P] * env[P];
                                                }(),
                                                ...);
                                        }(std::make_index_sequence<
                                            index_slots>{});
                                        if constexpr (dm.s[St].pol ==
                                                      Policy::Wrap) {
                                            v %= std::ptrdiff_t(ext);
                                            if (v < 0)
                                                v += std::ptrdiff_t(ext);
                                        } else if constexpr (dm.s[St].pol ==
                                                             Policy::Clamp) {
                                            v = v < 0 ? 0
                                                : v >= std::ptrdiff_t(ext)
                                                    ? std::ptrdiff_t(ext) - 1
                                                    : v;
                                        } else if (v < 0 ||
                                                   v >= std::ptrdiff_t(ext)) {
                                            ok = false; // None/Zero guard
                                        }
                                    }
                                }(),
                                ...);
                        }(std::make_index_sequence<max_stages>{});
                        if (!ok)
                            in = false;
                        else
                            at[K] = size_t(v);
                    }
                }(),
                ...);
        }(std::make_index_sequence<r>{});
        if (!in)
            return n.fill;
        return typename D::type(eval_node(n.e, at));
    } else if constexpr (is_placed_v<D>) {
        return eval_indexed<Proven>(n.c, env); // the raw coordinate
    } else if constexpr (FoldOnlyNode<D>) {
        // The fold at this cell's environment (the epilogue path): listed
        // ids loop, last innermost — the strict chain, guards per read.
        using FoldOp = typename D::op_type;
        using Acc = fold_accumulator_t<typename FoldOp::op,
                                       std::remove_cvref_t<typename D::type>>;
        const auto &[summand] = n;
        using S = std::remove_cvref_t<decltype(summand)>;
        static constexpr auto plan = contract_plan(
            std::meta::dealias(^^S),
            std::vector<size_t>(FoldOp::summed.begin(),
                                FoldOp::summed.end()));
        Acc acc = FoldOp::op::template identity<Acc>();
        IxEnv env2 = env;
        for (size_t c = 0; c < plan.fold_count; ++c) {
            size_t rem = c;
            for (size_t d = plan.n_summed; d-- > 0;) {
                env2[plan.summed_id[d]] =
                    std::ptrdiff_t(rem % plan.summed_ext[d]);
                rem /= plan.summed_ext[d];
            }
            acc = typename FoldOp::op{}(acc, Acc(eval_indexed(summand, env2)));
        }
        return std::remove_cvref_t<typename D::type>(acc);
    } else {
        const auto &[... children] = n;
        return typename D::op_type{}(eval_indexed<Proven>(children, env)...);
    }
}

// ── the shift interior ──────────────────────────────────────────────────────
// Inside the interior box (Tree.h's ShiftPlan) every leaf sits at one
// constant flat offset from the output index, so the read is in range by
// construction and no policy fires: the body becomes contiguous loads.

template <typename D> consteval std::ptrdiff_t shift_delta() {
    using E = std::remove_cvref_t<typename D::operand_type>;
    std::ptrdiff_t delta = 0, stride = 1;
    for (size_t k = E::rank; k-- > 0;) {
        delta += std::ptrdiff_t(D::maps[k].s[0].lin.off) * stride;
        stride *= std::ptrdiff_t(E::extents_type::static_extent(k));
    }
    return delta;
}

template <typename Node>
constexpr auto eval_shifted(const Node &n, std::ptrdiff_t q) {
    using D = std::remove_cvref_t<Node>;
    if constexpr (is_broadcast_scalar_v<D>) {
        return n;
    } else if constexpr (is_indexed_v<D>) {
        return typename D::type(eval_node(n.e, size_t(q + shift_delta<D>())));
    } else {
        const auto &[... children] = n;
        return typename D::op_type{}(eval_shifted(children, q)...);
    }
}

// ── the contraction loop nest ───────────────────────────────────────────────
// A guarded affine read on the summand's multiplicative spine yields the
// absorbing zero out of range, so its whole term is the fold identity there
// — the loop skips the range instead (term-absence for products).

// Guarded maps reachable from the summand root through absorbing ops only.
// T is the contraction's element type, a proxy for each node's own — fine
// while every absorber is T{}.
template <typename S, typename T> struct SpineGuardsOf {
    static consteval GuardSet value() {
        GuardSet g;
        if constexpr (is_indexed_v<S>) {
            using E = std::remove_cvref_t<typename S::operand_type>;
            // Only a single-stage read zeroing on a miss has the guard a
            // loop bound can absorb; wrap/clamp chains contribute none, and
            // a padded read misses to a value that does not vanish.
            for (size_t k = 0; k < E::rank; ++k)
                if (map_affine(S::maps[k]) && !S::padded &&
                    bare_slot(S::maps[k].s[0].lin) < 0)
                    g.add({S::maps[k].s[0].lin,
                           E::extents_type::static_extent(k)});
        }
        return g;
    }
};
template <typename Op, typename... Cs, typename T>
struct SpineGuardsOf<Expr<Op, Cs...>, T> {
    static consteval GuardSet value() {
        GuardSet g;
        if constexpr (absorbs_zero<Op, T>()) {
            // Any zeroed operand zeroes the node: every child's guards hold.
            (g.append(SpineGuardsOf<std::remove_cvref_t<Cs>, T>::value()), ...);
        } else if constexpr (ZeroStable<Op>) {
            // Only a guard EVERY operand carries zeroes the node.
            bool first = true;
            (
                [&] {
                    const GuardSet c =
                        SpineGuardsOf<std::remove_cvref_t<Cs>, T>::value();
                    if (first)
                        g = c, first = false;
                    else
                        g.intersect(c);
                }(),
                ...);
        }
        return g;
    }
};

// The spine guards usable as bounds at summed level q: a coefficient on that
// placeholder, every other one already bound (free, or summed earlier).
template <typename D, typename S> consteval GuardSet level_guards(size_t q) {
    using Op = typename D::op_type;
    using Fold = typename Op::op;
    using T = typename D::type;
    if (!(Fold::template identity<T>() == T{}))
        return {}; // skipping is a fold no-op only when the identity is zero
    const auto plan = contract_plan(
        ^^S, std::vector<size_t>(Op::summed.begin(), Op::summed.end()));
    const GuardSet all = SpineGuardsOf<S, T>::value();
    const size_t id = plan.summed_id[q];
    GuardSet out;
    for (size_t i = 0; i < all.n; ++i) {
        const Lin m = all.m[i].lin;
        if (m.c[id] == 0)
            continue;
        bool bound = true;
        for (size_t p = 0; p < index_slots && bound; ++p) {
            if (p == id || m.c[p] == 0)
                continue;
            bool b = false;
            for (size_t f = 0; f < plan.n_free; ++f)
                b |= plan.free_id[f] == p;
            for (size_t s = 0; s < q; ++s)
                b |= plan.summed_id[s] == p;
            bound = b;
        }
        if (bound)
            out.add(all.m[i]);
    }
    return out;
}

constexpr std::ptrdiff_t div_floor(std::ptrdiff_t a, std::ptrdiff_t b) {
    return a >= 0 ? a / b : -((-a + b - 1) / b); // b > 0
}
constexpr std::ptrdiff_t div_ceil(std::ptrdiff_t a, std::ptrdiff_t b) {
    return a > 0 ? (a + b - 1) / b : -(-a / b); // b > 0
}

// Every level's clamp holds inside the innermost body: their union is what
// eval_indexed may read unguarded. Unions exactly level_guards' outputs, so
// nothing is ever proven that was not actually clamped.
template <typename D, typename S> consteval GuardSet proven_guards() {
    using Op = typename D::op_type;
    const auto plan = contract_plan(
        ^^S, std::vector<size_t>(Op::summed.begin(), Op::summed.end()));
    GuardSet g;
    for (size_t q = 0; q < plan.n_summed; ++q)
        g.append(level_guards<D, S>(q));
    return g;
}

// One loop per summed placeholder (listed order, last innermost), each
// clamped to where its spine guards can be nonzero; the surviving terms
// keep the exact order of the flat fold this replaces.
template <size_t Q, typename D, typename S, typename T>
constexpr void contract_fold(const S &summand, IxEnv &env, T &acc) {
    using Op = typename D::op_type;
    using Fold = typename Op::op;
    static constexpr auto plan = contract_plan(
        ^^S, std::vector<size_t>(Op::summed.begin(), Op::summed.end()));
    if constexpr (Q == plan.n_summed) {
        acc = Fold{}(acc, eval_indexed<proven_guards<D, S>()>(summand, env));
    } else {
        static constexpr auto g = level_guards<D, S>(Q);
        constexpr size_t id = plan.summed_id[Q];
        std::ptrdiff_t lo = 0, hi = std::ptrdiff_t(plan.summed_ext[Q]);
        [&]<size_t... C>(std::index_sequence<C...>) {
            (
                [&] {
                    constexpr Lin m = g.m[C].lin;
                    constexpr auto ext = std::ptrdiff_t(g.m[C].ext);
                    constexpr std::ptrdiff_t c = m.c[id];
                    std::ptrdiff_t rest = m.off;
                    [&]<size_t... P>(std::index_sequence<P...>) {
                        (
                            [&] {
                                if constexpr (P != id && m.c[P] != 0)
                                    rest += m.c[P] * env[P];
                            }(),
                            ...);
                    }(std::make_index_sequence<index_slots>{});
                    if constexpr (c > 0) {
                        lo = std::max(lo, div_ceil(-rest, c));
                        hi = std::min(hi, div_floor(ext - 1 - rest, c) + 1);
                    } else {
                        lo = std::max(lo, div_ceil(rest - ext + 1, -c));
                        hi = std::min(hi, div_floor(rest, -c) + 1);
                    }
                }(),
                ...);
        }(std::make_index_sequence<g.n>{});
        for (std::ptrdiff_t v = lo; v < hi; ++v) {
            env[id] = v;
            contract_fold<Q + 1, D>(summand, env, acc);
        }
    }
}

// One evaluator for every node shape; `At` is flat or an Index<rank>.
template <typename Node, typename At>
constexpr auto eval_node(const Node &n, At at) {
    using D = std::remove_cvref_t<Node>;
    if constexpr (is_broadcast_scalar_v<D>) {
        return n;
    } else if constexpr (is_placed_v<D>) {
        return eval_node(n.c, at);
    } else if constexpr (FoldOnlyNode<D>) {
        // free coordinates from the output index; summed ones iterated by
        // contract_fold's clamped loop nest
        using Op = typename D::op_type;
        using Fold = typename Op::op;
        const auto &[summand] = n; // contractions are unary
        using S = std::remove_cvref_t<decltype(summand)>;
        static constexpr auto plan = contract_plan(
            ^^S, std::vector<size_t>(Op::summed.begin(), Op::summed.end()));

        auto out = to_multi_index<typename D::extents_type>(at);
        IxEnv env{};
        for (size_t q = 0; q < plan.n_free; ++q)
            env[plan.free_id[q]] = std::ptrdiff_t(out[q]);

        using T = std::remove_cvref_t<typename D::type>;
        using Acc = fold_accumulator_t<Fold, T>;
        Acc acc = Fold::template identity<Acc>();
        contract_fold<0, D>(summand, env, acc);
        return T(acc);
    } else if constexpr (ExprNode<D>) {
        const auto &[... children] = n;
        return typename D::op_type{}(eval_node(children, at)...);
    } else {
        return n[at];
    }
}

// ── the interchanged contraction ────────────────────────────────────────────
// A bare summand (all_maps_bare — nothing can read out of range) needs no
// clamps, so the summed loops move OUTSIDE the free ones: operands stream
// instead of column-walking per output cell, and each cell still sees its
// summed tuples in the identical order. from/to bound the FIRST FREE
// level — the summed levels above it forward them, deeper free levels
// take the full range — so a parallel chunk owns disjoint output rows.
template <size_t Q, typename D, typename S, typename T>
constexpr void contract_streamed(const S &summand, IxEnv &env, T *out,
                                 size_t oat, size_t from, size_t to) {
    using Op = typename D::op_type;
    using Fold = typename Op::op;
    static constexpr auto plan = contract_plan(
        ^^S, std::vector<size_t>(Op::summed.begin(), Op::summed.end()));
    if constexpr (Q == plan.n_summed + plan.n_free) {
        out[oat] = Fold{}(out[oat], eval_indexed(summand, env));
    } else if constexpr (Q < plan.n_summed) {
        constexpr size_t id = plan.summed_id[Q];
        for (size_t v = 0; v < plan.summed_ext[Q]; ++v) {
            env[id] = std::ptrdiff_t(v);
            contract_streamed<Q + 1, D>(summand, env, out, oat, from, to);
        }
    } else {
        constexpr size_t f = Q - plan.n_summed;
        constexpr size_t id = plan.free_id[f];
        constexpr size_t n = D::extents_type::static_extent(f);
        const size_t lo = f == 0 ? from : 0, hi = f == 0 ? to : n;
        for (size_t v = lo; v < hi; ++v) {
            env[id] = std::ptrdiff_t(v);
            contract_streamed<Q + 1, D>(summand, env, out, oat * n + v,
                                        from, to);
        }
    }
}

// The scalar type an expression produces per element.
template <typename E>
using element_t = decltype(eval_node(std::declval<const E &>(), size_t{}));

// ── shape checks ────────────────────────────────────────────────────────────

template <typename Carrier, typename C> consteval bool shape_matches() {
    if constexpr (TensorExpr<C>)
        return std::is_same_v<typename std::remove_cvref_t<C>::extents_type,
                              typename Carrier::extents_type>;
    else
        return true;
}

template <typename... Cs> consteval bool shapes_compatible() {
    return (shape_matches<shape_carrier_t<Cs...>, Cs>() && ...);
}

// ── builders ────────────────────────────────────────────────────────────────

// The node factory behind every operator and map<f>.
template <typename Op, typename... Cs>
    requires Operands<Cs...>
constexpr auto make_expr(Cs &&...cs) {
    // Every check below fires through a DISCARDED branch: a static_assert's
    // message is constant-evaluated even when the assertion holds, and these
    // messages walk the operands and build strings. make_expr runs once per
    // node, so an eagerly built message costs the whole tree, per node.
    if constexpr ((TensorExpr<Cs> || ...) && (IndexedExpr<Cs> || ...))
        static_assert(false, mixed_operand_error());
    if constexpr (!shapes_compatible<Cs...>())
        static_assert(false, shape_error({shape_of<Cs>()...}));
    // Index-bearing operands combine by index identity; what can disagree
    // is an id's pinned extent, caught at the point of combination.
    static constexpr auto ib_clash = [] { // {id, a, b}
        if constexpr ((IndexedExpr<Cs> || ...)) {
            const auto s = children_census<Cs...>();
            return std::array{s.clash_id, s.clash_a, s.clash_b};
        } else
            return std::array{index_slots, size_t{0}, size_t{0}};
    }();
    if constexpr (ib_clash[0] != index_slots)
        static_assert(false, contract_extent_error(ib_clash[0], ib_clash[1],
                                                   ib_clash[2]));
    // One fold per tree, elementwise above it: the epilogue is computed per
    // output cell before the store — still a single pass.
    if constexpr ((fold_count_v<std::remove_cvref_t<Cs>> + ... +
                   size_t{0}) > 1)
        static_assert(false, two_folds_error());
    return Expr<Op, decltype(as_child(std::forward<Cs>(cs)))...>{
        {{as_child(std::forward<Cs>(cs))}...}};
}

// A subscript's coordinate slot: the expression a gathered axis carries,
// NoCoord for an affine one. One slot per axis, so the axis index is the
// slot index.
template <typename S> consteval auto coord_probe() {
    using D = std::remove_cvref_t<S>;
    if constexpr (is_ix_data_v<D>)
        return std::type_identity<
            std::remove_cvref_t<decltype(std::declval<D>().c)>>{};
    else if constexpr (has_free_index_v<D>)
        return std::type_identity<D>{}; // undecorated: reaches the lint
    else
        return std::type_identity<NoCoord>{};
}
template <typename S> using coord_t = typename decltype(coord_probe<S>())::type;

template <typename S> constexpr auto coord_of(const S &s) {
    using D = std::remove_cvref_t<S>;
    if constexpr (is_ix_data_v<D>)
        return s.c;
    else if constexpr (has_free_index_v<D>)
        return s;
    else
        return NoCoord{};
}

// The shared body of every symbolic subscript: lint, then bind the operand
// to one affine map per axis.
template <typename Operand, typename Extents, typename... Sub>
constexpr auto make_indexed(const Operand &o, Sub... sub) {
    // Discarded branches again: this runs once per subscripted read, and an
    // eagerly built message would cost the whole extent pack per read.
    if constexpr (sizeof...(Sub) != Extents::rank())
        static_assert(false, subscript_arity_error(sizeof...(Sub), ^^Extents));
    static_assert((SubscriptTerm<Sub> && ...),
                  "an integer subscript in an indexed read must be a "
                  "compile-time constant — spell 0 as 0_c (tensor::indices)");
    if constexpr (sizeof...(Sub) == Extents::rank() &&
                  (SubscriptTerm<Sub> && ...)) {
        if constexpr (!subscript_consts_ok<Extents, term_map<Sub>()...>())
            static_assert(false,
                          subscript_const_error<Extents, term_map<Sub>()...>());
        if constexpr (!subscript_named<Extents, term_map<Sub>()...>())
            static_assert(false,
                          boundary_error<Extents, term_map<Sub>()...>());
        if constexpr ((size_t(map_padded(term_map<Sub>())) + ... + size_t{0}) >
                      1)
            static_assert(false, pad_count_error());
        // ACC-L4, the read side: a DATA coordinate must be able to name
        // every cell of the axis it indexes. The extent comes from the
        // operand here rather than from the decoration, which is the only
        // difference from the write policies' check.
        static constexpr size_t narrow = [] {
            size_t axis = 0, bad = Extents::rank();
            (
                [&] {
                    if constexpr (map_data(term_map<Sub>()))
                        if (bad == Extents::rank() &&
                            Extents::static_extent(axis) >
                                index_capacity<std::remove_cvref_t<
                                    typename coord_t<Sub>::type>>())
                            bad = axis;
                    ++axis;
                }(),
                ...);
            return bad;
        }();
        if constexpr (narrow != Extents::rank()) {
            using CT = std::remove_cvref_t<
                typename coord_t<Sub...[narrow]>::type>;
            static_assert(false,
                          index_capacity_error(^^CT,
                                               Extents::static_extent(narrow),
                                               index_capacity<CT>()));
        }
        using T = std::remove_cvref_t<typename Operand::type>;
        T fill{};
        ([&] {
            if constexpr (requires { sub.value; })
                fill = T(sub.value);
        }(),
         ...);
        return Indexed<Operand, Kids<coord_t<Sub>...>, term_map<Sub>()...>{
            o, {{coord_of(sub)}...}, fill};
    }
}

} // namespace tensor::detail

// The fold op is defined by the surface header that owns it.
namespace tensor::ops {
template <typename Op, size_t... Summed> struct Fold;
template <typename Op, size_t Id> struct Scan;
template <typename Op, size_t... Summed> struct Scatter;
struct Add;
} // namespace tensor::ops

namespace tensor::detail {

// ── the fold builders' shared body ──────────────────────────────────────────
// The lint-light core; the entry points below add the user-facing lints.
template <typename Op, size_t... Ids, typename S>
constexpr auto fold_core(S &&s) {
    return make_expr<ops::Fold<Op, Ids...>>(std::forward<S>(s));
}

// A read a non-Add fold can trust: resolved in range by construction.
consteval bool map_in_range(DecMap m) {
    if (m.s[m.n - 1].pol == Policy::Wrap || m.s[m.n - 1].pol == Policy::Clamp)
        return true;
    if (map_padded(m))
        return true; // the author named what a miss yields
    if (map_bare(m))
        return true;
    return map_affine(m) && lin_const(m.s[0].lin); // range-linted upstream
}

consteval bool all_maps_in_range(std::meta::info summand) {
    std::vector<std::vector<DecMap>> leaves;
    indexed_maps_under(summand, leaves);
    for (const auto &ms : leaves)
        for (const auto &m : ms)
            if (!map_in_range(m))
                return false;
    return true;
}

// Placeholder ids over an index-bearing summand.
template <typename Op, size_t... Ids, typename S>
constexpr auto fold_ids_impl(S &&s) {
    using D = std::remove_cvref_t<S>;
    using T = std::remove_cvref_t<typename D::type>;
    static_assert(Reducible<Op, T>,
                  "fold: this op declares no identity<T>(), so it cannot be "
                  "a fold — only associative, commutative ops are reducible "
                  "(Expr/Binary.h)");
    static_assert(sizeof...(Ids) > 0, contract_no_sum_error());
    if constexpr (fold_count_v<D> != 0)
        static_assert(false, fold_in_summand_error());
    // A scan is not a FoldOp, so fold_count_v does not see it — but it is
    // just as much a second pass, and eval would have to materialize it.
    if constexpr (scan_count_v<D> != 0)
        static_assert(false, scan_in_tree_error());
    if constexpr (sizeof...(Ids) > 0) {
        static_assert(axes_distinct({Ids...}),
                      contract_duplicate_error(first_duplicate({Ids...})));
        static constexpr auto unpinned =
            first_unpinned(^^D, {
                                    Ids...});
        static_assert(unpinned == index_slots,
                      contract_unpinned_error(unpinned));
        static_assert(!contraction_degenerate(^^D), fold_degenerate_error());
        static_assert(!contraction_separable(^^D,
                                             {
                                                 Ids...}),
                      contract_separable_error());
        static_assert(std::is_same_v<Op, ops::Add> || all_maps_in_range(^^D),
                      fold_range_error());
        return fold_core<Op, Ids...>(std::forward<S>(s));
    }
}

// Axis numbers over a plain operand: subscript internally with the fresh
// placeholders Ix<0..r-1>, so there is one node protocol. The identity
// maps are exempt from the user-spelling lints by construction.
template <typename Op, size_t... Axes, typename S>
constexpr auto fold_axes_impl(S &&s) {
    auto c = as_child(std::forward<S>(s));
    using C = std::remove_cvref_t<decltype(c)>;
    using T = std::remove_cvref_t<typename C::type>;
    static_assert(Reducible<Op, T>,
                  "fold: this op declares no identity<T>(), so it cannot be "
                  "a fold — only associative, commutative ops are reducible "
                  "(Expr/Binary.h)");
    // The placeholder form checks this too: a reduction inside a reduction
    // is a second pass, whichever way the ids are spelt.
    if constexpr (fold_count_v<C> != 0)
        static_assert(false, fold_in_summand_error());
    if constexpr (scan_count_v<C> != 0)
        static_assert(false, scan_in_tree_error());
    static_assert(((Axes < C::rank) && ...),
                  reduce_axis_error(
                      [] {
                          size_t bad = 0;
                          for (size_t a : {Axes...})
                              if (a >= C::rank)
                                  bad = a;
                          return bad;
                      }(),
                      ^^typename C::extents_type));
    static_assert(axes_distinct({Axes...}),
                  reduce_duplicate_error(first_duplicate({Axes...})));
    if constexpr (((Axes < C::rank) && ...)) {
        auto sub = [&]<size_t... K>(std::index_sequence<K...>) {
            return Indexed<C, Kids<decltype((void)K, NoCoord{})...>,
                           lift_lin(unit_lin(K))...>{std::move(c)};
        }(std::make_index_sequence<C::rank>{});
        return fold_core<Op, Axes...>(std::move(sub));
    }
}

// ── the scatter ─────────────────────────────────────────────────────────────
// Its children are [dest…, value]: every argument but the last carries a
// write policy, which is what splits them unambiguously.

// The type-level spelling of Tree.h's reflective plan: one answer, shared by
// the runtime loops and the emitter.
template <typename D> consteval ScatterLayout scatter_layout() {
    return scatter_layout(std::meta::dealias(^^D));
}

// Grains fix the association at COMPILE TIME — consumed extent and bin size
// both live in the type — so serial and parallel agree bit for bit at any
// thread count. Past the budget the grain count falls, never the guarantee.
inline constexpr size_t scatter_bin_budget = 1u << 16;
inline constexpr size_t scatter_grain_target = 8;

consteval size_t scatter_grains(size_t consumed, size_t bin) {
    if (consumed == 0 || bin == 0)
        return 1;
    const size_t cap = scatter_bin_budget / bin;
    if (cap == 0)
        return 1;
    const size_t g = consumed < scatter_grain_target ? consumed
                                                     : scatter_grain_target;
    return g < cap ? g : cap;
}

// One grain: consumed indices [lo, hi), every surviving cell, in order.
// Children arrive as an ordinary pack — a structured-binding pack cannot be
// captured by the nested generic lambda the destinations need (P1061).
template <typename D, typename T, typename... Ks>
constexpr void scatter_grain(T *bin, size_t lo, size_t hi, const Ks &...ks) {
    using Reduce = typename D::op_type::op;
    static constexpr auto l = scatter_layout<D>();
    IxEnv env{};
    for (size_t c = lo; c < hi; ++c) {
        for (size_t d = l.n_sum, rem = c; d-- > 0; rem /= l.sum_ext[d])
            env[l.sum_id[d]] = std::ptrdiff_t(rem % l.sum_ext[d]);
        for (size_t s = 0; s < l.surv_size; ++s) {
            for (size_t d = l.n_surv, rem = s; d-- > 0; rem /= l.surv_ext[d])
                env[l.surv_id[d]] = std::ptrdiff_t(rem % l.surv_ext[d]);
            size_t dest = 0;
            bool keep = true;
            [&]<size_t... K>(std::index_sequence<K...>) {
                (
                    [&] {
                        if (!keep)
                            return;
                        using P = Ks...[K];
                        size_t r = 0;
                        // coord_value, not a plain cast: a float coordinate
                        // FLOORS, the same rule a gathered subscript follows.
                        if (place_coord<P::policy, P::ext>(
                                coord_value(eval_indexed(ks...[K].c, env)), r))
                            dest = dest * P::ext + r;
                        else
                            keep = false; // drop: the contribution is absent
                    }(),
                    ...);
            }(std::make_index_sequence<l.n_dest>{});
            if (!keep)
                continue;
            T &cell = bin[dest * l.surv_size + s];
            cell = Reduce{}(cell, T(eval_indexed(ks...[l.n_dest], env)));
        }
    }
}

// The whole scatter into `acc`. A merge in grain order, never a per-cell
// walk: cell c alone would mean scanning every contribution to ask which
// ones land on it.
template <typename D, typename T, typename Node>
constexpr void eval_scatter(const Node &n, T *acc) {
    using Reduce = typename D::op_type::op;
    static constexpr auto l = scatter_layout<D>();
    static constexpr size_t cells = l.dest_size * l.surv_size;
    static constexpr size_t grains = scatter_grains(l.sum_size, cells);
    constexpr T id = Reduce::template identity<T>();

    const auto &[... kids] = n;
    if constexpr (grains == 1) {
        std::fill_n(acc, cells, id);
        scatter_grain<D>(acc, size_t{0}, l.sum_size, kids...);
    } else {
        std::vector<T> bins(grains * cells, id);
        const size_t span = (l.sum_size + grains - 1) / grains;
        // Each grain owns its bin, so the writes are disjoint and the merge
        // below still runs in grain order — the serial association.
        const size_t per = (span + 1) * l.surv_size;
        parallel_for(
            grains,
            [&](size_t from, size_t to) {
                for (size_t g = from; g < to; ++g) {
                    const size_t lo = g * span;
                    if (lo < l.sum_size)
                        scatter_grain<D>(bins.data() + g * cells, lo,
                                         std::min(lo + span, l.sum_size),
                                         kids...);
                }
            },
            std::max(size_t{1}, eval_grain / std::max(per, size_t{1})));
        for (size_t c = 0; c < cells; ++c) {
            T a = id;
            for (size_t g = 0; g < grains; ++g)
                a = Reduce{}(a, bins[g * cells + c]);
            acc[c] = a;
        }
    }
}

// The scatter node inside a tree that has exactly one.
template <size_t K = 0, typename Node>
constexpr const auto &scatter_node_of(const Node &n) {
    using D = std::remove_cvref_t<Node>;
    if constexpr (ScatterNode<D>) {
        return n;
    } else {
        const auto &[... kids] = n;
        using C = std::remove_cvref_t<decltype(kids...[K])>;
        if constexpr (scatter_count_v<C> == 1)
            return scatter_node_of(kids...[K]);
        else
            return scatter_node_of<K + 1>(n);
    }
}

// The epilogue, applied where every output cell is already visited once:
// reaching the scatter yields its finished cell instead of recursing.
template <typename Node, typename T>
constexpr auto eval_epilogue(const Node &n, size_t at, const T *acc) {
    using D = std::remove_cvref_t<Node>;
    if constexpr (ScatterNode<D>)
        return acc[at];
    else if constexpr (scatter_count_v<D> == 0)
        return eval_node(n, at);
    else {
        const auto &[... kids] = n;
        return typename D::op_type{}(eval_epilogue(kids, at, acc)...);
    }
}

template <typename Op, size_t... Ids, typename... Cs>
constexpr auto scatter_core(Cs &&...cs) {
    return make_expr<ops::Scatter<Op, Ids...>>(std::forward<Cs>(cs)...);
}

template <typename Op, auto... Ids, typename... Cs>
constexpr auto scatter_dispatch(Cs &&...cs) {
    constexpr size_t n = sizeof...(Cs);
    static_assert(n >= 2, scatter_arity_error());
    static_assert(sizeof...(Ids) > 0 &&
                      (is_placeholder_v<std::remove_cvref_t<decltype(Ids)>> &&
                       ...),
                  scatter_ids_error());
    if constexpr (n >= 2 && sizeof...(Ids) > 0 &&
                  (is_placeholder_v<std::remove_cvref_t<decltype(Ids)>> &&
                   ...)) {
        constexpr bool split = []<size_t... K>(std::index_sequence<K...>) {
            return (is_placed_v<std::remove_cvref_t<Cs...[K]>> && ...) &&
                   !is_placed_v<std::remove_cvref_t<Cs...[n - 1]>>;
        }(std::make_index_sequence<n - 1>{});
        static_assert(split, scatter_placed_error());
        if constexpr (split) {
            using T = element_t<std::remove_cvref_t<Cs...[n - 1]>>;
            static_assert(Reducible<Op, T>,
                          "scatter: this op declares no identity<T>(), so it "
                          "cannot reduce — only associative, commutative ops "
                          "are reducible (Expr/Binary.h)");
            static_assert(
                axes_distinct({std::remove_cvref_t<decltype(Ids)>::id...}),
                contract_duplicate_error(first_duplicate(
                    {std::remove_cvref_t<decltype(Ids)>::id...})));
            if constexpr ((fold_count_v<std::remove_cvref_t<Cs>> + ... +
                           size_t{0}) != 0)
                static_assert(false, fold_in_summand_error());
            if constexpr ((scan_count_v<std::remove_cvref_t<Cs>> + ... +
                           size_t{0}) != 0)
                static_assert(false, scan_in_tree_error());
            return scatter_core<Op, std::remove_cvref_t<decltype(Ids)>::id...>(
                std::forward<Cs>(cs)...);
        }
    }
}

// The H3 discrimination: placeholders take an index-bearing operand, axis
// numbers a plain one, none folds everything.
template <typename Op, auto... Ids, typename S>
constexpr auto fold_dispatch(S &&s) {
    using D = std::remove_cvref_t<S>;
    constexpr bool ph =
        (is_placeholder_v<std::remove_cvref_t<decltype(Ids)>> && ...);
    constexpr bool ax =
        (std::integral<std::remove_cvref_t<decltype(Ids)>> && ...);
    static_assert(ph || ax, fold_mixed_ids_error());
    if constexpr (sizeof...(Ids) == 0) {
        if constexpr (index_bearing_v<D>) {
            static constexpr auto p = free_plan(std::meta::dealias(^^D));
            return [&]<size_t... A>(std::index_sequence<A...>) {
                return fold_ids_impl<Op, p.id[A]...>(std::forward<S>(s));
            }(std::make_index_sequence<p.n>{});
        } else {
            return [&]<size_t... K>(std::index_sequence<K...>) {
                return fold_axes_impl<Op, K...>(std::forward<S>(s));
            }(std::make_index_sequence<D::rank>{});
        }
    } else if constexpr (ph) {
        static_assert(index_bearing_v<D>,
                      "fold: placeholders name an index-bearing operand's "
                      "free indices — subscript the operand, or name axis "
                      "numbers instead");
        if constexpr (index_bearing_v<D>)
            return fold_ids_impl<Op,
                                 std::remove_cvref_t<decltype(Ids)>::id...>(
                std::forward<S>(s));
    } else {
        static_assert(!index_bearing_v<D>,
                      "fold: axis numbers take a PLAIN operand — an "
                      "index-bearing one names placeholders");
        if constexpr (!index_bearing_v<D>)
            return fold_axes_impl<Op, size_t(Ids)...>(std::forward<S>(s));
    }
}

// ── the scan ────────────────────────────────────────────────────────────────
// The prefix: a running Op along ONE index, which it KEEPS. Everything the
// shape layer needs follows from ops::Scan carrying no `summed` — see
// ScanOp (detail/Core.h).

template <typename D>
concept ScanNode = ExprNode<D> && ScanOp<typename D::op_type>;

// The scan node inside a tree that has exactly one.
template <size_t K = 0, typename Node>
constexpr const auto &scan_node_of(const Node &n) {
    using D = std::remove_cvref_t<Node>;
    if constexpr (ScanNode<D>) {
        return n;
    } else {
        const auto &[... kids] = n;
        using C = std::remove_cvref_t<decltype(kids...[K])>;
        if constexpr (scan_count_v<C> == 1)
            return scan_node_of(kids...[K]);
        else
            return scan_node_of<K + 1>(n);
    }
}

// The epilogue above a scan. Unlike the scatter's, it is driven by the
// INDEX ENVIRONMENT rather than a flat cell: the ops above a scan are
// elementwise over the free-index space, so their own leaves are read at
// the coordinate the accumulator was produced at. Reaching the scan yields
// the running value instead of recursing into it.
template <typename Node, typename T>
constexpr auto eval_scan_epilogue(const Node &n, const IxEnv &env, T acc) {
    using D = std::remove_cvref_t<Node>;
    if constexpr (ScanNode<D>)
        return acc;
    else if constexpr (scan_count_v<D> == 0)
        return eval_indexed(n, env);
    else {
        const auto &[... kids] = n;
        return typename D::op_type{}(eval_scan_epilogue(kids, env, acc)...);
    }
}

template <typename Op, size_t Id, typename C> constexpr auto scan_core(C &&c) {
    return make_expr<ops::Scan<Op, Id>>(std::forward<C>(c));
}

template <typename Op, size_t Id, typename S>
constexpr auto scan_ids_impl(S &&s) {
    using D = std::remove_cvref_t<S>;
    using T = std::remove_cvref_t<typename D::type>;
    static_assert(Reducible<Op, T>,
                  "scan: this op declares no identity<T>(), so it cannot "
                  "accumulate — only associative ops are scannable "
                  "(Expr/Binary.h)");
    // One structural node per tree: a scan beside a fold, a scatter or
    // another scan is a second pass over the same data.
    if constexpr (fold_count_v<D> != 0 || scan_count_v<D> != 0)
        static_assert(false, scan_in_tree_error());
    static_assert((free_ids_v<D> >> Id) & 1u, scan_id_not_free_error(Id));
    if constexpr (((free_ids_v<D> >> Id) & 1u) && fold_count_v<D> == 0 &&
                  scan_count_v<D> == 0)
        return scan_core<Op, Id>(std::forward<S>(s));
}

// An axis number on a plain operand: subscript every axis, then scan the
// placeholder that axis became — one node protocol, as fold does it.
template <typename Op, size_t Axis, typename S>
constexpr auto scan_axes_impl(S &&s) {
    auto c = as_child(std::forward<S>(s));
    using C = std::remove_cvref_t<decltype(c)>;
    static_assert(Axis < C::rank,
                  reduce_axis_error(Axis, ^^typename C::extents_type));
    if constexpr (Axis < C::rank) {
        auto sub = [&]<size_t... K>(std::index_sequence<K...>) {
            return Indexed<C, Kids<decltype((void)K, NoCoord{})...>,
                           lift_lin(unit_lin(K))...>{std::move(c)};
        }(std::make_index_sequence<C::rank>{});
        return scan_ids_impl<Op, Axis>(std::move(sub));
    }
}

// Placeholders take an index-bearing operand, an axis number a plain one —
// the discrimination fold makes, with exactly one id either way.
template <typename Op, auto Id, typename S> constexpr auto scan_dispatch(S &&s) {
    using D = std::remove_cvref_t<S>;
    using I = std::remove_cvref_t<decltype(Id)>;
    constexpr bool ph = is_placeholder_v<I>;
    constexpr bool ax = std::integral<I>;
    static_assert(ph || ax, scan_id_error());
    if constexpr (ph) {
        static_assert(index_bearing_v<D>,
                      "scan: a placeholder names an index-bearing operand's "
                      "free index — subscript the operand, or name an axis "
                      "number instead");
        if constexpr (index_bearing_v<D>)
            return scan_ids_impl<Op, I::id>(std::forward<S>(s));
    } else if constexpr (ax) {
        static_assert(!index_bearing_v<D>,
                      "scan: an axis number takes a PLAIN operand — an "
                      "index-bearing one names a placeholder");
        if constexpr (!index_bearing_v<D>)
            return scan_axes_impl<Op, size_t(Id)>(std::forward<S>(s));
    }
}

} // namespace tensor::detail
