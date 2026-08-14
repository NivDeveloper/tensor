#pragma once

// The value-level node protocol: pack structured bindings, never member
// names or template-argument positions.

#include "../Core.h"
#include "Diagnostics.h"
#include "Meta.h"
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
        sync_leaf_hosts(n.e);
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
    if constexpr ((index_bearing_v<std::remove_cvref_t<Cs>> || ...) &&
                  !ContractOp<Op>)
        return free_extents_of(children_scan<Cs...>());
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
                    if constexpr (constexpr int b = map_bare_slot(dm); b >= 0) {
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
    } else if constexpr (ContractNode<D>) {
        // The fold at this cell's environment (the epilogue path): listed
        // ids loop, last innermost — the strict chain, guards per read.
        using FoldOp = typename D::op_type;
        using Acc = typename D::type;
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
            acc = typename FoldOp::op{}(acc, eval_indexed(summand, env2));
        }
        return acc;
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
    } else if constexpr (ContractNode<D>) {
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

        using T = typename D::type;
        T acc = Fold::template identity<T>();
        contract_fold<0, D>(summand, env, acc);
        return acc;
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
            const auto s = children_scan<Cs...>();
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
        if constexpr ((size_t(is_ix_pad_v<std::remove_cvref_t<Sub>>) + ... +
                       size_t{0}) > 1)
            static_assert(false, pad_count_error());
        using T = std::remove_cvref_t<typename Operand::type>;
        T fill{};
        ([&] {
            if constexpr (is_ix_pad_v<std::remove_cvref_t<Sub>>)
                fill = T(sub.value);
        }(),
         ...);
        return Indexed<Operand, term_map<Sub>()...>{o, fill};
    }
}

} // namespace tensor::detail

// The fold op is defined by the surface header that owns it.
namespace tensor::ops {
template <typename Op, size_t... Summed> struct Fold;
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
            return Indexed<C, lift_lin(unit_lin(K))...>{std::move(c)};
        }(std::make_index_sequence<C::rank>{});
        return fold_core<Op, Axes...>(std::move(sub));
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

} // namespace tensor::detail
