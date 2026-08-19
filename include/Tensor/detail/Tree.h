#pragma once

// The expression-tree protocol at the TYPE level — the reflective mirror of
// the value-level one (detail/Expr.h): op_of / children_of / symbol_of, the
// three op-protocol tests, and render(t, style, idx, …), where dialects are
// LeafStyle data, not new walkers.

#include "../Core.h"
#include "Meta.h"
#include "Sym.h"

#include <algorithm> // std::ranges::contains
#include <complex>
#include <cstddef>
#include <mdspan> // std::extents — reduced_extents substitutes into it
#include <meta>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace tensor::detail {

consteval std::meta::info op_of(std::meta::info node) {
    return alias_of(node, "op_type");
}

consteval std::vector<std::meta::info> children_of(std::meta::info node) {
    std::vector<std::meta::info> out;
    for (auto m : std::meta::nonstatic_data_members_of(
             node, std::meta::access_context::current())) {
        auto t = std::meta::dealias(std::meta::type_of(m));
        if (is_specialization_of(t, ^^SlotsImpl)) {
            auto args = std::meta::template_arguments_of(t);
            for (size_t q = 1; q < args.size(); ++q) // [0] is the index seq
                out.push_back(args[q]);
        }
        else
            out.push_back(t);
    }
    return out;
}

consteval std::string_view symbol_of(std::meta::info op) {
    auto anns = std::meta::annotations_of_with_type(op, ^^Symbol);
    return std::meta::extract<Symbol>(std::meta::constant_of(anns[0])).str;
}

// A mapped callable's identifier, or "fn" when it reflects nameless.
consteval Symbol fn_symbol(std::meta::info f) {
    return sym(std::meta::has_identifier(f) ? std::meta::identifier_of(f)
                                            : "fn");
}

// A sym-annotated type is a vocabulary op: map binds it as the node's op
// directly. is_class_type filters function types, which have no annotations.
consteval bool carries_symbol(std::meta::info type) {
    auto t = std::meta::dealias(type);
    return std::meta::is_class_type(t) &&
           !std::meta::annotations_of_with_type(t, ^^Symbol).empty();
}

// Twin of is_broadcast_scalar_v (detail/Core.h). Not "is_scalar_type" —
// std::meta has one, and ADL on info makes the unqualified call ambiguous.
consteval bool is_broadcast_scalar_type(std::meta::info type) {
    auto t = std::meta::dealias(type);
    return std::meta::is_arithmetic_type(t) ||
           (is_specialization_of(t, ^^std::complex) &&
            std::meta::is_arithmetic_type(
                std::meta::template_arguments_of(t)[0]));
}

// ── the fold protocol, type level ───────────────────────────────────────────
// Keyed on the static member `summed` — never on the op's name. Args:
// [Op, Summed…].
consteval bool is_contraction(std::meta::info op) {
    return std::meta::has_template_arguments(op) &&
           has_static_member(op, "summed");
}
consteval bool is_fold(std::meta::info op) { return is_contraction(op); }

// The scatter is a contraction plus the destination marker — so the tests
// above answer yes for it, and this is the only place the two part ways.
consteval bool is_placed_op(std::meta::info op) {
    return is_contraction(op) && has_static_member(op, "placed");
}

// The scan protocol: `scanned`, and no `summed` — it keeps its index. Every
// test above therefore answers NO for it, which is what lets the shape and
// index layers treat a scan node as an ordinary elementwise one.
consteval bool is_scan_op(std::meta::info op) {
    return std::meta::has_template_arguments(op) &&
           has_static_member(op, "scanned");
}
consteval size_t scanned_of(std::meta::info op) {
    return args_of<size_t>(op, 1)[0]; // ops::Scan's args are [Op, Id]
}

// A scatter destination: Placed<Policy, Extent, Coord>, its coordinate the
// one data member.
consteval bool is_placed(std::meta::info t) {
    return is_specialization_of(std::meta::dealias(t), ^^Placed);
}
consteval std::meta::info placed_coord_of(std::meta::info t) {
    return std::meta::dealias(children_of(std::meta::dealias(t))[0]);
}

// IxCoord<Id>: the index observed as a value. A leaf like a generator —
// no buffer, no scalar — but index-bearing, and it pins no extent.
consteval bool is_ix_coord(std::meta::info t) {
    return is_specialization_of(std::meta::dealias(t), ^^IxCoord);
}
consteval size_t ix_coord_id(std::meta::info t) {
    return extract<size_t>(template_arguments_of(std::meta::dealias(t))[0]);
}

// Placed<Policy, Extent, Coord> — its policy and destination extent. One
// argument at a time: args_of extracts to the end of the pack, and these
// three are heterogeneous.
consteval Place placed_policy_of(std::meta::info t) {
    return std::meta::extract<Place>(
        std::meta::template_arguments_of(std::meta::dealias(t))[0]);
}
consteval size_t placed_ext_of(std::meta::info t) {
    return std::meta::extract<size_t>(
        std::meta::template_arguments_of(std::meta::dealias(t))[1]);
}
consteval std::string_view place_name(Place p) {
    return p == Place::Wrap    ? "wrap"
           : p == Place::Clamp ? "clamp"
           : p == Place::Drop  ? "drop"
                               : "exact";
}

consteval std::vector<size_t> summed_of(std::meta::info op) {
    return args_of<size_t>(op, 1); // ops::Fold's args are [Op, Summed…]
}

consteval std::meta::info fold_op_of(std::meta::info op) {
    return alias_of(op, "op");
}

// The indexed leaf: Indexed<Operand, Maps…> — operand first, then the maps.
consteval bool is_indexed(std::meta::info t) {
    return is_specialization_of(t, ^^Indexed);
}
consteval std::meta::info indexed_operand_of(std::meta::info t) {
    return std::meta::dealias(
        std::meta::template_arguments_of(std::meta::dealias(t))[0]);
}
// Indexed<Operand, Data, Maps…>: the maps start after the coordinate pack.
consteval std::vector<DecMap> maps_of(std::meta::info t) {
    return args_of<DecMap>(t, 2);
}

// The coordinate expressions, one slot per axis (NoCoord where affine).
consteval std::vector<std::meta::info> indexed_coords_of(std::meta::info t) {
    auto d = std::meta::dealias(
        std::meta::template_arguments_of(std::meta::dealias(t))[1]);
    std::vector<std::meta::info> out;
    auto args = std::meta::template_arguments_of(d);
    for (size_t q = 1; q < args.size(); ++q) // [0] is the index sequence
        out.push_back(std::meta::dealias(args[q]));
    return out;
}

consteval bool is_no_coord(std::meta::info t) {
    return std::meta::dealias(t) == ^^NoCoord;
}

// First op satisfying pred, or {} — an indexed leaf's operand included.
template <typename P>
consteval std::meta::info first_op(std::meta::info node, P pred) {
    auto t = std::meta::dealias(node);
    if (is_broadcast_scalar_type(t))
        return {};
    if (is_indexed(t)) {
        for (auto c : indexed_coords_of(t)) // a gather's coordinates too
            if (!is_no_coord(c))
                if (auto f = first_op(c, pred); f != std::meta::info{})
                    return f;
        return first_op(indexed_operand_of(t), pred);
    }
    if (is_placed(t)) // a scatter destination is its coordinate
        return first_op(placed_coord_of(t), pred);
    auto op = op_of(t);
    if (op == std::meta::info{}) // view or scalar leaf
        return {};
    if (pred(op))
        return op;
    for (auto c : children_of(t))
        if (auto f = first_op(c, pred); f != std::meta::info{})
            return f;
    return {};
}

consteval bool tree_has_contraction(std::meta::info node) {
    return first_op(node, is_contraction) != std::meta::info{};
}

// Any GATHERED axis below — a subscript resolved from data rather than from
// an affine form.
consteval bool tree_has_gather(std::meta::info node) {
    auto t = std::meta::dealias(node);
    if (is_broadcast_scalar_type(t))
        return false;
    if (is_indexed(t)) {
        for (const auto &m : maps_of(t))
            if (map_data(m))
                return true;
        for (auto c : indexed_coords_of(t))
            if (!is_no_coord(c) && tree_has_gather(c))
                return true;
        return tree_has_gather(indexed_operand_of(t));
    }
    if (is_placed(t))
        return tree_has_gather(placed_coord_of(t));
    if (op_of(t) == std::meta::info{})
        return false;
    for (auto c : children_of(t))
        if (tree_has_gather(c))
            return true;
    return false;
}

// Any subscripted leaf below — the type-level index_bearing test.
consteval bool tree_has_indexed(std::meta::info node) {
    auto t = std::meta::dealias(node);
    if (is_broadcast_scalar_type(t))
        return false;
    if (is_placed(t))
        return tree_has_indexed(placed_coord_of(t));
    if (is_indexed(t) || is_ix_coord(t))
        return true;
    if (op_of(t) == std::meta::info{})
        return false;
    for (auto c : children_of(t))
        if (tree_has_indexed(c))
            return true;
    return false;
}

// The extent values of a std::extents type / of a node's extents_type.
consteval std::vector<size_t> extents_of(std::meta::info extents) {
    return args_of<size_t>(extents, 1);
}
consteval std::vector<size_t> node_extents_of(std::meta::info t) {
    return extents_of(alias_of(t, "extents_type"));
}

// ── index census of a summand ───────────────────────────────────────────────
// Which placeholders appear (first-appearance order = result axis order) and
// what pins each extent: a bare occurrence authoritatively, else a
// unit-coefficient one; scaled-only pins nothing.
// At most one entry per placeholder, so the census carries its own storage
// rather than a vector — which is what makes IdCensus a literal type, and
// therefore memoizable as a per-node-type constant (census_v below).
using Ids = SmallVec<size_t, index_slots>;

struct IdCensus {
    Ids order;                              // ids, first-appearance DFS order
    std::array<size_t, index_slots> bare{}; // extent pinned by a bare slot
    std::array<size_t, index_slots> unit{}; // extent pinned by a unit term
    size_t clash_id = index_slots; // set when pinning occurrences disagree
    size_t clash_a = 0, clash_b = 0;

    // The resolved extent of an id: bare wins, unit else, 0 = unpinned.
    constexpr size_t pinned(size_t id) const {
        return bare[id] ? bare[id] : unit[id];
    }
};

// One AFFINE axis's contribution: its map against the operand's extent.
// The operand's leaves are ordinary reads, not index algebra, so this never
// recurses — which is what makes it an O(1) memoization base. A gathered
// axis never reaches here: its coordinate is a value, so it pins no id and
// its empty Lin must not reach the pinning loops. The coordinate's own ids
// join at this axis's POSITION, so first-appearance order survives.
consteval void collect_indexed_axis(const std::vector<DecMap> &ms,
                                 const std::vector<size_t> &ext, size_t k,
                                 IdCensus &s) {
    {
        {
            for (size_t st = 0; st < ms[k].n; ++st)
                for (size_t p = 0; p < index_slots; ++p)
                    if (ms[k].s[st].lin.c[p] != 0 &&
                        !std::ranges::contains(s.order, p))
                        s.order.push_back(p);
            if (int b = map_bare_slot(ms[k]); b >= 0) {
                if (s.bare[size_t(b)] && s.bare[size_t(b)] != ext[k] &&
                    s.clash_id == index_slots) {
                    s.clash_id = size_t(b);
                    s.clash_a = s.bare[size_t(b)];
                    s.clash_b = ext[k];
                }
                if (!s.bare[size_t(b)])
                    s.bare[size_t(b)] = ext[k];
            } else {
                // Any stage's unit coefficient associates the id with this
                // axis (the read is guarded or boundary-resolved, so a
                // differing extent is normal).
                for (size_t st = 0; st < ms[k].n; ++st)
                    for (size_t p = 0; p < index_slots; ++p) {
                        if (ms[k].s[st].lin.c[p] != 1)
                            continue;
                        if (s.unit[p] && s.unit[p] != ext[k] && !s.bare[p] &&
                            s.clash_id == index_slots) {
                            s.clash_id = p;
                            s.clash_a = s.unit[p];
                            s.clash_b = ext[k];
                        }
                        if (!s.unit[p])
                            s.unit[p] = ext[k];
                    }
            }
        }
    }
}

consteval void collect_ids(std::meta::info node, IdCensus &s) {
    auto t = std::meta::dealias(node);
    if (is_broadcast_scalar_type(t))
        return;
    if (is_indexed(t)) {
        // Axis by axis, so a gathered axis registers its coordinate's ids
        // exactly where an affine axis would register its own.
        auto ms = maps_of(t);
        auto ext = node_extents_of(indexed_operand_of(t));
        auto cs = indexed_coords_of(t);
        for (size_t k = 0; k < ms.size(); ++k)
            if (map_data(ms[k]))
                collect_ids(cs[k], s);
            else
                collect_indexed_axis(ms, ext, k, s);
        return;
    }
    if (is_placed(t)) {
        collect_ids(placed_coord_of(t), s); // a destination's own reads
        return;
    }
    if (is_ix_coord(t)) {
        // The id joins the order — it IS an axis of the result — but pins
        // no extent: observing a coordinate says nothing about how far it
        // runs. Some read of the same id must pin it, or first_unpinned
        // names it.
        const size_t id = ix_coord_id(t);
        for (size_t q = 0; q < s.order.n; ++q)
            if (s.order.v[q] == id)
                return;
        s.order.push_back(id);
        return;
    }
    auto op = op_of(t);
    if (op == std::meta::info{})
        return; // view leaf
    if (is_contraction(op)) {
        // A fold consumes its listed ids: only the summand's survivors join
        // the outer census — the consumed ids are scoped to the fold. A
        // scatter is the same rule over all of its children.
        IdCensus inner;
        for (auto c : children_of(t))
            collect_ids(c, inner);
        auto ids = summed_of(op);
        for (auto id : inner.order) {
            if (std::ranges::contains(ids, id))
                continue;
            if (!std::ranges::contains(s.order, id))
                s.order.push_back(id);
            if (inner.bare[id]) {
                if (s.bare[id] && s.bare[id] != inner.bare[id] &&
                    s.clash_id == index_slots) {
                    s.clash_id = id;
                    s.clash_a = s.bare[id];
                    s.clash_b = inner.bare[id];
                }
                if (!s.bare[id])
                    s.bare[id] = inner.bare[id];
            }
            if (inner.unit[id] && !s.unit[id])
                s.unit[id] = inner.unit[id];
        }
        return;
    }
    for (auto c : children_of(t))
        collect_ids(c, s);
}

consteval IdCensus id_census(std::meta::info summand) {
    IdCensus s;
    collect_ids(summand, s);
    // A bare pin overrides any unit-occurrence disagreement (the unit read
    // is guarded, so a differing axis extent is normal, e.g. f[zero(i + j - k)]).
    if (s.clash_id != index_slots && s.bare[s.clash_id] &&
        s.bare[s.clash_id] != s.clash_a)
        s.clash_id = index_slots;
    return s;
}

// The census of a whole node from its children (an index-bearing node's
// free ids, in first-appearance order across the child list).
consteval IdCensus merged_census(const std::vector<std::meta::info> &children) {
    IdCensus s;
    for (auto c : children)
        collect_ids(c, s);
    if (s.clash_id != index_slots && s.bare[s.clash_id] &&
        s.bare[s.clash_id] != s.clash_a)
        s.clash_id = index_slots;
    return s;
}

// ── the census as a memoized per-type fact ──────────────────────────────────
// collect_ids re-walks the subtree on every call, and make_expr (plus
// Expr's extents_type) asks once per node — so the census costs the whole
// tree, per node. A variable template is memoized per specialization, so
// the same answer built by MERGING the children's summaries is O(arity).

// b merged into a, in DFS order: the first pin of an id wins and the first
// disagreement is the one reported.
consteval IdCensus merge_census(IdCensus a, const IdCensus &b) {
    for (auto id : b.order) {
        if (!std::ranges::contains(a.order, id))
            a.order.push_back(id);
        if (b.bare[id]) {
            if (a.bare[id] && a.bare[id] != b.bare[id] &&
                a.clash_id == index_slots) {
                a.clash_id = id;
                a.clash_a = a.bare[id];
                a.clash_b = b.bare[id];
            }
            if (!a.bare[id])
                a.bare[id] = b.bare[id];
        }
        if (b.unit[id]) {
            if (a.unit[id] && a.unit[id] != b.unit[id] && !a.bare[id] &&
                a.clash_id == index_slots) {
                a.clash_id = id;
                a.clash_a = a.unit[id];
                a.clash_b = b.unit[id];
            }
            if (!a.unit[id])
                a.unit[id] = b.unit[id];
        }
    }
    if (a.clash_id == index_slots && b.clash_id != index_slots) {
        a.clash_id = b.clash_id;
        a.clash_a = b.clash_a;
        a.clash_b = b.clash_b;
    }
    return a;
}

// A fold consumes its listed ids: only the summand's survivors are visible
// outside, and a disagreement among the consumed ids is scoped to the fold.
consteval IdCensus drop_summed(const IdCensus &inner,
                                const std::vector<size_t> &ids) {
    IdCensus s;
    for (auto id : inner.order) {
        if (std::ranges::contains(ids, id))
            continue;
        s.order.push_back(id);
        s.bare[id] = inner.bare[id];
        s.unit[id] = inner.unit[id];
    }
    return s;
}

// A bare pin overrides any unit-occurrence disagreement (the unit read is
// guarded, so a differing axis extent is normal, e.g. f[zero(i + j - k)]).
consteval IdCensus finalized(IdCensus s) {
    if (s.clash_id != index_slots && s.bare[s.clash_id] &&
        s.bare[s.clash_id] != s.clash_a)
        s.clash_id = index_slots;
    return s;
}

template <typename T> inline constexpr IdCensus census_v{};

template <typename Op, typename... Cs> consteval IdCensus expr_census();

// The memoized twin of collect_ids' indexed arm: axis by axis, a gathered
// axis merging its coordinate's own (already memoized) census, so the whole
// leaf stays O(arity) rather than re-walking the coordinate subtree.
template <typename I, typename... Ds> consteval IdCensus leaf_scan() {
    IdCensus s;
    const auto ms = maps_of(^^I);
    const auto ext = node_extents_of(indexed_operand_of(^^I));
    [&]<size_t... K>(std::index_sequence<K...>) {
        (
            [&] {
                if (ms[K].data)
                    s = merge_census(s, census_v<Ds...[K]>);
                else
                    collect_indexed_axis(ms, ext, K, s);
            }(),
            ...);
    }(std::index_sequence_for<Ds...>{});
    return s;
}

template <typename E, typename Seq, typename... Ds, DecMap... Ms>
inline constexpr IdCensus census_v<Indexed<E, SlotsImpl<Seq, Ds...>, Ms...>> =
    leaf_scan<Indexed<E, SlotsImpl<Seq, Ds...>, Ms...>, Ds...>();
template <Place P, size_t E, typename C>
inline constexpr IdCensus census_v<Placed<P, E, C>> = census_v<C>;
template <size_t Id>
inline constexpr IdCensus census_v<IxCoord<Id>> = [] {
    IdCensus s;
    s.order.push_back(Id); // in the order, pinning nothing
    return s;
}();
template <typename Op, typename... Cs>
inline constexpr IdCensus census_v<Expr<Op, Cs...>> = expr_census<Op, Cs...>();

template <typename Op, typename... Cs> consteval IdCensus expr_census() {
    if constexpr (FoldOp<Op>) {
        // A fold has exactly one child; a scatter has its destinations too,
        // and both consume the listed ids.
        IdCensus inner;
        ((inner = merge_census(inner, census_v<Cs>)), ...);
        return drop_summed(inner, {Op::summed.begin(), Op::summed.end()});
    } else {
        IdCensus a;
        ((a = merge_census(a, census_v<Cs>)), ...);
        return a;
    }
}

// The census of a node about to be built, from its operands.
template <typename... Cs> consteval IdCensus children_census() {
    IdCensus a;
    ((a = merge_census(a, census_v<std::remove_cvref_t<Cs>>)), ...);
    return finalized(a);
}

// The free-index space as a std::extents — an unpinned id contributes
// extent 0 (the eval-side lints name it before anything reads the shape).
consteval std::meta::info free_extents_of(const IdCensus &s) {
    std::vector<std::meta::info> args{^^size_t};
    for (size_t id : s.order)
        args.push_back(std::meta::reflect_constant(s.pinned(id)));
    return std::meta::substitute(^^std::extents, args);
}

// The first placeholder that appears (in a subscript, or in the contract
// list) with nothing to pin its extent — index_slots if none.
consteval size_t first_unpinned(std::meta::info summand,
                                const std::vector<size_t> &summed) {
    auto s = id_census(summand);
    for (auto u : s.order)
        if (s.pinned(u) == 0)
            return u;
    for (auto a : summed)
        if (!std::ranges::contains(s.order, a))
            return a;
    return index_slots;
}

// The free placeholders' extents, in first-appearance order.
consteval std::meta::info
contracted_extents(std::meta::info summand, const std::vector<size_t> &summed) {
    auto s = id_census(summand);
    std::vector<std::meta::info> args{^^size_t};
    for (auto id : s.order)
        if (!std::ranges::contains(summed, id))
            args.push_back(std::meta::reflect_constant(s.pinned(id)));
    return std::meta::substitute(^^std::extents, args);
}

// The census flattened into arrays a runtime loop can walk.
struct ContractPlan {
    size_t n_free = 0, n_summed = 0, fold_count = 1;
    std::array<size_t, index_slots> free_id{}, summed_id{}, summed_ext{};
};

consteval ContractPlan contract_plan(std::meta::info summand,
                                     const std::vector<size_t> &summed) {
    auto s = id_census(summand);
    ContractPlan p{};
    for (auto id : s.order)
        if (!std::ranges::contains(summed, id))
            p.free_id[p.n_free++] = id;
    for (auto a : summed) {
        p.summed_id[p.n_summed] = a;
        p.summed_ext[p.n_summed++] = s.pinned(a);
        p.fold_count *= s.pinned(a);
    }
    return p;
}

// The free census flattened for an index-bearing eval's env loop.
struct FreePlan {
    size_t n = 0;
    std::array<size_t, index_slots> id{}, ext{};
};

consteval FreePlan free_plan(std::meta::info node) {
    auto s = id_census(node);
    FreePlan p{};
    for (auto i : s.order) {
        p.id[p.n] = i;
        p.ext[p.n++] = s.pinned(i);
    }
    return p;
}

// A scatter's axes, flattened for both the runtime loops and the emitter.
// ContractPlan cannot describe it: a scatter has DESTINATIONS, which are
// axes of the result that no placeholder names.
struct ScatterLayout {
    size_t n_dest = 0, n_surv = 0, n_sum = 0;
    size_t dest_size = 1, surv_size = 1, sum_size = 1;
    std::array<size_t, index_slots> dest_ext{};
    std::array<size_t, index_slots> surv_id{}, surv_ext{};
    std::array<size_t, index_slots> sum_id{}, sum_ext{};

    // What the OUTPUT holds, and what the work is: they differ here and
    // coincide for every other node, which is why both have a name.
    constexpr size_t output_cells() const { return dest_size * surv_size; }
    constexpr size_t contributions() const { return sum_size * surv_size; }
};

// Reflective, so the CPU evaluator and the GPU emitter share one answer —
// the leading axes of the node's own extents ARE the destinations, because
// scatter_extents built them from these same children.
consteval ScatterLayout scatter_layout(std::meta::info node) {
    const auto t = std::meta::dealias(node);
    const auto kids = children_of(t);
    const auto ext = node_extents_of(t);
    const auto ids = summed_of(op_of(t));
    ScatterLayout l{};
    l.n_dest = kids.size() - 1;
    for (size_t k = 0; k < l.n_dest; ++k) {
        l.dest_ext[k] = ext[k];
        l.dest_size *= ext[k];
    }
    const auto s = merged_census(kids);
    for (size_t a : ids) {
        l.sum_id[l.n_sum] = a;
        l.sum_ext[l.n_sum++] = s.pinned(a);
        l.sum_size *= s.pinned(a);
    }
    for (size_t id : s.order) {
        if (std::ranges::contains(ids, id))
            continue;
        l.surv_id[l.n_surv] = id;
        l.surv_ext[l.n_surv++] = s.pinned(id);
        l.surv_size *= s.pinned(id);
    }
    return l;
}

// index_slots when ids is exactly a permutation of the free set, else an
// offending id (duplicate, unknown, or a free index left unnamed).
consteval size_t order_mismatch(const FreePlan &p,
                                const std::vector<size_t> &ids) {
    for (size_t x = 0; x < ids.size(); ++x)
        for (size_t y = x + 1; y < ids.size(); ++y)
            if (ids[x] == ids[y])
                return ids[x];
    for (size_t id : ids) {
        bool found = false;
        for (size_t a = 0; a < p.n; ++a)
            found |= (p.id[a] == id);
        if (!found)
            return id;
    }
    for (size_t a = 0; a < p.n; ++a) {
        bool named = false;
        for (size_t id : ids)
            named |= (id == p.id[a]);
        if (!named)
            return p.id[a];
    }
    return index_slots;
}

consteval FreePlan ordered_plan(const FreePlan &p,
                                const std::vector<size_t> &ids) {
    FreePlan q{};
    for (size_t id : ids)
        for (size_t a = 0; a < p.n; ++a)
            if (p.id[a] == id) {
                q.id[q.n] = id;
                q.ext[q.n++] = p.ext[a];
            }
    return q;
}

// A tree's output rank without demanding member aliases (a lone indexed
// leaf carries none).
template <typename D> consteval size_t rank_of() {
    if constexpr (index_bearing_v<D>)
        return free_plan(std::meta::dealias(^^D)).n;
    else
        return D::rank;
}

// ── the shift interior ──────────────────────────────────────────────────────
// A tree whose every leaf reads at a CONSTANT displacement of the output
// coordinate, axis for axis. Inside the box where no displaced read can
// leave the grid, no boundary policy can fire and each leaf sits at one
// fixed flat offset from the output index — so the body is contiguous
// loads and vectorizes like the hand-written loop. Outside it, the
// general per-cell path still applies the policies.
struct ShiftPlan {
    bool ok = false;
    std::array<size_t, index_slots> lo{}, hi{};
};

consteval void shift_scan(std::meta::info node, const FreePlan &fp,
                          ShiftPlan &sp) {
    if (!sp.ok)
        return;
    auto t = std::meta::dealias(node);
    if (is_broadcast_scalar_type(t))
        return;
    if (is_indexed(t)) {
        const auto maps = maps_of(t);
        const auto ext = node_extents_of(indexed_operand_of(t));
        // A gathered axis is not a constant displacement of anything, and
        // eval_shifted's shift_delta reads s[0].lin.off with no test at all
        // — so this bail is the only thing standing between a gather and a
        // fixed-offset read of the wrong element on the vectorized path.
        for (const auto &m : maps)
            if (map_data(m)) {
                sp.ok = false;
                return;
            }
        if (maps.size() != fp.n || ext.size() != fp.n) {
            sp.ok = false; // a leaf of another shape: not a pure shift
            return;
        }
        for (size_t k = 0; k < fp.n; ++k) {
            if (ext[k] != fp.ext[k] || maps[k].n != 1) {
                sp.ok = false; // different extents, or a decoration chain
                return;
            }
            const Lin l = maps[k].s[0].lin;
            for (size_t p = 0; p < index_slots; ++p)
                if (l.c[p] != (p == fp.id[k] ? 1 : 0)) {
                    sp.ok = false; // not this axis, or not unit coefficient
                    return;
                }
            if (l.off > 0) {
                const size_t h = size_t(l.off) >= fp.ext[k]
                                     ? 0
                                     : fp.ext[k] - size_t(l.off);
                sp.hi[k] = h < sp.hi[k] ? h : sp.hi[k];
            } else if (l.off < 0) {
                const size_t lo = size_t(-l.off);
                sp.lo[k] = lo > sp.lo[k] ? lo : sp.lo[k];
            }
        }
        return;
    }
    const auto op = op_of(t);
    if (op == std::meta::info{} || is_contraction(op)) {
        sp.ok = false; // a bare leaf or a fold: the general path owns those
        return;
    }
    for (auto c : children_of(t))
        shift_scan(c, fp, sp);
}

consteval ShiftPlan shift_plan(std::meta::info node, const FreePlan &fp) {
    ShiftPlan sp;
    sp.ok = fp.n > 0;
    for (size_t a = 0; a < fp.n; ++a) {
        sp.lo[a] = 0;
        sp.hi[a] = fp.ext[a];
    }
    shift_scan(node, fp, sp);
    for (size_t a = 0; a < fp.n; ++a)
        if (sp.lo[a] >= sp.hi[a])
            sp.hi[a] = sp.lo[a]; // an empty interior: all shell
    return sp;
}

// ── the contraction lints ───────────────────────────────────────────────────
consteval void indexed_maps_under(std::meta::info node,
                                  std::vector<std::vector<DecMap>> &out) {
    auto t = std::meta::dealias(node);
    if (is_broadcast_scalar_type(t))
        return;
    if (is_indexed(t)) {
        out.push_back(maps_of(t));
        for (auto c : indexed_coords_of(t)) // reads inside a gather count too
            if (!is_no_coord(c))
                indexed_maps_under(c, out);
        return;
    }
    if (is_placed(t)) {
        indexed_maps_under(placed_coord_of(t), out);
        return;
    }
    if (op_of(t) == std::meta::info{})
        return;
    for (auto c : children_of(t))
        indexed_maps_under(c, out);
}

// A summand whose every leaf map is bare cannot read out of range — the
// condition under which eval may interchange the contraction's loops.
consteval bool all_maps_bare(std::meta::info summand) {
    std::vector<std::vector<DecMap>> leaves;
    indexed_maps_under(summand, leaves);
    for (const auto &ms : leaves)
        for (const auto &m : ms)
            if (!map_bare(m))
                return false;
    return true;
}

// Where the per-cell fold walks contiguous memory: the innermost summed id
// (listed last, so innermost in the loop nest) must move each leaf along
// its OWN innermost axis, or not move it at all. That is the case the
// loop interchange must not take — streaming the operands would stride
// what is already sequential, and re-walk the whole output per summed
// step. Folding a strided axis (a column sum, Aᵀv, one side of a matmul)
// fails this and keeps the interchange.
consteval bool fold_reads_contiguously(std::meta::info summand,
                                       const std::vector<size_t> &summed) {
    if (summed.empty())
        return false;
    const size_t last = summed.back();
    std::vector<std::vector<DecMap>> leaves;
    indexed_maps_under(summand, leaves);
    for (const auto &ms : leaves) {
        int moved = -1; // the axis the innermost summed id advances
        for (size_t k = 0; k < ms.size(); ++k) {
            if (map_data(ms[k]))
                return false; // a gathered axis advances unpredictably
            const int c = ms[k].s[0].lin.c[last];
            if (ms[k].n != 1 || c == 0)
                continue;
            if (c != 1 || moved != -1)
                return false; // scaled, or two axes move together
            moved = int(k);
        }
        if (moved != -1 && size_t(moved) + 1 != ms.size())
            return false; // moves a non-innermost axis: strided reads
    }
    return true;
}

// Degenerate: every leaf reads the same bare distinct placeholders — that is
// reduce's job. A diagonal read is not degenerate; reduce cannot reach it.
consteval bool contraction_degenerate(std::meta::info summand) {
    std::vector<std::vector<DecMap>> leaves;
    indexed_maps_under(summand, leaves);
    std::vector<size_t> seq;
    for (size_t n = 0; n < leaves.size(); ++n) {
        std::vector<size_t> ids;
        for (auto &m : leaves[n]) {
            int b = map_bare_slot(m);
            if (b < 0)
                return false; // an affine form: not an elementwise read
            if (std::ranges::contains(ids, size_t(b)))
                return false; // diagonal: reduce cannot spell it
            ids.push_back(size_t(b));
        }
        if (n == 0)
            seq = ids;
        else if (seq != ids)
            return false; // different maps: a genuine contraction
    }
    return true;
}

// Separable: in some monomial (split at binary +/−) the summed indices fall
// into groups no leaf couples — the fold factors, so one node wastes work.
consteval void monomials_of(std::meta::info node,
                            std::vector<std::vector<std::vector<DecMap>>> &out) {
    auto t = std::meta::dealias(node);
    if (is_broadcast_scalar_type(t))
        return;
    auto op = op_of(t);
    if (op != std::meta::info{} && !is_indexed(t)) {
        auto cs = children_of(t);
        auto sym = symbol_of(op);
        if (cs.size() == 2 && (sym == "+" || sym == "-")) {
            monomials_of(cs[0], out);
            monomials_of(cs[1], out);
            return;
        }
    }
    std::vector<std::vector<DecMap>> leaves;
    indexed_maps_under(t, leaves);
    if (!leaves.empty())
        out.push_back(leaves);
}

consteval bool contraction_separable(std::meta::info summand,
                                     const std::vector<size_t> &summed) {
    if (summed.size() < 2)
        return false;
    std::vector<std::vector<std::vector<DecMap>>> monomials;
    monomials_of(summand, monomials);
    for (auto &leaves : monomials) {
        // union-find: a leaf involving several summed ids couples them
        std::array<size_t, index_slots> parent{};
        for (size_t p = 0; p < index_slots; ++p)
            parent[p] = p;
        auto find = [&](size_t x) {
            while (parent[x] != x)
                x = parent[x];
            return x;
        };
        std::vector<size_t> present;
        for (auto &maps : leaves) {
            size_t first = index_slots;
            for (auto id : summed) {
                bool in_leaf = false;
                for (auto &m : maps)
                    for (size_t st = 0; st < m.n; ++st)
                        in_leaf |= (m.s[st].lin.c[id] != 0);
                if (!in_leaf)
                    continue;
                if (!std::ranges::contains(present, id))
                    present.push_back(id);
                if (first == index_slots)
                    first = id;
                else
                    parent[find(id)] = find(first);
            }
        }
        if (present.size() < 2)
            continue;
        size_t roots = 0;
        for (auto id : present)
            roots += (find(id) == id);
        if (roots > 1)
            return true;
    }
    return false;
}

// ── the index-string algebra ────────────────────────────────────────────────

consteval std::vector<size_t> strides_of(const std::vector<size_t> &ext) {
    std::vector<size_t> s(ext.size(), 1); // row-major
    for (size_t k = ext.size(); k-- > 1;)
        s[k - 1] = s[k] * ext[k];
    return s;
}

// A read under collected in-range guards; the false side never evaluates
// and yields the read's fill.
consteval std::string guarded(const std::string &guards,
                              const std::string &inner,
                              const std::string &fill = "0") {
    return guards.empty() ? inner : "(" + guards + " ? " + inner + " : " + fill +
                                        ")";
}


// Axis k of a flat index — with the elisions the index range guarantees
// (axis 0 needs no %, the last axis no /).
consteval std::string axis_coord(const std::string &idx,
                                 const std::vector<size_t> &ext, size_t k) {
    if (ext.size() <= 1)
        return idx;
    const auto stride = strides_of(ext);
    if (k == 0)
        return idx + " / " + to_string(stride[0]);
    if (k == ext.size() - 1)
        return idx + " % " + to_string(ext[k]);
    return idx + " / " + to_string(stride[k]) + " % " + to_string(ext[k]);
}

// One shift as one inline spelling, valid in both dialects: decompose,
// displace with constants folded, recompose. Zero collects guards instead.
// Identifier symbols spell as calls, the rest infix.
consteval bool identifier_like(std::string_view sym) {
    return !sym.empty() && (sym[0] == '_' || (sym[0] >= 'a' && sym[0] <= 'z') ||
                            (sym[0] >= 'A' && sym[0] <= 'Z') ||
                            (sym[0] >= '0' && sym[0] <= '9'));
}

// ── the contraction index algebra ───────────────────────────────────────────
// An indexed operand's flat subscript from the placeholder coordinates.
// Bare/constant maps are in range by construction; other affine forms
// collect guards (the lower one elided when no term can go negative).
struct AffineIndex {
    std::string index;
    std::string guards; // empty for bare/constant maps
};

// One stage's affine expression over the coordinate environment, seeded
// with the carried coordinate of the previous stage.
struct LinExpr {
    std::string expr;
    bool negative = false; // some term can go negative
};

consteval LinExpr lin_expr(Lin m, const std::vector<std::string> &coord,
                           const std::string &seed) {
    std::string e = seed;
    bool negative = m.off < 0;
    for (size_t p = 0; p < index_slots; ++p) {
        const int c = m.c[p];
        if (c == 0)
            continue;
        negative |= (c < 0);
        const auto mag = size_t(c < 0 ? -c : c);
        std::string term =
            (mag == 1 ? coord[p] : to_string(mag) + " * " + coord[p]);
        if (e.empty())
            e = (c < 0 ? "-" : "") + term;
        else
            e += (c < 0 ? " - " : " + ") + term;
    }
    if (m.off != 0)
        e += (m.off < 0 ? " - " : " + ") +
             to_string(size_t(m.off < 0 ? -m.off : m.off));
    return {e, negative};
}

// `data` carries one pre-rendered coordinate per axis, non-empty exactly
// where the map is a gathered one. The caller renders those first so their
// leaf slots are claimed in for_each_leaf's order.
consteval AffineIndex
decorated_index(const std::vector<DecMap> &maps,
                const std::vector<size_t> &ext,
                const std::vector<std::string> &coord,
                const std::vector<std::string> &data = {}) {
    const size_t r = ext.size();
    const auto stride = strides_of(ext); // over the operand
    std::string out, guards;
    size_t terms = 0;
    for (size_t k = 0; k < r; ++k) {
        const DecMap m = maps[k];
        std::string x;
        if (map_data(m)) {
            // A gathered axis: the caller has already rendered the
            // coordinate (claiming its leaf slots), so seeding x with it
            // makes every policy spelling below apply verbatim.
            x = data[k];
        } else if (int b = map_bare_slot(m); b >= 0) {
            x = coord[size_t(b)];
        } else if (map_plain(m) && lin_const(m.s[0].lin)) {
            x = to_string(size_t(m.s[0].lin.off)); // range-checked upstream
        }
        if (!map_bare(m) && !(map_plain(m) && lin_const(m.s[0].lin))) {
            // Stages innermost-first: each policy resolves the carried
            // coordinate before the next stage displaces it. After a
            // Wrap/Clamp stage the carry is non-negative by construction.
            const std::string es = to_string(ext[k]);
            for (size_t st = 0; st < m.n; ++st) {
                auto le = lin_expr(m.s[st].lin, coord, x);
                // A runtime coordinate is never provably non-negative, and
                // no coefficient can say so — force the lower guard.
                le.negative |= map_data(m);
                if (m.s[st].pol == Policy::Wrap) {
                    x = "((" + le.expr + ") % " + es + " + " + es + ") % " +
                        es;
                } else if (m.s[st].pol == Policy::Clamp) {
                    const std::string e2 = "(" + le.expr + ")";
                    x = "(" + e2 + " < 0 ? 0 : " + e2 + " >= " + es + " ? " +
                        es + " - 1 : " + e2 + ")";
                } else { // None/Zero: collect guards, keep the coordinate
                    if (le.negative)
                        guards +=
                            (guards.empty() ? "" : " && ") + le.expr + " >= 0";
                    guards += (guards.empty() ? "" : " && ") + le.expr +
                              " < " + es;
                    x = "(" + le.expr + ")";
                }
            }
        }
        if (ext[k] == 1)
            continue; // a length-1 axis contributes nothing to the index
        if (terms++)
            out += " + ";
        out += x + (stride[k] != 1 ? " * " + to_string(stride[k]) : "");
    }
    if (terms == 0)
        return {"0", guards};
    return {terms > 1 ? "(" + out + ")" : out, guards};
}

// The output cell's own spelling, for the leaf that reads exactly it. Free
// coordinates are the flat index taken apart; a leaf reading every axis bare
// in the output's own order and extents puts it back together unchanged, so
// it subscripts with the flat index and skips the arithmetic. An empty
// `flat` disables the test — the environment then carries loop variables,
// which decompose nothing.
struct DirectRead {
    std::string flat;
    std::vector<size_t> id, ext;
};

consteval bool reads_output_cell(const std::vector<DecMap> &maps,
                                 const std::vector<size_t> &ext,
                                 const DirectRead &d) {
    if (d.flat.empty() || maps.size() != d.id.size() || ext != d.ext)
        return false;
    for (size_t k = 0; k < maps.size(); ++k)
        if (int b = map_bare_slot(maps[k]); b < 0 || size_t(b) != d.id[k])
            return false;
    return true;
}

// The element type's spelling in the emitted dialect. Lives here rather
// than with the rest of the GPU emitter because a generator renders a cast
// of its coordinate, and both dialects spell that the same way.
consteval std::string_view gpu_type(std::meta::info t) {
    t = std::meta::dealias(t);
    if (t == (^^float))
        return "float";
    if (t == (^^double))
        return "double";
    if (t == (^^int))
        return "int";
    if (t == (^^unsigned))
        return "uint";
    // A type that carries its own Slang name says so with the annotation an
    // op uses for its symbol — a structured accumulator's state and result
    // are declared in that op's emitted block under exactly this name.
    if (auto a = std::meta::annotations_of_with_type(t, ^^Symbol); !a.empty())
        return std::meta::extract<Symbol>(std::meta::constant_of(a[0])).str;
    return std::meta::display_string_of(t); // best effort beyond the basics
}

// Twin of is_generator_v (detail/Core.h).
consteval bool is_generator_type(std::meta::info t) {
    return is_specialization_of(std::meta::dealias(t), ^^Generator);
}

consteval GenKind gen_kind(std::meta::info t) {
    return std::meta::extract<GenKind>(
        std::meta::template_arguments_of(std::meta::dealias(t))[0]);
}

consteval size_t gen_count(std::meta::info t) {
    auto args = std::meta::template_arguments_of(std::meta::dealias(t));
    size_t n = 1;
    for (size_t k = 2; k < args.size(); ++k)
        n *= std::meta::extract<size_t>(args[k]);
    return n;
}

// How a dialect spells the two leaf kinds.
struct LeafStyle {
    std::string_view view_prefix;
    std::string_view view_open;
    std::string_view view_close;
    std::string_view scalar_prefix;
};

// The formula() dialect: in0[i], in1[i], … and s0, s1, …
inline constexpr LeafStyle formula_style{"in", "[", "]", "s"};
// The fill spelling of an indexed leaf, taking its scalar slot when the
// read is padded — claimed BEFORE the operand's leaves, as for_each_leaf
// visits it.
consteval std::string fill_spelling(std::meta::info t, LeafStyle style,
                                    size_t &scalars) {
    for (auto m : maps_of(t))
        if (map_padded(m))
            return std::string(style.scalar_prefix) + to_string(scalars++);
    return "0";
}

// A generator renders inline — its parameters and the coordinate, never a
// buffer. This is the algebra Generator::operator[] runs, so linspace still
// interpolates from BOTH ends and its end cells land exactly on a and b.
// Constants are spelt as casts in the element type (exact for any extent a
// buffer could address) so nothing widens to double on the way.
consteval void gen_into(std::meta::info t, const LeafStyle &style,
                        const std::string &idx, size_t &scalars,
                        std::string &out) {
    auto param = [&](std::string &o) {
        o += style.scalar_prefix;
        append_number(o, scalars++);
    };
    const auto tn = gpu_type(alias_of(t, "type"));
    const auto k = gen_kind(t);
    if (gen_is_sampler(k)) {
        // The two scalars are the key halves; the coordinate is the
        // counter. Everything else about the draw lives in the helper.
        out += k == GenKind::Uniform ? "rng_uniform(" : "rng_normal(";
        param(out);
        out += ", ";
        param(out);
        out += ", ";
        out += idx;
        out += ")";
        return;
    }
    if (k == GenKind::Fill) {
        param(out);
        return;
    }
    if (k == GenKind::Iota) {
        out += "(";
        param(out);
        out += " + ";
        out += tn;
        out += "(";
        out += idx;
        out += "))";
        return;
    }
    // LinSpace claims both parameters even where the ramp degenerates, so
    // the scalar numbering stays in step with gen_params.
    std::string a, b;
    param(a);
    param(b);
    const size_t n = gen_count(t);
    if (n <= 1) {
        out += a;
        return;
    }
    // t is parenthesised: b * i / L would divide the PRODUCT and lose the
    // exact endpoint that computing t first is there to guarantee.
    std::string s;
    s += "(";
    s += tn;
    s += "(";
    s += idx;
    s += ") / ";
    s += tn;
    s += "(";
    append_number(s, n - 1);
    s += "))";
    out += "(";
    out += a;
    out += " * (";
    out += tn;
    out += "(1) - ";
    out += s;
    out += ") + ";
    out += b;
    out += " * ";
    out += s;
    out += ")";
}

// Binary operators spell infix, everything else as a call.
consteval std::string spell_node(std::meta::info op,
                                 const std::vector<std::string> &child) {
    auto sym = std::string(symbol_of(op));
    if (child.size() == 2 && !identifier_like(sym))
        return "(" + child[0] + " " + sym + " " + child[1] + ")";
    std::string call = sym + "(";
    for (size_t i = 0; i < child.size(); ++i)
        call += (i ? ", " : "") + child[i];
    return call + ")";
}

// The APL-style fold spelling: `+/j0/j1(inner)`.
consteval std::string fold_spelling(std::meta::info op, size_t n,
                                    const std::string &inner) {
    std::string s(symbol_of(fold_op_of(op)));
    for (size_t d = 0; d < n; ++d)
        s += "/j" + to_string(d);
    return s + "(" + inner + ")";
}

// The renderers APPEND into one buffer rather than returning a string per
// node. Returning by value copies every child's text into its parent, which
// is O(output x depth) CONSTEXPR OPERATIONS — quadratic against GCC's
// -fconstexpr-ops-limit, which a deep tree then exhausts and fails to
// compile. Appending is linear. Wall time barely notices; the operation
// budget is what this protects.
consteval void render_into(std::meta::info node, const LeafStyle &style,
                           const std::string &idx, size_t &views,
                           size_t &scalars, std::string &out);
consteval void render_indexed_into(std::meta::info node, const LeafStyle &style,
                                   const std::vector<std::string> &coord,
                                   size_t &views, size_t &scalars,
                                   std::string &out,
                                   const DirectRead &direct = {});

// A gathered coordinate as a SIGNED index. Off an unsigned buffer clamp's
// low branch is dead and wrap's modulo is wrong near the top of the range,
// so the cast is not cosmetic. A floating-point coordinate is floored, and
// saturated first: converting one out of int range is undefined, and the
// policy that follows cannot undo it.
consteval std::string data_seed(std::meta::info coord, const std::string &text) {
    if (std::meta::is_floating_point_type(
            std::meta::dealias(alias_of(coord, "type"))))
        return "int(clamp(floor(" + text + "), -1.0e9, 1.0e9))";
    return "int(" + text + ")";
}

// One rendered coordinate per axis, empty where the axis is affine.
consteval std::vector<std::string>
render_data_coords(std::meta::info t, const LeafStyle &style,
                   const std::vector<std::string> &coord, size_t &views,
                   size_t &scalars) {
    const auto ms = maps_of(t);
    std::vector<std::string> data(ms.size());
    bool any = false;
    for (const auto &m : ms)
        any |= map_data(m);
    if (!any)
        return data;
    const auto cs = indexed_coords_of(t);
    for (size_t k = 0; k < ms.size(); ++k)
        if (map_data(ms[k])) {
            std::string text;
            render_indexed_into(cs[k], style, coord, views, scalars, text);
            data[k] = data_seed(cs[k], text);
        }
    return data;
}

// `op(a, b)` for an identifier-like symbol, `(a op b)` for an operator.
consteval void spell_children(std::meta::info op,
                              const std::vector<std::meta::info> &child,
                              auto &&emit, std::string &out) {
    auto sym = symbol_of(op);
    if (child.size() == 2 && !identifier_like(sym)) {
        out += "(";
        emit(child[0]);
        out += " ";
        out += sym;
        out += " ";
        emit(child[1]);
        out += ")";
        return;
    }
    out += sym;
    out += "(";
    for (size_t i = 0; i < child.size(); ++i) {
        if (i)
            out += ", ";
        emit(child[i]);
    }
    out += ")";
}

// The APL-style scan spelling: `+\i0(inner)` — the operator that KEEPS its
// index, against the fold's `/` that consumes one.
consteval void spell_scan_open(std::meta::info op, const std::string &at,
                               std::string &out) {
    out += symbol_of(fold_op_of(op));
    out += "\\";
    out += at;
    out += "(";
}

// The APL-style fold spelling: `+/j0/j1(inner)`.
consteval void spell_fold_open(std::meta::info op, size_t n, std::string &out) {
    out += symbol_of(fold_op_of(op));
    for (size_t d = 0; d < n; ++d) {
        out += "/j";
        append_number(out, d);
    }
    out += "(";
}

// The summand renderer: the index is an environment (one coordinate per
// placeholder) each indexed leaf turns into its own subscript.
consteval void render_indexed_into(std::meta::info node, const LeafStyle &style,
                                   const std::vector<std::string> &coord,
                                   size_t &views, size_t &scalars,
                                   std::string &out, const DirectRead &direct) {
    auto t = std::meta::dealias(node);
    if (is_broadcast_scalar_type(t)) {
        out += style.scalar_prefix;
        append_number(out, scalars++);
        return;
    }
    // Unreachable while a shaped generator cannot be subscripted (the
    // plain-meets-indexed lint stops it earlier), but a missing branch here
    // would render one as a buffer read — silently the wrong program.
    if (is_generator_type(t)) {
        gen_into(t, style, direct.flat, scalars, out);
        return;
    }
    // The index observed: the coordinate string itself, no leaf, no slot.
    if (is_ix_coord(t)) {
        out += coord[ix_coord_id(t)];
        return;
    }
    if (is_indexed(t)) {
        auto operand = indexed_operand_of(t);
        if (reads_output_cell(maps_of(t), node_extents_of(operand), direct)) {
            render_into(operand, style, direct.flat, views, scalars, out);
            return;
        }
        // the fill claims its scalar slot BEFORE the operand's leaves
        auto fill = fill_spelling(t, style, scalars);
        // then each gathered axis's coordinate, in axis order — the order
        // for_each_leaf visits them, so the slots stay in lockstep. Rendered
        // unconditionally: decorated_index elides a length-1 axis, and
        // skipping the render there would desynchronize the whole program.
        auto data = render_data_coords(t, style, coord, views, scalars);
        auto ai =
            decorated_index(maps_of(t), node_extents_of(operand), coord, data);
        if (ai.guards.empty()) {
            render_into(operand, style, ai.index, views, scalars, out);
            return;
        }
        out += "(" + ai.guards + " ? ";
        render_into(operand, style, ai.index, views, scalars, out);
        out += " : " + fill + ")";
        return;
    }
    // A scan keeps its index, so it renders in place: the running operator
    // named at the coordinate it runs along.
    if (auto op = op_of(t); op != std::meta::info{} && is_scan_op(op)) {
        // Named by its placeholder, not by the coordinate's spelling: at the
        // root that spelling is the flat index decomposed, which says
        // nothing about which axis the running op runs along.
        spell_scan_open(op, ix_name(scanned_of(op)), out);
        render_indexed_into(std::meta::dealias(children_of(t)[0]), style, coord,
                            views, scalars, out, direct);
        out += ")";
        return;
    }
    // Before the contraction test, which is TRUE for a scatter: its first
    // child is a destination, not a summand. A scatter can only appear here
    // under an elementwise epilogue, and it spells the same in both
    // renderers.
    if (auto op = op_of(t); op != std::meta::info{} && is_placed_op(op)) {
        render_into(t, style, "", views, scalars, out);
        return;
    }
    // An unsubscripted tensor beside one — the plain-meets-indexed lint
    // allows it only above a scatter, where the ops are elementwise over
    // the finished shape, so it reads the output cell.
    if (op_of(t) == std::meta::info{}) {
        render_into(t, style, direct.flat, views, scalars, out);
        return;
    }
    if (auto op = op_of(t); op != std::meta::info{} && is_contraction(op)) {
        // A fold below an epilogue: its ids become the loop spellings.
        auto summed = summed_of(op);
        std::vector<std::string> inner = coord;
        for (size_t d = 0; d < summed.size(); ++d)
            inner[summed[d]] = "j" + to_string(d);
        spell_fold_open(op, summed.size(), out);
        render_indexed_into(std::meta::dealias(children_of(t)[0]), style, inner,
                            views, scalars, out, direct);
        out += ")";
        return;
    }
    spell_children(
        op_of(t), children_of(t),
        [&](std::meta::info c) {
            render_indexed_into(c, style, coord, views, scalars, out, direct);
        },
        out);
}

// The descent below render_into's dispatch. It is only ever reached where
// the subtree carries no indexed leaf, so the index-bearing tests cannot
// fire again — leaving them out of the recursion is what keeps rendering
// linear in the tree rather than quadratic.
consteval void render_plain_into(std::meta::info node, const LeafStyle &style,
                                 const std::string &idx, size_t &views,
                                 size_t &scalars, std::string &out) {
    auto t = std::meta::dealias(node);
    if (is_broadcast_scalar_type(t)) {
        out += style.scalar_prefix;
        append_number(out, scalars++);
        return;
    }
    if (is_generator_type(t)) {
        gen_into(t, style, idx, scalars, out);
        return;
    }
    auto op = op_of(t);
    if (op == std::meta::info{}) { // tensor leaf
        out += style.view_prefix;
        append_number(out, views++);
        out += style.view_open;
        out += idx;
        out += style.view_close;
        return;
    }
    spell_children(
        op, children_of(t),
        [&](std::meta::info c) {
            render_plain_into(c, style, idx, views, scalars, out);
        },
        out);
}

// Leaves are numbered in DFS visit order — the same order for_each_leaf
// sees at runtime; `idx` is "i" at the root, displaced below an index op.
consteval void render_into(std::meta::info node, const LeafStyle &style,
                           const std::string &idx, size_t &views,
                           size_t &scalars, std::string &out) {
    auto t = std::meta::dealias(node);
    if (is_broadcast_scalar_type(t)) {
        out += style.scalar_prefix;
        append_number(out, scalars++);
        return;
    }
    if (is_generator_type(t)) {
        gen_into(t, style, idx, scalars, out);
        return;
    }
    // An index-bearing root renders through the coordinate environment:
    // free ids decompose idx over the OUTPUT extents, in census order.
    const auto op = op_of(t);
    if (tree_has_indexed(t)) {
        if (op == std::meta::info{} || !is_fold(op)) {
            auto p = free_plan(t);
            std::vector<size_t> out_ext;
            for (size_t a = 0; a < p.n; ++a)
                out_ext.push_back(p.ext[a]);
            std::vector<std::string> coord(index_slots);
            DirectRead direct{idx, {}, out_ext};
            for (size_t a = 0; a < p.n; ++a) {
                coord[p.id[a]] = axis_coord(idx, out_ext, a);
                direct.id.push_back(p.id[a]);
            }
            render_indexed_into(t, style, coord, views, scalars, out, direct);
            return;
        }
    }
    if (op == std::meta::info{}) { // tensor leaf
        out += style.view_prefix;
        append_number(out, views++);
        out += style.view_open;
        out += idx;
        out += style.view_close;
        return;
    }
    if (is_placed_op(op)) {
        // A scatter: the consumed ids are loop variables and there is no
        // output cell to decompose — the destinations say where each
        // contribution lands. `+>i0[clamp4(in0[i0])](in1[i0])`.
        const auto kids = children_of(t);
        const auto summed = summed_of(op);
        std::vector<std::string> coord(index_slots);
        for (size_t d = 0; d < summed.size(); ++d)
            coord[summed[d]] = "i" + to_string(d);
        for (size_t id : merged_census(kids).order)
            if (coord[id].empty())
                coord[id] = "n" + to_string(id);
        out += symbol_of(fold_op_of(op));
        out += ">";
        for (size_t d = 0; d < summed.size(); ++d)
            out += coord[summed[d]];
        out += "[";
        for (size_t k = 0; k + 1 < kids.size(); ++k) {
            if (k)
                out += ", ";
            out += place_name(placed_policy_of(kids[k]));
            append_number(out, placed_ext_of(kids[k]));
            out += "(";
            render_indexed_into(placed_coord_of(kids[k]), style, coord, views,
                                scalars, out);
            out += ")";
        }
        out += "](";
        render_indexed_into(std::meta::dealias(kids.back()), style, coord,
                            views, scalars, out);
        out += ")";
        return;
    }
    if (is_contraction(op)) {
        auto summed = summed_of(op);
        auto summand = std::meta::dealias(children_of(t)[0]);
        auto census = id_census(summand);
        auto out_ext = node_extents_of(t);
        // free coordinates decompose idx; summed ones are loop variables
        std::vector<std::string> coord(index_slots);
        DirectRead direct{idx, {}, out_ext};
        size_t kept = 0;
        for (auto id : census.order)
            if (!std::ranges::contains(summed, id)) {
                coord[id] = axis_coord(idx, out_ext, kept++);
                direct.id.push_back(id);
            }
        for (size_t d = 0; d < summed.size(); ++d)
            coord[summed[d]] = "j" + to_string(d);
        spell_fold_open(op, summed.size(), out);
        render_indexed_into(summand, style, coord, views, scalars, out, direct);
        out += ")";
        return;
    }
    spell_children(
        op, children_of(t),
        [&](std::meta::info c) {
            render_plain_into(c, style, idx, views, scalars, out);
        },
        out);
}

// The string-returning spellings the emitters call — one buffer per program.
// render() opens by asking whether the tree is index-bearing, which is a walk
// of the whole tree; render_plain is for a caller that already knows it is
// not, so the answer is established once per program rather than per render.
consteval std::string render(std::meta::info node, const LeafStyle &style,
                             std::string idx, size_t &views, size_t &scalars) {
    std::string out;
    render_into(node, style, idx, views, scalars, out);
    return out;
}
consteval std::string render_plain(std::meta::info node, const LeafStyle &style,
                                   std::string idx, size_t &views,
                                   size_t &scalars) {
    std::string out;
    render_plain_into(node, style, idx, views, scalars, out);
    return out;
}
consteval std::string render_indexed(std::meta::info node,
                                     const LeafStyle &style,
                                     const std::vector<std::string> &coord,
                                     size_t &views, size_t &scalars,
                                     const DirectRead &direct = {}) {
    std::string out;
    render_indexed_into(node, style, coord, views, scalars, out, direct);
    return out;
}

} // namespace tensor::detail
