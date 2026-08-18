#pragma once

// Every expression-layer diagnostic and the helpers that compute what a
// message names. One voice: complete sentences, naming the offender.

#include "../Core.h"
#include "Meta.h"

#include <cstddef>
#include <initializer_list>
#include <meta>
#include <string>
#include <string_view>
#include <vector>

namespace tensor::detail {

// A literal fill lists every element; a miscount is the one way to get it
// wrong, and the counts are what name it.
consteval std::string literal_count_error(size_t want, size_t got) {
    return "a literal fill lists exactly one value per element, row-major: "
           "this tensor holds " +
           to_string(want) + " elements but " + to_string(got) +
           " values were given.";
}

// A shapeless sampler broadcasts, so something else in its node has to say
// how many cells to draw. Scalars cannot: they broadcast too.
consteval std::string gen_unpinned_error() {
    return "a shapeless sampler has no extent of its own and nothing in this "
           "expression pins one — give it extents, rng::Uniform<float, 64>(), or "
           "combine it with an operand that has a shape.";
}

// The operand shape an error names: a tensor's extents, a scalar's own type.
template <typename E> consteval std::meta::info shape_of() {
    using D = std::remove_cvref_t<E>;
    if constexpr (TensorExpr<D>)
        return ^^typename D::extents_type;
    else
        return ^^D;
}

// Shared by the fold builders: are the listed axes/indices distinct, and
// which one repeats.
consteval bool axes_distinct(std::initializer_list<size_t> axes) {
    for (auto a = axes.begin(); a != axes.end(); ++a)
        for (auto b = a + 1; b != axes.end(); ++b)
            if (*a == *b)
                return false;
    return true;
}

consteval size_t first_duplicate(std::initializer_list<size_t> axes) {
    for (auto a = axes.begin(); a != axes.end(); ++a)
        for (auto b = a + 1; b != axes.end(); ++b)
            if (*a == *b)
                return *a;
    return 0;
}

// ── elementwise ─────────────────────────────────────────────────────────────

consteval std::string shape_error(std::vector<std::meta::info> shapes) {
    std::string s = "tensor shape mismatch: ";
    for (size_t i = 0; i < shapes.size(); ++i) {
        if (i != 0)
            s += "  vs  ";
        s += type_name(shapes[i]);
    }
    return s;
}

consteval std::string mixed_operand_error() {
    return "a plain tensor cannot meet an indexed read in one node — "
           "positional alignment would be ambiguous; subscript every tensor "
           "operand (scalars broadcast)";
}

consteval std::string indexed_element_error() {
    return "an index-bearing expression has no element access — its axes are "
           "its free indices; eval(...) it first";
}

consteval std::string order_error(size_t id) {
    return "eval<...> must name exactly the free indices, each once — " +
           ix_name(id) + " does not match";
}

consteval std::string order_on_plain_error() {
    return "eval<...> names free indices — this expression has none";
}

consteval std::string two_folds_error() {
    return "two folds in one expression are two passes — eval each fold "
           "separately";
}

consteval std::string fold_in_summand_error() {
    return "a fold inside another fold's summand is a second pass — eval "
           "the inner fold first";
}

// ── scan ────────────────────────────────────────────────────────────────────

consteval std::string scan_id_error() {
    return "scan: the axis is one placeholder (scan<ops::Add, m>(e)) or one "
           "axis number on a plain operand (scan<ops::Add, 1>(t)) — not both, "
           "and never several";
}

consteval std::string scan_id_not_free_error(size_t id) {
    return "scan: '" + ix_name(id) +
           "' is not a free index of the operand, so there is no axis to run "
           "along — subscript the operand with it, or name the axis it "
           "already has";
}

consteval std::string scan_in_tree_error() {
    return "a scan beside another scan, a fold or a scatter is a second pass "
           "— eval the scan first";
}

// A structured accumulator separates accumulate from merge; a scan's
// running value IS its output element, so only M = T ops scan.
consteval std::string scan_structured_error() {
    return "scan: this op accumulates through a structured state (a "
           "structured accumulator separates accumulate from merge), but a "
           "scan's running value is its own output element, so only plain "
           "ops scan — fold or scatter compute structured statistics.";
}

consteval std::string scan_below_subscript_error() {
    return "subscripting a scan recomputes its whole prefix per read — eval "
           "the scan first, then subscript the result (which is also how an "
           "exclusive scan is spelt: s[pad(m - 1_c, identity)])";
}

consteval std::string fold_below_subscript_error() {
    return "a fold cannot be subscripted into another read — eval it first";
}

consteval std::string fold_mixed_ids_error() {
    return "fold takes placeholders with an indexed operand or axis numbers "
           "with a plain one — never both in one list";
}

consteval std::string fold_degenerate_error() {
    return "every operand reads the same bare indices — drop the subscripts "
           "and name axes: fold<Op, 0, 2>(expr)";
}

consteval std::string fold_range_error() {
    return "a non-Add fold requires every read provably in range — wrap or "
           "clamp the subscript; a phantom out-of-range zero is not this "
           "op's identity";
}

consteval std::string scatter_arity_error() {
    return "scatter takes at least one destination and a value — "
           "scatter<i>(wrap<C>(cell[i]), q[i])";
}

consteval std::string scatter_ids_error() {
    return "scatter consumes placeholders, one or more, named in its "
           "template list — scatter<i>(wrap<C>(cell[i]), q[i]); axis numbers "
           "name no coordinate to scatter by";
}

consteval std::string scatter_placed_error() {
    return "every scatter argument but the last is a destination and must "
           "name where an out-of-range write goes — wrap<C>(…) is periodic, "
           "clamp<C>(…) holds the edge, drop<C>(…) discards the "
           "contribution; a runtime coordinate is never in range by "
           "construction";
}

consteval std::string scatter_epilogue_error() {
    return "an expression above a scatter that still reads free indices "
           "would evaluate the scatter once per cell — eval the scatter "
           "first, then combine";
}

consteval std::string scatter_element_error() {
    return "element access on a scatter would run the whole scatter per "
           "element — eval(expr) first";
}

consteval std::string scatter_order_error() {
    return "eval<Order…> names free indices, and a scatter's axes are its "
           "destinations — there is no placeholder to permute them by";
}

consteval std::string identity_permutation_error() {
    return "this subscript reads as a permutation but materializes as the "
           "identity (free axes order themselves by first appearance) — name "
           "the output order: eval<...>(...)";
}

consteval std::string map_vocabulary_error(std::meta::info op) {
    std::string s = "map<math::";
    s += std::meta::identifier_of(op);
    s += "> — the math vocabulary is spelt directly: math::";
    s += std::meta::identifier_of(op);
    s += "(...); map<f> lifts plain functions";
    return s;
}

// ── fold ────────────────────────────────────────────────────────────────────

consteval std::string reduce_axis_error(size_t axis, std::meta::info extents) {
    return "fold axis out of range: axis " + to_string(axis) + " of " +
           type_name(extents);
}

consteval std::string reduce_duplicate_error(size_t axis) {
    return "fold axis named twice: axis " + to_string(axis) +
           " — each axis folds once, and several axes belong in ONE fold "
           "(two nodes would be two passes over memory)";
}

consteval std::string contract_no_sum_error() {
    return "fold<> names nothing to fold — a rank-0 operand has no axes and "
           "no free indices";
}

consteval std::string contract_duplicate_error(size_t id) {
    return "fold: index " + ix_name(id) +
           " named twice — each folded index folds once";
}

consteval std::string contract_unpinned_error(size_t id) {
    return "fold: index " + ix_name(id) +
           " has no occurrence that pins its extent — a placeholder must "
           "appear alone in a subscript slot, or with unit coefficient in "
           "some axis (whose extent it then takes); a scaled-only index "
           "cannot infer one";
}

consteval std::string contract_extent_error(size_t id, size_t a, size_t b) {
    return "fold: the pinning occurrences of " + ix_name(id) +
           " disagree about its extent (" + to_string(a) + " vs " +
           to_string(b) + ")";
}

consteval std::string contract_separable_error() {
    return "fold: this fold is separable — no operand couples its folded "
           "indices, so it factors into cheaper independent folds; split it "
           "into two evals";
}

// ── bins ────────────────────────────────────────────────────────────────────

// bins computes its scale NB / (hi - lo) ONCE, at build time — which is
// only possible when the bounds are floating-point scalars.
consteval std::string bins_bounds_error(std::meta::info t) {
    return "bins takes floating-point SCALAR bounds — lo and hi were deduced "
           "as " +
           type_name(t) +
           ". Spell integer bounds 0.0f, 100.0f; per-cell bounds are the "
           "composition written out: math::Floor((x - lo) * (T(NB) / (hi - "
           "lo))).";
}

consteval std::string bins_zero_error() {
    return "bins<0> partitions [lo, hi) into no bins, so no value has an "
           "index — the bin count NB must be at least 1.";
}

// ── symbolic subscripts ─────────────────────────────────────────────────────

consteval std::string subscript_arity_error(size_t n, std::meta::info extents) {
    return "subscript count mismatch: " + to_string(n) + " subscript(s) for " +
           type_name(extents);
}

consteval std::string pad_count_error() {
    return "one pad value per read: a missed read yields a single value, so "
           "only one subscript of a read may carry pad(…) — decorate the "
           "other axes with wrap, clamp or zero";
}

template <typename Extents, DecMap... Ms> consteval bool subscript_named() {
    return (!map_open(Ms) && ...);
}

// ACC-L4: a coordinate has to be able to NAME every cell it can address.
// A float32 holds consecutive integers only to 2^24, so past that distinct
// cells silently merge — deposits land in the wrong bin and read-backs
// fetch the wrong cell, with nothing to report it. Caught where the extent
// and the coordinate type first meet.
consteval std::string index_capacity_error(std::meta::info coord, size_t ext,
                                           size_t cap) {
    return "this coordinate cannot address " + to_string(ext) +
           " cells: " + type_name(coord) + " represents consecutive integers "
           "exactly only up to " + to_string(cap) +
           ", so beyond that distinct cells collide silently — hold the "
           "coordinate in a wider or integral type (docs/numerical-"
           "contract.md ACC-L4)";
}

template <typename Extents, DecMap... Ms>
consteval std::string boundary_error() {
    constexpr DecMap ms[] = {Ms...};
    std::string s = "this read can leave the grid, so it must name its "
                    "boundary — wrap(…) reads periodically, clamp(…) holds "
                    "the edge, zero(…) reads absent; an undecorated "
                    "subscript must be a bare index or a constant. Offending "
                    "axis:";
    for (size_t k = 0; k < sizeof...(Ms); ++k)
        if (map_open(ms[k]))
            s += " " + to_string(k) + " (extent " +
                 to_string(Extents::static_extent(k)) + ")";
    return s;
}

// A constant subscript out of range is a bug, not a zero read — however
// it is spelt; a wrapped or clamped one resolves in range.
consteval bool const_out_of_range(DecMap m, size_t extent) {
    return map_affine(m) && lin_const(m.s[0].lin) &&
           (m.s[0].lin.off < 0 || size_t(m.s[0].lin.off) >= extent);
}

template <typename Extents, DecMap... Ms> consteval bool subscript_consts_ok() {
    constexpr DecMap ms[] = {Ms...};
    for (size_t k = 0; k < sizeof...(Ms); ++k)
        if (const_out_of_range(ms[k], Extents::static_extent(k)))
            return false;
    return true;
}

template <typename Extents, DecMap... Ms>
consteval std::string subscript_const_error() {
    constexpr DecMap ms[] = {Ms...};
    std::string s = "constant subscript out of range:";
    for (size_t k = 0; k < sizeof...(Ms); ++k)
        if (const_out_of_range(ms[k], Extents::static_extent(k))) {
            const int off = ms[k].s[0].lin.off;
            s += " " + to_string(size_t(off < 0 ? -off : off)) +
                 " on an axis of extent " +
                 to_string(Extents::static_extent(k));
        }
    return s;
}

} // namespace tensor::detail
