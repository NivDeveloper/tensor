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

// The placeholder spelling users wrote: i…n, or Ix<N> beyond them.
consteval std::string ix_name(size_t id) {
    constexpr std::string_view names[] = {"i", "j", "k", "l", "m", "n"};
    if (id < 6)
        return std::string(names[id]);
    return "Ix<" + to_string(id) + ">";
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
