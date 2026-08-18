#pragma once

// eval's dispatch — the free-index environment loop, the fold paths
// (interchange for bare summands, clamped per-cell nests otherwise, the
// per-cell epilogue), the plain elementwise loop — and use_threads.

#include "../Tensor.h"
#include "../detail/Pool.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <meta>
#include <type_traits>
#include <vector>

namespace tensor {

inline void use_threads(size_t n) {
    detail::pool_slot().reset(); // join old workers before spawning new
    if (n >= 2)
        detail::pool_slot() = std::make_unique<detail::ThreadPool>(n - 1);
}

template <auto... Order, AnyExpr E>
eval_return_t<E, Order...> eval(const E &e) {
    detail::sync_leaf_hosts(e); // residency: leaves defer, eval settles
    using B = std::remove_cvref_t<E>;
    static_assert(sizeof...(Order) == 0 || detail::index_bearing_v<B>,
                  detail::order_on_plain_error());
    static_assert(
        (detail::is_placeholder_v<std::remove_cvref_t<decltype(Order)>> &&
         ...),
        "eval's order takes the index placeholders (i, j, ...)");
    if constexpr (detail::scatter_count_v<B> == 1) {
        // A scatter's output cell comes from data, so cell c alone would
        // mean scanning every contribution to ask which ones land on it —
        // the very cost the node exists to remove. It is therefore never
        // dispatched per output cell: accumulate it whole, then apply the
        // epilogue at the merge, where each cell is visited once anyway.
        static_assert(sizeof...(Order) == 0, detail::scatter_order_error());
        static_assert(detail::ScatterNode<B> || !detail::index_bearing_v<B>,
                      detail::scatter_epilogue_error());
        eval_result_t<E> out;
        const auto &node = detail::scatter_node_of(e);
        using S = std::remove_cvref_t<decltype(node)>;
        // Bins hold the op's state; for a degenerate op that IS the output
        // element type, so the scatter still accumulates in place.
        using ElT = detail::element_t<std::remove_cvref_t<
            decltype(detail::scatter_value_of(node))>>;
        using Bin = detail::scatter_bin_t<typename S::op_type::op, ElT>;
        constexpr size_t cells = [] {
            size_t c = 1;
            for (size_t r = 0; r < S::rank; ++r)
                c *= S::extents_type::static_extent(r);
            return c;
        }();
        // Structured is the discriminator, NOT Bin == T: MinMax and ArgMax
        // finish into their own state type, and finish must still run.
        constexpr bool structured =
            detail::Structured<typename S::op_type::op, ElT>;
        if constexpr (detail::ScatterNode<B> && !structured) {
            detail::eval_scatter<S>(node, out.data());
        } else if constexpr (detail::ScatterNode<B>) {
            // A structured op finishes its states into the output.
            std::vector<Bin> acc(cells);
            detail::eval_scatter<S>(node, acc.data());
            for (size_t c = 0; c < out.size(); ++c)
                out.data()[c] =
                    detail::acc_finish<typename S::op_type::op, ElT>(acc[c]);
        } else {
            std::vector<Bin> acc(cells);
            detail::eval_scatter<S>(node, acc.data());
            using T = typename eval_result_t<E>::type;
            for (size_t c = 0; c < out.size(); ++c)
                out.data()[c] = T(detail::eval_epilogue(e, c, acc.data()));
        }
        return out;
    } else if constexpr (detail::index_bearing_v<B>) {
        // The free-index space IS the output: one env per output cell,
        // free ids decomposed row-major in first-appearance order — or in
        // the named order when eval<Order...> spells one.
        static constexpr auto base = detail::free_plan(std::meta::dealias(^^B));
        static constexpr auto unpinned =
            detail::first_unpinned(std::meta::dealias(^^B), {});
        static_assert(unpinned == detail::index_slots,
                      detail::contract_unpinned_error(unpinned));
        static constexpr auto mismatch = [] {
            if constexpr (sizeof...(Order) == 0)
                return detail::index_slots;
            else
                return detail::order_mismatch(base,
                                              {detail::order_id<Order>()...});
        }();
        static_assert(mismatch == detail::index_slots,
                      detail::order_error(mismatch));
        static constexpr auto plan = [] {
            if constexpr (sizeof...(Order) == 0)
                return base;
            else
                return detail::ordered_plan(base,
                                            {detail::order_id<Order>()...});
        }();
        // A lone permuted bare read materializes as the identity — demand
        // an explicit order rather than surprise.
        static constexpr bool identity_trap = [] {
            if constexpr (sizeof...(Order) > 0 || !detail::is_indexed_v<B>)
                return false;
            else {
                int prev = -1;
                bool ascending = true;
                bool seen[detail::index_slots] = {};
                for (auto m : B::maps) {
                    const int b = detail::map_bare_slot(m);
                    if (b < 0 || seen[size_t(b)])
                        return false; // affine or diagonal: a real read
                    seen[size_t(b)] = true;
                    ascending &= (b > prev);
                    prev = b;
                }
                return !ascending;
            }
        }();
        static_assert(!identity_trap, detail::identity_permutation_error());
        if constexpr (detail::scan_count_v<B> == 1) {
            // A scan KEEPS its index, so the output is this same free-index
            // space; what differs is that one axis is walked in ascending
            // order with a running accumulator. Rows are independent and
            // parallelize; the axis itself never splits, which is why the
            // answer is bit-identical at any thread count and why the order
            // can be specified rather than left open as a fold's is.
            const auto &node = detail::scan_node_of(e);
            using S = std::remove_cvref_t<decltype(node)>;
            using Op = typename S::op_type::op;
            using A = std::remove_cvref_t<typename S::type>;
            using T = typename eval_result_t<E, Order...>::type;
            static constexpr size_t sid = S::op_type::scanned;
            static constexpr size_t axis = [] {
                for (size_t a = 0; a < plan.n; ++a)
                    if (plan.id[a] == sid)
                        return a;
                return detail::index_slots; // unreachable: the id is free
            }();
            // Row-major strides over the (possibly reordered) output plan,
            // so a scanned axis anywhere writes to the right cell.
            static constexpr auto stride = [] {
                std::array<size_t, detail::index_slots> s{};
                size_t acc = 1;
                for (size_t a = plan.n; a-- > 0;) {
                    s[a] = acc;
                    acc *= plan.ext[a];
                }
                return s;
            }();
            static constexpr size_t ext = plan.ext[axis];
            static constexpr size_t rows =
                eval_result_t<E, Order...>::element_count / ext;

            eval_result_t<E, Order...> t;
            auto *out = t.data();
            const auto &[... kids] = node;
            const auto &child = kids...[0];
            detail::parallel_for(rows, [&](size_t r0, size_t r1) {
                for (size_t r = r0; r < r1; ++r) {
                    detail::IxEnv env{};
                    size_t rem = r, base = 0;
                    for (size_t a = plan.n; a-- > 0;)
                        if (a != axis) {
                            const size_t v = rem % plan.ext[a];
                            rem /= plan.ext[a];
                            env[plan.id[a]] = std::ptrdiff_t(v);
                            base += v * stride[a];
                        }
                    A acc = Op::template identity<A>();
                    for (size_t k = 0; k < ext; ++k) {
                        env[sid] = std::ptrdiff_t(k);
                        acc = Op{}(acc, detail::eval_indexed(child, env));
                        out[base + k * stride[axis]] =
                            T(detail::eval_scan_epilogue(e, env, acc));
                    }
                }
            });
            return t;
        } else if constexpr (detail::ContractNode<B>) {
            // A fold at the root: the v1 contraction machinery, op-generic.
            if constexpr (sizeof...(Order) > 0) {
                // A named layout writes through the ordered environment,
                // each cell folding its own strict chain.
                eval_result_t<E, Order...> t;
                auto *out = t.data();
                detail::parallel_for(t.size(), [&](size_t begin, size_t end) {
                    for (size_t q = begin; q < end; ++q) {
                        detail::IxEnv env{};
                        size_t rem = q;
                        for (size_t a = plan.n; a-- > 0;) {
                            env[plan.id[a]] =
                                std::ptrdiff_t(rem % plan.ext[a]);
                            rem /= plan.ext[a];
                        }
                        out[q] = detail::eval_indexed(e, env);
                    }
                });
                return t;
            } else if constexpr (B::rank == 0) {
                return detail::eval_node(e, size_t{0});
            } else {
                eval_result_t<E> t;
                // The interchange streams the operands, which is a win only
                // when the fold would otherwise read them strided; where the
                // per-cell fold is already contiguous it is a pessimization.
                if constexpr (detail::all_maps_bare(
                                  detail::children_of(^^B)[0]) &&
                              !detail::fold_reads_contiguously(
                                  detail::children_of(^^B)[0],
                                  std::vector<size_t>(
                                      B::op_type::summed.begin(),
                                      B::op_type::summed.end()))) {
                    // Bare summand: summed loops outside, operands stream;
                    // chunks split the first free axis into output rows.
                    using Op = typename B::op_type;
                    using Fold = typename Op::op;
                    using T = typename B::type;
                    const auto &[summand] = e;
                    using S = std::remove_cvref_t<decltype(summand)>;
                    // The accumulator keys on the SUMMAND's element type —
                    // the node's own is finish's result for a structured op.
                    using ElT = std::remove_cvref_t<typename S::type>;
                    constexpr bool structured = detail::Structured<Fold, ElT>;
                    static constexpr auto cplan = detail::contract_plan(
                        ^^S, std::vector<size_t>(Op::summed.begin(),
                                                 Op::summed.end()));
                    // This path's accumulator is the OUTPUT cell itself —
                    // the interchange puts the summed loops outside, so a
                    // cell's terms arrive spread across the traversal and
                    // there is nowhere local to hold them. Accumulating in
                    // accumulator_t (ACC-G9) therefore costs a buffer, and
                    // only where that type differs from the element one.
                    // The two arms are spelt out rather than sharing one
                    // parallel_for: the buffer types differ, and folding
                    // them into one pointer makes GCC check the discarded
                    // branch of the `if constexpr` against the wrong type.
                    using Acc = detail::fold_accumulator_t<Fold, ElT>;
                    constexpr size_t rows = B::extents_type::static_extent(0);
                    constexpr size_t per_row =
                        eval_result_t<E>::element_count / rows;
                    constexpr size_t row_cost = cplan.fold_count * per_row;
                    constexpr size_t grain =
                        std::max<size_t>(1, detail::eval_grain / row_cost);
                    // A structured op always takes the wide arm — even when
                    // M == R, finish must still run over every cell.
                    if constexpr (std::is_same_v<Acc, T> && !structured) {
                        std::fill_n(t.data(), t.size(),
                                    Fold::template identity<T>());
                        detail::parallel_for(
                            rows,
                            [&](size_t r0, size_t r1) {
                                detail::IxEnv env{};
                                detail::contract_streamed<0, B>(
                                    summand, env, t.data(), size_t{0}, r0, r1);
                            },
                            grain);
                    } else {
                        // A structured axis fold holds the whole output as
                        // states before finish (Welford: 24 B/cell) — the
                        // same shape as this f64 arm, bigger constant.
                        std::vector<Acc> wide(
                            t.size(), detail::acc_identity<Fold, ElT>());
                        detail::parallel_for(
                            rows,
                            [&](size_t r0, size_t r1) {
                                detail::IxEnv env{};
                                detail::contract_streamed<0, B>(
                                    summand, env, wide.data(), size_t{0}, r0,
                                    r1);
                                // chunks own disjoint output rows, so each
                                // narrows its own without synchronization
                                for (size_t c = r0 * per_row;
                                     c < r1 * per_row; ++c)
                                    t.data()[c] =
                                        detail::acc_finish<Fold, ElT>(wide[c]);
                            },
                            grain);
                    }
                } else {
                    // A fold's output cell costs fold_count reads, so the
                    // chunk that carries a grain's worth of work is that
                    // many cells smaller.
                    using Op = typename B::op_type;
                    const auto &[summand] = e;
                    using S = std::remove_cvref_t<decltype(summand)>;
                    static constexpr auto cplan = detail::contract_plan(
                        ^^S, std::vector<size_t>(Op::summed.begin(),
                                                 Op::summed.end()));
                    auto *out = t.data();
                    detail::parallel_for(
                        t.size(),
                        [&](size_t begin, size_t end) {
                            for (size_t q = begin; q < end; ++q)
                                out[q] = detail::eval_node(e, q);
                        },
                        std::max<size_t>(1, detail::eval_grain /
                                                std::max<size_t>(
                                                    1, cplan.fold_count)));
                }
                return t;
            }
        } else if constexpr (plan.n == 0) {
            // Every subscript constant: the read is one element.
            detail::IxEnv env{};
            return detail::eval_indexed(e, env);
        } else {
            eval_result_t<E, Order...> t;
            auto *out = t.data();
            // A pure-shift tree splits: a span whose coordinates are all
            // interior runs as contiguous constant-offset reads, and every
            // edge keeps the general per-cell path.
            static constexpr auto shift =
                detail::shift_plan(std::meta::dealias(^^B), plan);
            constexpr size_t last = plan.n - 1;
            constexpr size_t inner = plan.ext[last];
            constexpr size_t rows =
                eval_result_t<E, Order...>::element_count / inner;

            // One row's span [c0, c1): edge, interior, edge.
            auto run_span = [&](detail::IxEnv env, bool interior, size_t base,
                                size_t c0, size_t c1) {
                const auto clip = [&](size_t x) {
                    return x < c0 ? c0 : (x > c1 ? c1 : x);
                };
                const size_t mid_lo = interior ? clip(shift.lo[last]) : c1;
                const size_t mid_hi = interior ? clip(shift.hi[last]) : c1;
                const auto cell = [&](size_t c) {
                    env[plan.id[last]] = std::ptrdiff_t(c);
                    out[base + c] = detail::eval_indexed(e, env);
                };
                for (size_t c = c0; c < mid_lo; ++c)
                    cell(c);
                if constexpr (shift.ok)
                    for (size_t c = mid_lo; c < mid_hi; ++c)
                        out[base + c] =
                            detail::eval_shifted(e, std::ptrdiff_t(base + c));
                for (size_t c = mid_hi; c < c1; ++c)
                    cell(c);
            };

            if constexpr (plan.n == 1) {
                // One row: the chunks divide it, so each carries its own
                // share of the interior.
                detail::parallel_for(inner, [&](size_t b, size_t c) {
                    run_span(detail::IxEnv{}, shift.ok, size_t{0}, b, c);
                });
            } else {
                detail::parallel_for(
                    rows,
                    [&](size_t r0, size_t r1) {
                        for (size_t r = r0; r < r1; ++r) {
                            detail::IxEnv env{};
                            bool interior = shift.ok;
                            size_t rem = r;
                            for (size_t a = last; a-- > 0;) {
                                const size_t c = rem % plan.ext[a];
                                rem /= plan.ext[a];
                                env[plan.id[a]] = std::ptrdiff_t(c);
                                interior = interior && c >= shift.lo[a] &&
                                           c < shift.hi[a];
                            }
                            run_span(env, interior, r * inner, size_t{0},
                                     inner);
                        }
                    },
                    std::max<size_t>(1, detail::eval_grain / inner));
            }
            return t;
        }
    } else if constexpr (std::remove_cvref_t<E>::rank == 0) {
        return detail::eval_node(e, size_t{0});
    } else {
        eval_result_t<E> t;
        auto *out = t.data();
        detail::parallel_for(t.size(), [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
                out[i] = detail::eval_node(e, i);
        });
        return t;
    }
}

} // namespace tensor
