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
    if constexpr (detail::index_bearing_v<B>) {
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
        if constexpr (detail::ContractNode<B>) {
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
                    static constexpr auto cplan = detail::contract_plan(
                        ^^S, std::vector<size_t>(Op::summed.begin(),
                                                 Op::summed.end()));
                    std::fill_n(t.data(), t.size(),
                                Fold::template identity<T>());
                    constexpr size_t rows = B::extents_type::static_extent(0);
                    constexpr size_t row_cost =
                        cplan.fold_count *
                        (eval_result_t<E>::element_count / rows);
                    detail::parallel_for(
                        rows,
                        [&](size_t r0, size_t r1) {
                            detail::IxEnv env{};
                            detail::contract_streamed<0, B>(
                                summand, env, t.data(), size_t{0}, r0, r1);
                        },
                        std::max<size_t>(1, detail::eval_grain / row_cost));
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
