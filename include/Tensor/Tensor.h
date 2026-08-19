#pragma once

// The Tensor container — the library's entry point:
//
//   Tensor<float, 4, 3> t;                // 4x3 float, uninitialized
//   Tensor<float, 4, 3> u(f);             // filled from f(i, j)
//   t[i, j] = v;                          // element access (variadic)
//   auto c = eval(a + b * 2.0f);          // one fused pass → deduced Tensor
//   auto m = eval(map<my_fn>(a, 0.5f));   // any plain function composes
//   auto s = eval(fold(a));               // fold every axis → the scalar
//   use_threads(8);                       // opt-in: eval on 8 threads
//
// Definitions live in Tensor/*.h, included at the bottom.

#include "Core.h"
#include "Expr.h"
#include "Introspect.h"
#include "detail/Tensor.h"

#include <concepts>
#include <cstddef>
#include <mdspan>
#include <memory>
#include <meta>
#include <tuple>
#include <type_traits>

namespace tensor {

// A Tensor's device-residency seam: when a device backend evaluates an
// expression, it may cache uploads and park results behind this interface
// (the GPU eval header owns the only implementation — CPU-only builds
// never create one). A Tensor holding such a shadow must be destroyed
// before the device that created it.
struct DeviceStorage {
    virtual ~DeviceStorage() = default;
    virtual void download(void *dst, size_t bytes) = 0;
};

template <typename T, size_t... Extents> class Tensor {
  public:
    using type = T;
    using extents_type = std::extents<size_t, Extents...>;
    using span_type = std::mdspan<T, extents_type>;

    static constexpr size_t rank = sizeof...(Extents);
    static constexpr size_t element_count = (size_t{1} * ... * Extents);
    static constexpr size_t byte_size = element_count * sizeof(T);

    // ── Lifecycle ────────────────────────────────────────────────
    Tensor() = default;

    // Index fill: one index per axis, row-major visiting order. In-class
    // body: the constraint names rank, which an out-of-line definition
    // cannot repeat token-identically.
    //   Tensor<float, 4, 3> t([](size_t i, size_t j) { return float(i * j); });
    template <typename F>
        requires detail::IndexFill<F, T, rank>
    explicit Tensor(F &&f) {
        for (size_t i = 0; i < element_count; ++i)
            data_[i] = std::apply(f, detail::to_multi_index<extents_type>(i));
    }

    // Literal fill: exactly one value per element, row-major. In-class for
    // the same reason as the index fill — the constraint names element_count.
    //   Tensor<float, 2, 3> a{1, 2, 3, 4, 5, 6};
    // A miscounted LIST is a candidate so it reaches the assert below rather
    // than failing as "no matching function"; a lone value stays a candidate
    // only where the tensor really holds one element, so nothing scalar
    // becomes convertible to a bigger tensor.
    template <typename... Vs>
        requires((sizeof...(Vs) >= 2 || sizeof...(Vs) == element_count) &&
                 (std::convertible_to<Vs, T> && ...))
    Tensor(Vs... vs) {
        if constexpr (sizeof...(Vs) != element_count) {
            static_assert(
                sizeof...(Vs) == element_count,
                detail::literal_count_error(element_count, sizeof...(Vs)));
        } else {
            size_t k = 0;
            ((data_[k++] = static_cast<T>(vs)), ...);
        }
    }

    // Construction and assignment from an expression both route through
    // these deletes.
    template <TensorExpr E>
    Tensor(const E &) =
        delete("a tensor is materialized from an expression only with "
               "eval(expr) — neither construction nor assignment evaluates");
    template <IndexedExpr E>
    Tensor(const E &) =
        delete("a tensor is materialized from an expression only with "
               "eval(expr) — neither construction nor assignment evaluates");
    // A tensor of another element type or shape is also a TensorExpr, but
    // the mistake behind it is a promotion, not a missing eval — so it gets
    // its own sentence rather than the one above accusing the user of
    // skipping eval they wrote.
    template <typename U, size_t... Es>
    Tensor(const Tensor<U, Es...> &) =
        delete("this tensor's element type or extents differ from the "
               "source's, and nothing converts between tensor types — a bare "
               "double literal (0.1) widens a float expression to double; "
               "spell it 0.1f, or take the widened result with auto");

    // ── Element access (through mdspan) ──────────────────────────
    template <std::convertible_to<size_t>... Idx>
        requires(sizeof...(Idx) == rank)
    T &operator[](Idx... idx) {
        return span()[idx...];
    }
    template <std::convertible_to<size_t>... Idx>
        requires(sizeof...(Idx) == rank)
    const T &operator[](Idx... idx) const {
        return span()[idx...];
    }

    T &operator[](Index<rank> idx);
    const T &operator[](Index<rank> idx) const;

    // Symbolic subscript, A[i, j] / f[zero(i + j - k)]: an indexed operand for
    // the indexed grammar, not an element access.
    template <detail::SubscriptArg... Sub>
        requires(detail::SubscriptTerm<Sub> || ...)
    constexpr auto operator[](Sub... sub) const;

    // ── Views / metadata ─────────────────────────────────────────
    // The const view is the expression-leaf mechanism and NEVER syncs —
    // leaves defer, so a resident result can feed the next device eval
    // without a round trip. Every other host access syncs a stale host
    // first; the non-const ones then drop the shadow, making host memory
    // the sole authority.
    TensorView<T, Extents...> view();
    TensorView<const T, Extents...> view() const;

    span_type span();
    std::mdspan<const T, extents_type> span() const;

    T *data();
    const T *data() const;

    static constexpr size_t size();
    static constexpr size_t extent(size_t r);

  private:
    void sync_host() const; // a read: download if stale, keep the shadow
    void drop_shadow();     // a write may follow: sync, then host-only

    std::unique_ptr<T[]> data_ =
        std::make_unique_for_overwrite<T[]>(element_count);
    // mutable: caching an upload from a const leaf is still a read.
    mutable detail::ShadowSlot shadow_;
};

// ── eval: materialize an expression into a deduced owning Tensor ────────────
template <AnyExpr E, auto... Order>
using eval_result_t = typename[:detail::eval_result_of<E, Order...>():];

// A rank-0 result evaluates to the VALUE — a one-element Tensor would mean
// a heap allocation per reduction.
template <AnyExpr E, auto... Order>
using eval_return_t =
    std::conditional_t<detail::rank_of<std::remove_cvref_t<E>>() == 0,
                       typename std::remove_cvref_t<E>::type,
                       eval_result_t<E, Order...>>;

// n >= 2: every later eval runs on n threads (caller + n-1 workers),
// results bit-identical to serial; n <= 1 is the serial default.
// Externally synchronized with eval; a map<f> fn must be thread-safe.
void use_threads(size_t n);

// eval(e) materializes with output axes in first-appearance order for an
// index-bearing tree; eval<i,j>(e) names the order explicitly — the one
// place layout is physical.
template <auto... Order, AnyExpr E>
eval_return_t<E, Order...> eval(const E &e);

// A range-tagged coordinate evaluates as the coordinate it is: the labels
// themselves, numpy's digitize. The tag only ever mattered to a policy.
template <auto... Order, detail::RangedExpr E>
eval_return_t<typename std::remove_cvref_t<E>::coord_type, Order...>
eval(const E &e);

// bins<NB>(i) names a destination, not a value: the extent it divides is
// the one some READ of i pins, and alone there is no read.
template <auto... Order, typename E>
    requires detail::is_ix_token_v<std::remove_cvref_t<E>>
auto eval(const E &) = delete(
    "a grouping of an index names a scatter destination — it groups an axis "
    "rather than producing a value, and alone nothing pins how far the axis "
    "runs. Use it as a destination: scatter<i>(bins<NB>(i), v[i])");

} // namespace tensor

// The definitions.
#include "Tensor/Container.h" // IWYU pragma: export
#include "Tensor/Eval.h"      // IWYU pragma: export
