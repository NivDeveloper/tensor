#pragma once

// The symbolic-index machinery: affine forms, the indexed leaf, and the
// traits Core.h's concepts are defined over.

#include "Utils.h"

#include <array>
#include <complex>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace tensor {

namespace detail {
// What a generator leaf computes from its coordinate. The deterministic
// kinds read their typed parameters (Fill and Iota one, LinSpace both); the
// sampler kinds read neither, carrying a 64-bit key instead.
enum class GenKind : unsigned char {
    Fill,
    Iota,
    LinSpace,
    Uniform,
    Normal,
    Stream // the per-cell Rng a sample<f> function draws from
};
} // namespace detail

template <typename T, size_t... Extents> class Tensor;
template <typename Op, typename... Cs> struct Expr;
template <detail::GenKind K, typename T, size_t... Extents> struct Generator;
struct DeviceStorage; // Tensor.h — a Tensor's device-residency seam

namespace detail {

struct ShadowSlot; // detail/Tensor.h — the per-Tensor residency state

inline constexpr size_t index_slots = 8; // Ix<0> … Ix<7>

// One affine form, Σ c[p]·Ix<p> + off — structural, so it can be an NTTP.
struct Lin {
    int c[index_slots]{};
    int off = 0;
    constexpr bool operator==(const Lin &) const = default;
};

// ── decoration chains ───────────────────────────────────────────────────────
// A per-read boundary pipeline: stages resolve innermost-first, each adding
// its affine part to the running coordinate and then applying its policy.
// None, Zero and Pad all miss rather than resolve; the read then yields its
// fill — the element type's zero, or the value pad(…) supplied.
enum class Policy : unsigned char { None, Wrap, Clamp, Zero, Pad };

struct Stage {
    Lin lin{};
    Policy pol = Policy::None;
    constexpr bool operator==(const Stage &) const = default;
};

inline constexpr size_t max_stages = 4;

// `data` marks an axis whose coordinate is a runtime expression carried on
// the node (a gather) rather than an affine form of placeholders. Trailing,
// so the struct stays structural and every aggregate initializer survives.
struct DecMap {
    Stage s[max_stages]{};
    size_t n = 0;
    bool data = false;
    constexpr bool operator==(const DecMap &) const = default;
};

} // namespace detail

template <size_t N> struct Ix;
template <detail::Lin L> struct IxForm;
template <detail::DecMap M> struct IxDec;
template <detail::DecMap M, typename V> struct IxPad;
template <detail::Policy P, typename C> struct IxData;
template <detail::Policy P, typename C, typename V> struct IxDataPad;

namespace detail {

template <typename S> inline constexpr bool is_placeholder_v = false;
template <size_t N> inline constexpr bool is_placeholder_v<Ix<N>> = true;
template <typename S> inline constexpr bool is_ix_form_v = false;
template <Lin L> inline constexpr bool is_ix_form_v<IxForm<L>> = true;
template <typename S> inline constexpr bool is_ix_dec_v = false;
template <DecMap M> inline constexpr bool is_ix_dec_v<IxDec<M>> = true;
template <typename S> inline constexpr bool is_ix_pad_v = false;
template <DecMap M, typename V> inline constexpr bool is_ix_pad_v<IxPad<M, V>> = true;
template <typename S> inline constexpr bool is_ix_data_v = false;
template <Policy P, typename C>
inline constexpr bool is_ix_data_v<IxData<P, C>> = true;
template <Policy P, typename C, typename V>
inline constexpr bool is_ix_data_v<IxDataPad<P, C, V>> = true;

// SubscriptArg admits raw integers so they can become "spell 0 as 0_c"
// advice instead of a missing-overload error.
template <typename S>
concept IndexTerm = is_placeholder_v<std::remove_cvref_t<S>> ||
                    is_ix_form_v<std::remove_cvref_t<S>>;
template <typename S>
concept DecoratedTerm =
    is_ix_dec_v<std::remove_cvref_t<S>> || is_ix_pad_v<std::remove_cvref_t<S>>;
// Declared here, specialized below: DataTerm needs it, and an undecorated
// coordinate has to reach make_indexed's boundary lint rather than failing
// as a missing overload.
template <typename T> inline constexpr bool has_free_index_v = false;

// A gathered coordinate — a decorated one, or a bare expression that has
// not named its boundary yet and must reach the lint to be told so.
// Deliberately NOT a DecoratedTerm: the affine displacement operators are
// constrained on that concept, and a data coordinate takes no part in the
// index algebra — the shift belongs in the expression, `clamp(cell[i] - 1)`,
// where it is ordinary arithmetic.
template <typename S>
concept DataTerm = is_ix_data_v<std::remove_cvref_t<S>> ||
                   has_free_index_v<std::remove_cvref_t<S>>;
// A subscript resolved entirely at compile time — what the consteval
// decorations take. Named apart from SubscriptTerm because a data term is
// also a subscript term, and the two overload sets must not collide.
template <typename S>
concept AffineTerm = IndexTerm<S> || DecoratedTerm<S>;
template <typename S>
concept SubscriptTerm = AffineTerm<S> || DataTerm<S>;
template <typename S>
concept SubscriptArg = SubscriptTerm<S> || std::integral<std::remove_cvref_t<S>>;

consteval Lin unit_lin(size_t id) {
    Lin l{};
    l.c[id] = 1;
    return l;
}
consteval Lin const_lin(int off) {
    Lin l{};
    l.off = off;
    return l;
}
consteval Lin add_lin(Lin a, Lin b, int s) {
    Lin r{};
    for (size_t p = 0; p < index_slots; ++p)
        r.c[p] = a.c[p] + s * b.c[p];
    r.off = a.off + s * b.off;
    return r;
}
consteval Lin scale_lin(Lin a, int s) {
    Lin r{};
    for (size_t p = 0; p < index_slots; ++p)
        r.c[p] = s * a.c[p];
    r.off = s * a.off;
    return r;
}
consteval bool lin_const(Lin l) {
    for (size_t p = 0; p < index_slots; ++p)
        if (l.c[p] != 0)
            return false;
    return true;
}
// The single unit slot of a bare form (one coefficient 1, no offset), or -1.
consteval int bare_slot(Lin l) {
    if (l.off != 0)
        return -1;
    int found = -1;
    for (size_t p = 0; p < index_slots; ++p) {
        if (l.c[p] == 0)
            continue;
        if (l.c[p] != 1 || found != -1)
            return -1;
        found = int(p);
    }
    return found;
}
template <typename S> consteval Lin term_lin() {
    using D = std::remove_cvref_t<S>;
    if constexpr (is_placeholder_v<D>)
        return unit_lin(D::id);
    else
        return D::lin;
}

// ── the chain algebra behind wrap/clamp/zero ────────────────────────────────
consteval DecMap lift_lin(Lin l) {
    DecMap m{};
    m.s[0] = {l, Policy::None};
    m.n = 1;
    return m;
}

// A gathered axis: one stage carrying only the policy. Its Lin is empty and
// stays empty — the coordinate is a value, not a form.
consteval DecMap data_map(Policy p) {
    DecMap m{};
    m.s[0] = {Lin{}, p};
    m.n = 1;
    m.data = true;
    return m;
}

// Any subscript term as a chain: decorated forms carry theirs, data forms
// carry a policy-only one, affine forms lift to one open (None) stage. An
// UNDECORATED expression lands on Policy::None, which map_open reports so
// the boundary lint names the four decorations.
template <typename S> consteval DecMap term_map() {
    using D = std::remove_cvref_t<S>;
    if constexpr (is_ix_dec_v<D> || is_ix_pad_v<D> || is_ix_data_v<D>)
        return D::map;
    else if constexpr (has_free_index_v<D>)
        return data_map(Policy::None);
    else
        return lift_lin(term_lin<S>());
}

// Close the open last stage with a policy, or start a new stage.
consteval DecMap decorate(DecMap m, Policy p) {
    if (m.s[m.n - 1].pol == Policy::None)
        m.s[m.n - 1].pol = p;
    else
        m.s[m.n++] = {Lin{}, p};
    return m;
}

// Fold a displacement into the open last stage, or start a new one.
consteval DecMap displace(DecMap m, Lin d, int s) {
    if (m.s[m.n - 1].pol == Policy::None)
        m.s[m.n - 1].lin = add_lin(m.s[m.n - 1].lin, d, s);
    else
        m.s[m.n++] = {scale_lin(d, s), Policy::None};
    return m;
}

// A gathered axis: its coordinate is a value, so its Lin is empty and means
// nothing. Excluding it from the two classifiers below is what keeps every
// affine analysis in the library from reading that empty Lin as a
// coordinate — see the map_open and map_in_range notes.
consteval bool map_data(DecMap m) { return m.data; }

// The classifications the fast paths, extent pinning, and guard
// collection key on: a still-undecorated single-stage map (v1's affine
// read), and its bare special case (one open coefficient-1 read).
consteval bool map_plain(DecMap m) {
    return m.n == 1 && m.s[0].pol == Policy::None && !map_data(m);
}
consteval int map_bare_slot(DecMap m) {
    return map_plain(m) ? bare_slot(m.s[0].lin) : -1;
}
consteval bool map_bare(DecMap m) { return map_bare_slot(m) >= 0; }

// One affine stage with no coordinate transform: the coordinate IS the
// form, and a miss yields zero. zero(…) states it; a bare or constant
// form cannot miss, which is why it may stay undecorated. A data axis is
// excluded: its miss is not a property of any form, so calling it affine
// would let map_in_range pass a zero(…) gather under a Max fold and let
// covered() elide a bounds check on Lin equality every gather shares.
consteval bool map_affine(DecMap m) {
    return m.n == 1 && !map_data(m) &&
           (m.s[0].pol == Policy::None || m.s[0].pol == Policy::Zero ||
            m.s[0].pol == Policy::Pad);
}

// A read whose miss yields a user-supplied fill rather than the zero: it
// carries a runtime value, so its terms never vanish and never skip.
consteval bool map_padded(DecMap m) {
    for (size_t q = 0; q < m.n; ++q)
        if (m.s[q].pol == Policy::Pad)
            return true;
    return false;
}

// An undecorated read that can leave the grid — what the boundary lint
// rejects, chains with an open tail included.
consteval bool map_open(DecMap m) {
    if (m.s[m.n - 1].pol != Policy::None)
        return false;
    return !map_bare(m) && !(map_plain(m) && lin_const(m.s[0].lin));
}

// ── write-side boundaries ───────────────────────────────────────────────────
// A read resolves against the operand's own axis; a scatter writes into an
// axis of nothing, so the extent is stated. No None: a runtime coordinate is
// never in range by construction, so the policy is mandatory.
enum class Place : unsigned char { Wrap, Clamp, Drop };

// One scatter destination: the coordinate expression, how it resolves, and
// the extent it resolves into.
template <Place P, size_t E, typename C> struct Placed {
    using coord_type = C;
    using type = std::remove_cvref_t<typename C::type>;
    static constexpr Place policy = P;
    static constexpr size_t ext = E;
    C c;
};

template <typename T> inline constexpr bool is_placed_v = false;
template <Place P, size_t E, typename C>
inline constexpr bool is_placed_v<Placed<P, E, C>> = true;

// Resolve a coordinate through its policy; false means the contribution is
// dropped, which only Drop can be. Signed: an out-of-range write coordinate
// is routinely negative and the guard must be able to say so.
template <Place P, size_t E>
constexpr bool place_coord(std::ptrdiff_t v, size_t &out) {
    constexpr auto ext = std::ptrdiff_t(E);
    if constexpr (P == Place::Wrap) {
        auto m = v % ext;
        out = size_t(m < 0 ? m + ext : m);
        return true;
    } else if constexpr (P == Place::Clamp) {
        out = size_t(v < 0 ? 0 : (v >= ext ? ext - 1 : v));
        return true;
    } else {
        if (v < 0 || v >= ext)
            return false;
        out = size_t(v);
        return true;
    }
}

// A node's children: the flat slot aggregate from Utils.h, not a
// std::tuple — see the note there.
template <typename... Cs> using Kids = Slots<Cs...>;

// An axis whose coordinate is affine, so it carries no expression.
struct NoCoord {};

// The indexed leaf, A[i,j]: one decorated map per axis of the operand —
// the terminal of every index-bearing expression. fill is what a missed
// read yields: the element type's zero, or pad(…)'s value. `d` holds one
// slot per AXIS — the coordinate expression of a gathered axis, NoCoord
// otherwise — so the axis index is the slot index and nothing maps between
// them.
//
// NOTE for future walkers: children_of() inlines a Slots member's
// arguments, so it reports [E, Ds…, type] here. Every walker tests
// is_indexed first and never reaches it; one that forgets would silently
// see the coordinates as children.
template <typename E, typename Data, DecMap... Ms> struct Indexed {
    using operand_type = E;
    using data_type = Data;
    using type = std::remove_cvref_t<typename E::type>;
    static constexpr std::array<DecMap, sizeof...(Ms)> maps{Ms...};
    static constexpr bool padded = (map_padded(Ms) || ...);
    static constexpr bool gathered = (map_data(Ms) || ...);
    E e;
    Data d{};
    type fill{};
    type operator[](size_t) const; // declared only: the element-type probe
};

template <typename T> inline constexpr bool is_indexed_v = false;
template <typename E, typename Data, DecMap... Ms>
inline constexpr bool is_indexed_v<Indexed<E, Data, Ms...>> = true;

// A generator leaf: storage-free, its value a closed-form function of the
// coordinate. Shaped like a view (type/extents_type/rank + operator[]), so
// the CPU walkers need no branch for it; only the sites that bind BUFFERS
// do. Type-level twin: is_generator_type (detail/Tree.h).
template <typename T> inline constexpr bool is_generator_v = false;
template <GenKind K, typename T, size_t... Es>
inline constexpr bool is_generator_v<Generator<K, T, Es...>> = true;

// The shapeless spelling: an operand that pins no shape but is not a
// scalar, so it broadcasts and still varies per cell.
template <typename T> inline constexpr bool is_shapeless_generator_v = false;
template <GenKind K, typename T>
inline constexpr bool is_shapeless_generator_v<Generator<K, T>> = true;

// A sampler draws from the counter-based stream its key names; every other
// kind is a closed-form function of the coordinate alone.
consteval bool gen_is_sampler(GenKind k) {
    return k == GenKind::Uniform || k == GenKind::Normal ||
           k == GenKind::Stream;
}

// How many scalar slots a kind claims — what the emitter declares and the
// host packs, kept in one place so the two cannot disagree. A sampler
// claims two: the halves of its key.
consteval size_t gen_params(GenKind k) {
    return k == GenKind::LinSpace || gen_is_sampler(k) ? 2 : 1;
}

// A broadcast-scalar operand: arithmetic, or std::complex of arithmetic.
// Type-level twin: is_broadcast_scalar_type (detail/Tree.h).
template <typename T>
inline constexpr bool is_broadcast_scalar_v = std::is_arithmetic_v<T>;
template <typename T>
inline constexpr bool is_broadcast_scalar_v<std::complex<T>> =
    std::is_arithmetic_v<T>;

// The fold protocol at the type level: `summed` plus the `op` alias, the
// same pair detail/Tree.h's is_contraction tests on an info.
template <typename Op>
concept FoldOp = requires {
    Op::summed;
    typename Op::op;
};

// A scatter is a fold whose output cell is chosen by data: the same
// protocol, plus the marker. Every whole-tree rule keyed on FoldOp — one
// fold per tree, no nesting, the epilogue — therefore covers it unchanged.
template <typename Op>
concept ScatterOp = FoldOp<Op> && requires { requires Op::placed; };

// The scan protocol: `scanned` plus the `op` alias. Deliberately NOT a
// FoldOp — a scan PRESERVES its index rather than consuming it, so it has
// no `summed`, and that one absence is what gives it its child's shape
// through node_extents and its child's free ids through free_ids_v below.
template <typename Op>
concept ScanOp = requires {
    Op::scanned;
    typename Op::op;
};

// Which placeholders a node reads, as a bitmask — what lets a scatter ask
// whether any id SURVIVES it without a census walk.
consteval unsigned map_ids(DecMap m) {
    unsigned b = 0;
    for (size_t q = 0; q < m.n; ++q)
        for (size_t p = 0; p < index_slots; ++p)
            if (m.s[q].lin.c[p] != 0)
                b |= 1u << p;
    return b;
}

template <typename Op> consteval unsigned summed_mask() {
    unsigned b = 0;
    if constexpr (FoldOp<Op>)
        for (size_t a : Op::summed)
            b |= 1u << a;
    return b;
}

template <typename T> inline constexpr unsigned free_ids_v = 0;
// A gathered axis reads no placeholder itself; its coordinate does.
template <typename Data> inline constexpr unsigned data_ids_v = 0;
template <typename Seq, typename... Ds>
inline constexpr unsigned data_ids_v<SlotsImpl<Seq, Ds...>> =
    (free_ids_v<Ds> | ... | 0u);

template <typename E, typename Data, DecMap... Ms>
inline constexpr unsigned free_ids_v<Indexed<E, Data, Ms...>> =
    (map_ids(Ms) | ... | 0u) | data_ids_v<Data>;
template <Place P, size_t E, typename C>
inline constexpr unsigned free_ids_v<Placed<P, E, C>> = free_ids_v<C>;
template <typename Op, typename... Cs>
inline constexpr unsigned free_ids_v<Expr<Op, Cs...>> =
    (free_ids_v<Cs> | ... | 0u) & ~summed_mask<Op>();

// A subscripted leaf below — purely structural: a fold node keeps its
// summand's surviving indices visible (what lets an epilogue join them),
// so it propagates like every other node. A SCATTER is the exception: its
// axes are positional, so once it consumes every id it stops being a
// function of indices and combines with plain operands again.
// (The primary template is declared above, beside DataTerm.)
template <typename E, typename Data, DecMap... Ms>
inline constexpr bool has_free_index_v<Indexed<E, Data, Ms...>> = true;
template <Place P, size_t E, typename C>
inline constexpr bool has_free_index_v<Placed<P, E, C>> = has_free_index_v<C>;
template <typename Op, typename... Cs>
inline constexpr bool has_free_index_v<Expr<Op, Cs...>> =
    ScatterOp<Op> ? free_ids_v<Expr<Op, Cs...>> != 0u
                  : (has_free_index_v<Cs> || ...);

// The index-model name: a node that is a function of free indices — its
// shape is its free-index space, not a carrier's.
template <typename T>
inline constexpr bool index_bearing_v = has_free_index_v<T>;

// Folds below a node. Checked at EVERY combination (one fold per tree), so
// it has to be a per-type fact the compiler memoizes — a variable template
// is cached per specialization, a consteval walk re-runs on every call and
// makes the check cost the subtree, per node. A scatter counts as a fold
// here, which is what gives it one-per-tree and no-nesting for free.
template <typename T> inline constexpr size_t fold_count_v = 0;
template <typename Data> inline constexpr size_t data_folds_v = 0;
template <typename Seq, typename... Ds>
inline constexpr size_t data_folds_v<SlotsImpl<Seq, Ds...>> =
    (fold_count_v<Ds> + ... + size_t{0});
// A fold hidden in a gather's coordinate must still count, or it bypasses
// the one-fold-per-tree rule entirely.
template <typename E, typename Data, DecMap... Ms>
inline constexpr size_t fold_count_v<Indexed<E, Data, Ms...>> =
    fold_count_v<E> + data_folds_v<Data>;
template <Place P, size_t E, typename C>
inline constexpr size_t fold_count_v<Placed<P, E, C>> = fold_count_v<C>;
template <typename Op, typename... Cs>
inline constexpr size_t fold_count_v<Expr<Op, Cs...>> =
    (FoldOp<Op> ? size_t{1} : size_t{0}) +
    (fold_count_v<Cs> + ... + size_t{0});

// Scatters below a node — eval must know before it dispatches, since a
// scatter can never be computed per output cell.
template <typename T> inline constexpr size_t scatter_count_v = 0;
template <typename Data> inline constexpr size_t data_scatters_v = 0;
template <typename Seq, typename... Ds>
inline constexpr size_t data_scatters_v<SlotsImpl<Seq, Ds...>> =
    (scatter_count_v<Ds> + ... + size_t{0});
template <typename E, typename Data, DecMap... Ms>
inline constexpr size_t scatter_count_v<Indexed<E, Data, Ms...>> =
    scatter_count_v<E> + data_scatters_v<Data>;
template <Place P, size_t E, typename C>
inline constexpr size_t scatter_count_v<Placed<P, E, C>> = scatter_count_v<C>;
template <typename Op, typename... Cs>
inline constexpr size_t scatter_count_v<Expr<Op, Cs...>> =
    (ScatterOp<Op> ? size_t{1} : size_t{0}) +
    (scatter_count_v<Cs> + ... + size_t{0});

// Scans below a node, counted exactly as folds are — including inside a
// gather's COORDINATE, or the one-per-tree rule is bypassable by hiding a
// scan in a subscript (the hole fold_count_v had to close above).
template <typename T> inline constexpr size_t scan_count_v = 0;
template <typename Data> inline constexpr size_t data_scans_v = 0;
template <typename Seq, typename... Ds>
inline constexpr size_t data_scans_v<SlotsImpl<Seq, Ds...>> =
    (scan_count_v<Ds> + ... + size_t{0});
template <typename E, typename Data, DecMap... Ms>
inline constexpr size_t scan_count_v<Indexed<E, Data, Ms...>> =
    scan_count_v<E> + data_scans_v<Data>;
template <Place P, size_t E, typename C>
inline constexpr size_t scan_count_v<Placed<P, E, C>> = scan_count_v<C>;
template <typename Op, typename... Cs>
inline constexpr size_t scan_count_v<Expr<Op, Cs...>> =
    (ScanOp<Op> ? size_t{1} : size_t{0}) +
    (scan_count_v<Cs> + ... + size_t{0});

// An eval<Order...> entry's id; a non-placeholder yields the none
// sentinel and the order lint names it.
template <auto O> consteval size_t order_id() {
    using T = std::remove_cvref_t<decltype(O)>;
    if constexpr (is_placeholder_v<T>)
        return T::id;
    else
        return index_slots;
}

} // namespace detail
} // namespace tensor
