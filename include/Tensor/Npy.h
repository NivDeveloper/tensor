#pragma once

// NumPy .npy / .npz I/O — the way out of this library and into anything that
// reads NumPy's format (Python, Julia, R, MATLAB):
//
//   npy::save("mom.npy", Mom);                        // one tensor
//   auto Mom = npy::load<Tensor<float, 8192, 3>>("mom.npy");
//
//   struct State { Tensor<float, N, 3> Pos, Mom; };
//   npy::savez("state.npz", state);                   // a struct of tensors,
//   auto state = npy::loadz<State>("state.npz");      // keyed by FIELD NAME
//
// then, in Python, no schema to declare:
//
//   d = np.load("state.npz"); plt.hist(d["Mom"][:, 0], bins=80)
//
// A tensor's template arguments ARE the format's header: element type gives
// the dtype, the extents give the shape, and the layout is always C-order —
// all of it derived at compile time, so a file this writes is byte-identical
// to numpy.save's. load inverts that into a CONTRACT: it accepts only the
// dtype, shape and order the requested type demands, and names the field that
// disagreed. There is deliberately no plotting here.
//
// Definitions live in Tensor/IO/Npy.h, included at the bottom.

#include "Tensor.h"
#include "detail/Npy.h"

#include <filesystem>
#include <string_view>
#include <type_traits>

namespace tensor {

// A materialized tensor — the structural gate, constrained rather than
// asserted so a wrong KIND of argument is simply not a candidate and the
// constraint names itself, the way the index fill refuses a wrong-rank
// function. Whether the format can spell the element TYPE is a separate
// question, and one worth a message rather than a candidacy failure: the
// definitions assert it, naming the type, its width and the set that works.
template <typename T>
concept NpyTensor = detail::is_npy_tensor<std::remove_cvref_t<T>>();

// An aggregate whose every non-static member is such a tensor: one .npz
// entry per member, named by the member.
template <typename S>
concept NpyAggregate = detail::is_npy_aggregate<std::remove_cvref_t<S>>();

namespace npy {

// The exact header bytes for a tensor type — a compile-time constant, so it
// can be pinned in a static_assert the way the generated GPU program is.
template <NpyTensor T> consteval std::string_view header();

template <NpyTensor T>
void save(const std::filesystem::path &path, const T &tensor);

template <NpyTensor T>
std::remove_cvref_t<T> load(const std::filesystem::path &path);

// eval is the only way to evaluate: .npy writes a materialized buffer, and an
// expression has none until eval gives it one. A Tensor satisfies AnyExpr
// too, so this must exclude the case that HAS a buffer or it would merely be
// ambiguous with the definition above.
template <AnyExpr E>
    requires(!NpyTensor<E>)
void save(const std::filesystem::path &path, const E &expr) =
    delete("npy::save writes a tensor's buffer, and an expression has none — "
           "npy::save(path, eval(expr))");

// A range-tagged coordinate is an expression too, and must reach the same
// sentence rather than falling off the overload set.
template <detail::RangedExpr E>
void save(const std::filesystem::path &path, const E &expr) =
    delete("npy::save writes a tensor's buffer, and an expression has none — "
           "npy::save(path, eval(expr))");

template <NpyAggregate S>
void savez(const std::filesystem::path &path, const S &fields);

template <NpyAggregate S>
std::remove_cvref_t<S> loadz(const std::filesystem::path &path);

} // namespace npy
} // namespace tensor

// The definitions.
#include "IO/Npy.h" // IWYU pragma: export
