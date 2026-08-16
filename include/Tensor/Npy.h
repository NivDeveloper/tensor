#pragma once

// NumPy .npy I/O for owning, statically-shaped Tensors:
//
//   npy::save("state.npy", state);
//   auto state = npy::load<Tensor<float, 24576, 3>>("state.npy");
//
// Files use NumPy v1.0, C-order, and little-endian scalar representations.
// load deliberately treats the header as a schema: it accepts only the exact
// dtype, shape, and layout emitted for the requested Tensor type. Definitions
// live in Tensor/IO/Npy.h, included at the bottom.

#include "Tensor.h"

#include <filesystem>
#include <string>
#include <type_traits>

namespace tensor::npy {

template <typename T> std::string header();

template <typename T>
void save(const std::filesystem::path &path, const T &tensor);

template <typename T>
std::remove_cvref_t<T> load(const std::filesystem::path &path);

} // namespace tensor::npy

// The definitions.
#include "Tensor/IO/Npy.h" // IWYU pragma: export
