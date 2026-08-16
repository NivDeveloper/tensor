#pragma once

// The NumPy format's type and byte-level mechanics — not a public surface.

#include "Meta.h"

#include <array>
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace tensor::detail {

template <typename> struct npy_type;

template <> struct npy_type<float> { static constexpr std::string_view descr = "<f4"; };
template <> struct npy_type<double> { static constexpr std::string_view descr = "<f8"; };
template <> struct npy_type<int> { static constexpr std::string_view descr = "<i4"; };
template <> struct npy_type<unsigned> { static constexpr std::string_view descr = "<u4"; };
template <> struct npy_type<std::complex<float>> { static constexpr std::string_view descr = "<c8"; };
template <> struct npy_type<std::complex<double>> { static constexpr std::string_view descr = "<c16"; };

template <typename T>
concept NpyElement = requires { npy_type<std::remove_cv_t<T>>::descr; };

// Tensor's template arguments ARE the .npy schema. Reflection keeps that
// relationship generic: no parallel Tensor<T, Es...> trait needs maintaining.
template <typename T> consteval bool is_npy_tensor() {
    return is_specialization_of(^^T, ^^Tensor);
}

template <typename T> consteval std::meta::info npy_element_of() {
    auto args = std::meta::template_arguments_of(std::meta::dealias(^^T));
    return args[0];
}

template <typename T> std::string npy_header() {
    if constexpr (!is_npy_tensor<T>()) {
        static_assert(is_npy_tensor<T>(),
                      "npy I/O requires an owning Tensor<T, Es...>");
    } else {
        using E = typename[:npy_element_of<T>():];
        static_assert(NpyElement<E>,
                      "npy supports float, double, int, unsigned, and "
                      "std::complex<float/double>");
        static_assert(sizeof(int) == 4 && sizeof(unsigned) == 4,
                      "npy's int and unsigned mapping requires 32-bit types");

        std::string header = "{'descr': '";
        header += npy_type<E>::descr;
        header += "', 'fortran_order': False, 'shape': (";
        for (size_t i = 0; i < T::rank; ++i) {
            if (i != 0)
                header += ", ";
            header += std::to_string(T::extent(i));
        }
        if constexpr (T::rank == 1)
            header += ','; // NumPy's required one-tuple syntax
        header += "), }";

        // v1.0's preamble is ten bytes; its complete header is 16-byte aligned.
        const size_t padding = 16 - ((10 + header.size() + 1) % 16);
        header.append(padding, ' ');
        header += '\n';
        return header;
    }
}

inline std::string npy_escape_header(std::string_view header) {
    std::string out;
    out.reserve(header.size());
    for (unsigned char c : header) {
        if (c == '\n')
            out += "\\n";
        else if (c >= 32 && c < 127)
            out += static_cast<char>(c);
        else
            out += '?';
    }
    return out;
}

[[noreturn]] inline void npy_fail(const std::filesystem::path &path,
                                  const std::string &what) {
    throw std::runtime_error("npy: " + path.string() + ": " + what);
}

inline void npy_write_exact(std::FILE *file, const void *data, size_t size,
                            const std::filesystem::path &path) {
    if (std::fwrite(data, 1, size, file) != size)
        npy_fail(path, "write failed");
}

inline void npy_read_exact(std::FILE *file, void *data, size_t size,
                           const std::filesystem::path &path) {
    if (std::fread(data, 1, size, file) != size)
        npy_fail(path, std::feof(file) ? "unexpected end of file" : "read failed");
}

template <typename T> void check_npy_tensor() {
    if constexpr (!is_npy_tensor<T>()) {
        static_assert(is_npy_tensor<T>(),
                      "npy I/O requires an owning Tensor<T, Es...>");
    } else {
        using E = typename[:npy_element_of<T>():];
        static_assert(std::endian::native == std::endian::little,
                      ".npy I/O currently requires a little-endian host");
        static_assert(std::is_trivially_copyable_v<E>,
                      ".npy elements must be trivially copyable");
    }
}

} // namespace tensor::detail
