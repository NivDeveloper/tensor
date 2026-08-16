#pragma once

// Definitions for Tensor/Npy.h — included there, not a direct surface.

#include "../detail/Npy.h"

namespace tensor::npy {

template <typename T> std::string header() {
    using U = std::remove_cvref_t<T>;
    return tensor::detail::npy_header<U>();
}

template <typename T>
void save(const std::filesystem::path &path, const T &tensor) {
    using U = std::remove_cvref_t<T>;
    tensor::detail::check_npy_tensor<U>();
    const std::string h = header<U>();
    if (h.size() > 0xffff)
        tensor::detail::npy_fail(path, "header is too large for NumPy v1.0");
    std::FILE *raw = std::fopen(path.string().c_str(), "wb");
    if (!raw)
        tensor::detail::npy_fail(path, "could not open for writing");
    struct Closer { std::FILE *file; ~Closer() { std::fclose(file); } } closer{raw};

    constexpr std::array<unsigned char, 8> magic{
        0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    const std::array<unsigned char, 2> length{
        static_cast<unsigned char>(h.size() & 0xff),
        static_cast<unsigned char>(h.size() >> 8)};
    tensor::detail::npy_write_exact(raw, magic.data(), magic.size(), path);
    tensor::detail::npy_write_exact(raw, length.data(), length.size(), path);
    tensor::detail::npy_write_exact(raw, h.data(), h.size(), path);
    tensor::detail::npy_write_exact(raw, tensor.data(), U::byte_size, path);
}

template <typename T>
std::remove_cvref_t<T> load(const std::filesystem::path &path) {
    using U = std::remove_cvref_t<T>;
    tensor::detail::check_npy_tensor<U>();
    std::FILE *raw = std::fopen(path.string().c_str(), "rb");
    if (!raw)
        tensor::detail::npy_fail(path, "could not open for reading");
    struct Closer { std::FILE *file; ~Closer() { std::fclose(file); } } closer{raw};

    constexpr std::array<unsigned char, 8> magic{
        0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    std::array<unsigned char, 8> actual_magic{};
    tensor::detail::npy_read_exact(raw, actual_magic.data(), actual_magic.size(), path);
    if (actual_magic != magic)
        tensor::detail::npy_fail(path, "expected a NumPy v1.0 file");
    std::array<unsigned char, 2> length{};
    tensor::detail::npy_read_exact(raw, length.data(), length.size(), path);
    const size_t size = size_t(length[0]) | (size_t(length[1]) << 8);
    std::string actual(size, '\0');
    tensor::detail::npy_read_exact(raw, actual.data(), actual.size(), path);
    const std::string expected = header<U>();
    if (actual != expected)
        tensor::detail::npy_fail(path, "header does not match Tensor schema; expected '" +
            tensor::detail::npy_escape_header(expected) + "', got '" +
            tensor::detail::npy_escape_header(actual) + "'");

    U result;
    tensor::detail::npy_read_exact(raw, result.data(), U::byte_size, path);
    return result;
}

} // namespace tensor::npy
