#pragma once

// Definitions for Tensor/Npy.h — included there, not a direct surface.

// Own surface header first, the way every definition folder does it: these
// definitions must see the declarations they define, and the header has to
// stand alone for the lint's sweep.
#include "../Npy.h"

#include "../detail/Npy.h"

#include <cstring>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

namespace tensor::npy {

// The header is a pure function of the type, so it is computed once at
// compile time and handed out as a view into static storage — the same shape
// as gpu_source<E>().
template <NpyTensor T> consteval std::string_view header() {
    return std::define_static_string(
        detail::npy_header_text(^^std::remove_cvref_t<T>));
}

// One place asserts the element type, so every entry point inherits the same
// message. Fired through a DISCARDED branch: a static_assert's message is
// constant-evaluated even when the assertion holds, and this one renders a
// type name.
template <typename T> constexpr void require_npy_element() {
    if constexpr (!detail::npy_element_spellable(
                      detail::npy_element_of(^^std::remove_cvref_t<T>)))
        static_assert(false, detail::npy_element_error(^^std::remove_cvref_t<T>));
}

// ── one tensor, one file ────────────────────────────────────────────────────

// The bytes of a .npy member: preamble, header, buffer. Shared by save and by
// the archive, so both write the identical thing.
template <NpyTensor T> std::string npy_bytes(const T &tensor) {
    using U = std::remove_cvref_t<T>;
    require_npy_element<U>();
    constexpr std::string_view head = header<U>();
    std::string out = detail::npy_preamble_bytes(head.size());
    out += head;
    out.append(reinterpret_cast<const char *>(tensor.data()), U::byte_size);
    return out;
}

template <NpyTensor T>
void save(const std::filesystem::path &path, const T &tensor) {
    const std::string bytes = npy_bytes(tensor);
    detail::NpyFile file(path, "wb", "could not open for writing");
    detail::npy_write_exact(file.handle, bytes.data(), bytes.size(), path);
}

// The header is a SCHEMA check, and it compares what the fields MEAN rather
// than how they were spelt: NumPy pads to 64 bytes and other writers to 16,
// so byte-matching would reject files NumPy itself produced. What must agree
// exactly is the dtype, the layout and the shape — and a mismatch names which.
template <NpyTensor T>
void check_header(const std::filesystem::path &path, std::string_view text) {
    using U = std::remove_cvref_t<T>;
    const detail::NpyHeader found = detail::npy_parsed_header(text);
    if (!found.parsed)
        detail::npy_fail(path, "malformed header '" +
                                   detail::npy_escape_header(text) + "'");

    static constexpr std::string_view want_descr = std::define_static_string(
        detail::npy_descr(detail::npy_element_of(^^U)));
    if (found.descr != want_descr)
        detail::npy_fail(path, "dtype is '" + found.descr + "', but " +
                                   std::string(detail::npy_type_name<U>()) +
                                   " needs '" + std::string(want_descr) + "'");
    if (found.fortran_order)
        detail::npy_fail(path, "the file is Fortran-ordered; this reader is "
                               "C-order (np.ascontiguousarray before saving)");

    static constexpr auto want_shape =
        std::define_static_array(detail::npy_extents_of(^^U));
    if (!std::ranges::equal(found.shape, want_shape))
        detail::npy_fail(
            path, "shape is " + detail::npy_shape_text(found.shape) +
                      ", but " + std::string(detail::npy_type_name<U>()) +
                      " needs " +
                      detail::npy_shape_text({want_shape.begin(),
                                              want_shape.end()}));
}

// A .npy member's bytes, already in memory, into a tensor.
template <NpyTensor T>
std::remove_cvref_t<T> from_npy_bytes(const std::filesystem::path &path,
                                      std::string_view bytes) {
    using U = std::remove_cvref_t<T>;
    require_npy_element<U>();
    if (bytes.size() < detail::npy_preamble)
        detail::npy_fail(path, "too short to be a .npy member");
    if (bytes.compare(0, 6, "\x93NUMPY") != 0)
        detail::npy_fail(path, "expected a NumPy file (bad magic)");
    if (bytes[6] != 1)
        detail::npy_fail(path, "unsupported .npy version " +
                                   std::to_string(int(bytes[6])) + "." +
                                   std::to_string(int(bytes[7])));
    const size_t head =
        size_t(static_cast<unsigned char>(bytes[8])) |
        (size_t(static_cast<unsigned char>(bytes[9])) << 8);
    if (bytes.size() < detail::npy_preamble + head)
        detail::npy_fail(path, "header runs past the end of the file");
    check_header<U>(path, bytes.substr(detail::npy_preamble, head));

    const size_t offset = detail::npy_preamble + head;
    if (bytes.size() - offset < U::byte_size)
        detail::npy_fail(path, "unexpected end of file: " +
                                   std::to_string(bytes.size() - offset) +
                                   " data bytes, " +
                                   std::to_string(U::byte_size) + " needed");
    U result;
    std::memcpy(result.data(), bytes.data() + offset, U::byte_size);
    return result;
}

template <NpyTensor T>
std::remove_cvref_t<T> load(const std::filesystem::path &path) {
    detail::NpyFile file(path, "rb", "could not open for reading");
    std::string bytes;
    char buffer[8192];
    for (size_t n; (n = std::fread(buffer, 1, sizeof(buffer), file.handle));)
        bytes.append(buffer, n);
    return from_npy_bytes<T>(path, bytes);
}

// ── a struct of tensors, one archive, keyed by member name ──────────────────
// This is the whole point of doing it reflectively: the .npz keys come from
// the STRUCT's field names, so there is no list to keep in step with the
// type, and np.load hands Python a mapping under exactly those names.

template <NpyAggregate S>
void savez(const std::filesystem::path &path, const S &fields) {
    using D = std::remove_cvref_t<S>;
    static constexpr auto members =
        std::define_static_array(detail::npy_members_of<D>());

    std::string blob;
    std::vector<detail::NpzEntry> entries;
    template for (constexpr auto m : members) {
        static constexpr std::string_view name =
            std::define_static_string(std::string(std::meta::identifier_of(m)) +
                                      ".npy");
        const std::string member = npy_bytes(fields.[:m:]);
        if (member.size() > detail::npz_limit)
            detail::npy_fail(path, std::string(name) +
                                       " needs zip64, which this writer does "
                                       "not emit — save it on its own");
        detail::NpzEntry e{std::string(name),
                           detail::npy_crc32(member.data(), member.size()),
                           member.size(), blob.size()};
        blob += detail::npz_local_header(e);
        blob += member;
        entries.push_back(std::move(e));
    }
    if (blob.size() > detail::npz_limit)
        detail::npy_fail(path, "archive exceeds 4 GB, which needs zip64");
    blob += detail::npz_central_directory(entries, blob.size());

    detail::NpyFile file(path, "wb", "could not open for writing");
    detail::npy_write_exact(file.handle, blob.data(), blob.size(), path);
}

template <NpyAggregate S>
std::remove_cvref_t<S> loadz(const std::filesystem::path &path) {
    using D = std::remove_cvref_t<S>;
    static constexpr auto members =
        std::define_static_array(detail::npy_members_of<D>());

    detail::NpyFile file(path, "rb", "could not open for reading");
    std::string blob;
    char buffer[8192];
    for (size_t n; (n = std::fread(buffer, 1, sizeof(buffer), file.handle));)
        blob.append(buffer, n);

    D result;
    template for (constexpr auto m : members) {
        static constexpr std::string_view name =
            std::define_static_string(std::string(std::meta::identifier_of(m)) +
                                      ".npy");
        const std::uint64_t at = detail::npz_find(blob, name);
        if (at == std::uint64_t(-1))
            detail::npy_fail(path, "no member named '" + std::string(name) +
                                       "' (the archive must carry one entry "
                                       "per field of " +
                                       std::string(detail::npy_type_name<D>()) +
                                       ")");
        using M = std::remove_cvref_t<decltype(result.[:m:])>;
        result.[:m:] = from_npy_bytes<M>(
            path, std::string_view(blob).substr(size_t(at)));
    }
    return result;
}

} // namespace tensor::npy
