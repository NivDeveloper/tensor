#pragma once

// The NumPy format's type and byte-level mechanics — not a public surface.
// Diagnostics for this layer live here rather than in Diagnostics.h, the way
// the GPU layer keeps its own (detail/Gpu.h): the messages name formats and
// dtypes, not expressions.

// Core.h for the Tensor declaration this file reflects over; Meta.h for
// is_specialization_of and type_name.
#include "Core.h"
#include "Meta.h"

#include <array>
#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <meta>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace tensor::detail {

// detail::to_string is consteval, and these two spellings are needed at
// runtime as well — the shape a FILE claims goes into a diagnostic.
constexpr std::string npy_number(size_t v) {
    if (v == 0)
        return "0";
    std::string digits;
    for (; v; v /= 10)
        digits += char('0' + v % 10);
    return std::string(digits.rbegin(), digits.rend());
}

// ── the dtype, DERIVED ──────────────────────────────────────────────────────
// A .npy descriptor is three facts about the element type — byte order, kind,
// width — and the type already carries all three. Deriving them instead of
// tabulating them is what makes every arithmetic type work at once (i1 … i8,
// u1 … u8, f2/f4/f8, c8/c16, b1) with nothing to keep in step.

// 'b' bool, 'i' signed, 'u' unsigned, 'f' floating, 'c' complex, or '\0' for
// a type the format cannot spell.
consteval char npy_kind(std::meta::info t) {
    t = std::meta::dealias(t);
    if (t == ^^bool)
        return 'b';
    if (is_specialization_of(t, ^^std::complex)) {
        const auto v = std::meta::dealias(std::meta::template_arguments_of(t)[0]);
        return std::meta::is_floating_point_type(v) ? 'c' : '\0';
    }
    if (std::meta::is_floating_point_type(t))
        return 'f';
    if (std::meta::is_integral_type(t))
        return std::meta::is_signed_type(t) ? 'i' : 'u';
    return '\0';
}

// What NumPy portably names. A width outside these is refused rather than
// guessed: 'f16' is x86's 80-bit long double padded, which no other platform
// reproduces, so a file claiming it would not mean the same thing twice.
consteval bool npy_width_portable(char kind, size_t width) {
    switch (kind) {
    case 'b':
        return width == 1;
    case 'i':
    case 'u':
        return width == 1 || width == 2 || width == 4 || width == 8;
    case 'f':
        return width == 2 || width == 4 || width == 8;
    case 'c':
        return width == 8 || width == 16;
    default:
        return false;
    }
}

consteval bool npy_element_spellable(std::meta::info t) {
    const char kind = npy_kind(t);
    return kind != '\0' &&
           npy_width_portable(kind, std::meta::size_of(std::meta::dealias(t)));
}

// The descriptor itself. '|' is the format's "byte order does not apply",
// which is what a single byte is; anything wider says which end it wrote,
// so a big-endian host produces correct files rather than being refused.
consteval std::string npy_descr(std::meta::info t) {
    t = std::meta::dealias(t);
    const size_t width = std::meta::size_of(t);
    const char order =
        width == 1 ? '|'
                   : (std::endian::native == std::endian::little ? '<' : '>');
    std::string out;
    out += order;
    out += npy_kind(t);
    out += npy_number(width);
    return out;
}

// ── the tensor as a schema ──────────────────────────────────────────────────
// Tensor's template arguments ARE the .npy header. Reflection keeps that
// relationship generic: no parallel Tensor<T, Es...> trait to maintain.
template <typename T> consteval bool is_npy_tensor() {
    return is_specialization_of(^^T, ^^Tensor);
}

consteval std::meta::info npy_element_of(std::meta::info tensor) {
    return std::meta::template_arguments_of(std::meta::dealias(tensor))[0];
}

consteval std::vector<size_t> npy_extents_of(std::meta::info tensor) {
    return args_of<size_t>(std::meta::dealias(tensor), 1);
}

// constexpr, not consteval: the header builder calls it at compile time and
// the shape-mismatch diagnostic calls it with a shape parsed from a file.
constexpr std::string npy_shape_text(const std::vector<size_t> &shape) {
    std::string s = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0)
            s += ", ";
        s += npy_number(shape[i]);
    }
    if (shape.size() == 1)
        s += ','; // Python's one-tuple needs the comma
    return s + ")";
}

// The complete v1.0 header for a tensor type, padding included. NumPy aligns
// the whole preamble to 64 bytes so the data starts on a mappable boundary;
// matching it makes a file byte-identical to numpy.save's for the same array.
inline constexpr size_t npy_align = 64;
inline constexpr size_t npy_preamble = 10; // magic(6) + version(2) + len(2)

consteval std::string npy_header_text(std::meta::info tensor) {
    std::string s = "{'descr': '";
    s += npy_descr(npy_element_of(tensor));
    s += "', 'fortran_order': False, 'shape': ";
    s += npy_shape_text(npy_extents_of(tensor));
    s += ", }";
    s.append(npy_align - ((npy_preamble + s.size() + 1) % npy_align), ' ');
    s += '\n';
    return s;
}

// ── diagnostics ─────────────────────────────────────────────────────────────

consteval std::string npy_element_error(std::meta::info tensor) {
    const auto e = std::meta::dealias(npy_element_of(tensor));
    return "npy: no portable NumPy dtype for element type " + type_name(e) +
           " (" + npy_number(std::meta::size_of(e)) +
           " bytes) — the format spells bool, the signed and unsigned "
           "integers of 1/2/4/8 bytes, floating point of 2/4/8, and "
           "std::complex of the 4- and 8-byte floats";
}

// ── byte-level file mechanics ───────────────────────────────────────────────

[[noreturn]] inline void npy_fail(const std::filesystem::path &path,
                                  const std::string &what) {
    throw std::runtime_error("npy: " + path.string() + ": " + what);
}

inline void npy_write_exact(std::FILE *file, const void *data, size_t size,
                            const std::filesystem::path &path) {
    if (size && std::fwrite(data, 1, size, file) != size)
        npy_fail(path, "write failed");
}

inline void npy_read_exact(std::FILE *file, void *data, size_t size,
                           const std::filesystem::path &path) {
    if (size && std::fread(data, 1, size, file) != size)
        npy_fail(path,
                 std::feof(file) ? "unexpected end of file" : "read failed");
}

// Closes its file however the scope exits — every reader and writer below
// opens through one of these.
struct NpyFile {
    std::FILE *handle = nullptr;
    NpyFile(const std::filesystem::path &path, const char *mode,
            const char *what) {
        handle = std::fopen(path.string().c_str(), mode);
        if (!handle)
            npy_fail(path, what);
    }
    NpyFile(const NpyFile &) = delete;
    NpyFile &operator=(const NpyFile &) = delete;
    ~NpyFile() {
        if (handle)
            std::fclose(handle);
    }
};

// The preamble, as bytes: \x93NUMPY, version 1.0, then the header length as
// a little-endian u16.
inline std::string npy_preamble_bytes(size_t header_size) {
    std::string out = "\x93NUMPY";
    out += char{1};
    out += char{0};
    out += static_cast<char>(header_size & 0xff);
    out += static_cast<char>((header_size >> 8) & 0xff);
    return out;
}

// ── reading a header back ───────────────────────────────────────────────────
// Parsed rather than string-matched: the dict text is canonical in practice
// but its PADDING is not — NumPy pads to 64 bytes and older writers to 16, so
// a byte comparison rejects files NumPy itself wrote. Comparing meaning also
// lets a mismatch name the field that disagreed.
struct NpyHeader {
    bool parsed = false;
    std::string descr;
    bool fortran_order = false;
    std::vector<size_t> shape;
};

// A deliberately small reader for the one dict shape the format allows: find
// each key, read the literal after it, ignore whitespace.
inline NpyHeader npy_parsed_header(std::string_view text) {
    NpyHeader out;
    auto value_at = [&](std::string_view key) -> size_t {
        const size_t k = text.find(key);
        if (k == std::string_view::npos)
            return std::string_view::npos;
        const size_t colon = text.find(':', k + key.size());
        if (colon == std::string_view::npos)
            return std::string_view::npos;
        return text.find_first_not_of(" \t", colon + 1);
    };

    size_t p = value_at("'descr'");
    if (p == std::string_view::npos || text[p] != '\'')
        return out;
    const size_t close = text.find('\'', p + 1);
    if (close == std::string_view::npos)
        return out;
    out.descr = std::string(text.substr(p + 1, close - p - 1));

    p = value_at("'fortran_order'");
    if (p == std::string_view::npos)
        return out;
    if (text.compare(p, 4, "True") == 0)
        out.fortran_order = true;
    else if (text.compare(p, 5, "False") == 0)
        out.fortran_order = false;
    else
        return out;

    p = value_at("'shape'");
    if (p == std::string_view::npos || text[p] != '(')
        return out;
    for (size_t q = p + 1; q < text.size() && text[q] != ')'; ++q) {
        if (text[q] < '0' || text[q] > '9')
            continue;
        size_t value = 0;
        while (q < text.size() && text[q] >= '0' && text[q] <= '9')
            value = value * 10 + size_t(text[q++] - '0');
        out.shape.push_back(value);
        --q;
    }
    out.parsed = true;
    return out;
}

// The offending type, spelt as the user wrote it — a runtime message naming
// Tensor<float, 2, 3> beats one quoting two padded header strings.
// define_static_string: type_name builds a consteval std::string, which
// cannot itself cross into a runtime expression.
template <typename T> constexpr std::string_view npy_type_name() {
    return std::define_static_string(type_name(^^T));
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

// ── the archive: a zip of .npy members, which is all a .npz is ──────────────
// Stored (uncompressed) entries only. numpy.savez writes exactly this shape,
// and np.load returns a mapping keyed by the member names.

inline std::uint32_t npy_crc32(const void *data, size_t size,
                               std::uint32_t crc = 0) {
    static const auto table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    const auto *bytes = static_cast<const unsigned char *>(data);
    crc = ~crc;
    for (size_t i = 0; i < size; ++i)
        crc = table[(crc ^ bytes[i]) & 0xff] ^ (crc >> 8);
    return ~crc;
}

inline void npy_put_le(std::string &out, std::uint64_t value, size_t bytes) {
    for (size_t i = 0; i < bytes; ++i)
        out += static_cast<char>((value >> (8 * i)) & 0xff);
}

// One member of the archive, as the central directory will need to recall it.
struct NpzEntry {
    std::string name;
    std::uint32_t crc = 0;
    std::uint64_t size = 0;   // the member's bytes, header included
    std::uint64_t offset = 0; // where its local header begins
};

inline std::string npz_local_header(const NpzEntry &e) {
    std::string h;
    npy_put_le(h, 0x04034b50, 4); // local file header
    npy_put_le(h, 20, 2);         // version needed
    npy_put_le(h, 0, 2);          // flags
    npy_put_le(h, 0, 2);          // method: stored
    npy_put_le(h, 0, 2);          // time
    npy_put_le(h, 0, 2);          // date
    npy_put_le(h, e.crc, 4);
    npy_put_le(h, e.size, 4); // compressed == uncompressed
    npy_put_le(h, e.size, 4);
    npy_put_le(h, e.name.size(), 2);
    npy_put_le(h, 0, 2); // extra
    h += e.name;
    return h;
}

inline std::string npz_central_directory(const std::vector<NpzEntry> &entries,
                                        std::uint64_t offset) {
    std::string cd;
    for (const auto &e : entries) {
        npy_put_le(cd, 0x02014b50, 4); // central directory header
        npy_put_le(cd, 20, 2);         // version made by
        npy_put_le(cd, 20, 2);         // version needed
        npy_put_le(cd, 0, 2);          // flags
        npy_put_le(cd, 0, 2);          // method: stored
        npy_put_le(cd, 0, 2);          // time
        npy_put_le(cd, 0, 2);          // date
        npy_put_le(cd, e.crc, 4);
        npy_put_le(cd, e.size, 4);
        npy_put_le(cd, e.size, 4);
        npy_put_le(cd, e.name.size(), 2);
        npy_put_le(cd, 0, 2); // extra
        npy_put_le(cd, 0, 2); // comment
        npy_put_le(cd, 0, 2); // disk
        npy_put_le(cd, 0, 2); // internal attrs
        npy_put_le(cd, 0, 4); // external attrs
        npy_put_le(cd, e.offset, 4);
        cd += e.name;
    }
    std::string end;
    npy_put_le(end, 0x06054b50, 4); // end of central directory
    npy_put_le(end, 0, 2);          // this disk
    npy_put_le(end, 0, 2);          // disk with the directory
    npy_put_le(end, entries.size(), 2);
    npy_put_le(end, entries.size(), 2);
    npy_put_le(end, cd.size(), 4);
    npy_put_le(end, offset, 4);
    npy_put_le(end, 0, 2); // comment
    return cd + end;
}

// Zip's 32-bit fields; past this an archive needs the zip64 extensions, which
// nothing here writes.
inline constexpr std::uint64_t npz_limit = 0xffffffffull;

// Where an entry named `name` starts inside the archive, or npos.
inline std::uint64_t npz_find(const std::string &blob, std::string_view name) {
    const std::string want(name);
    for (size_t p = 0; p + 30 <= blob.size();) {
        if (blob.compare(p, 4, "PK\x03\x04") != 0)
            return std::uint64_t(-1);
        const auto u16 = [&](size_t off) {
            return size_t(static_cast<unsigned char>(blob[p + off])) |
                   (size_t(static_cast<unsigned char>(blob[p + off + 1])) << 8);
        };
        const auto u32 = [&](size_t off) {
            std::uint64_t v = 0;
            for (size_t i = 0; i < 4; ++i)
                v |= std::uint64_t(
                         static_cast<unsigned char>(blob[p + off + i]))
                     << (8 * i);
            return v;
        };
        const size_t name_len = u16(26), extra_len = u16(28);
        const std::uint64_t size = u32(18);
        const size_t data = p + 30 + name_len + extra_len;
        if (blob.compare(p + 30, name_len, want) == 0)
            return data;
        p = data + size;
    }
    return std::uint64_t(-1);
}

// ── an aggregate of tensors, by member name ─────────────────────────────────
// The archive's keys come from the STRUCT: reflection reads the member names,
// so `struct State { Vecs Pos, Mom; }` lands as Pos.npy and Mom.npy and
// np.load returns them under exactly those names. Nothing has to be listed
// twice.
template <typename S> consteval std::vector<std::meta::info> npy_members_of() {
    return std::meta::nonstatic_data_members_of(
        ^^S, std::meta::access_context::current());
}

template <typename S> consteval bool is_npy_aggregate() {
    if (!std::meta::is_class_type(^^S) || is_npy_tensor<S>())
        return false;
    const auto members = npy_members_of<S>();
    if (members.empty())
        return false;
    for (auto m : members) {
        const auto t = std::meta::dealias(std::meta::type_of(m));
        if (!is_specialization_of(t, ^^Tensor) || !std::meta::has_identifier(m))
            return false; // the element type is asserted per member, with a
                          // message, where that member is written
    }
    return true;
}

} // namespace tensor::detail
