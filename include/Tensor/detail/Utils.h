#pragma once

// Context-free helpers: no reflection, no tensor types, nothing above std.
// The lowest layer — everything else in detail/ may include this, and it
// includes nothing of ours.
//
// What lives here is what the standard library does not give us in a form
// this library can use: a constexpr to_string that does not exist yet, and
// storage whose generality would cost more compile time than the feature is
// worth. Each is a drop-in whose only claim is being cheaper.

#include <array>
#include <cstddef>
#include <string>
#include <utility>

namespace tensor::detail {

// ── consteval string formatting ─────────────────────────────────────────────

// Append n's decimal spelling. The appending form is the one string builders
// should reach for: it neither allocates nor returns, so a builder that emits
// thousands of numbers pays no constant-evaluation cost per number beyond its
// digits.
constexpr void append_number(std::string &out, size_t n) {
    char buf[20];
    size_t i = sizeof buf;
    do {
        buf[--i] = char('0' + n % 10);
        n /= 10;
    } while (n != 0);
    out.append(buf + i, sizeof buf - i);
}

consteval std::string to_string(size_t n) { // until std::to_string is constexpr
    std::string s;
    append_number(s, n);
    return s;
}

// Fixed-width hex, for emitting a value as an exact bit pattern when its
// decimal spelling would not round-trip.
consteval std::string to_hex(unsigned long long n, size_t digits) {
    std::string s;
    for (size_t i = 0; i < digits; ++i, n >>= 4)
        s.insert(s.begin(), "0123456789abcdef"[n & 0xf]);
    return s;
}

// ── storage ─────────────────────────────────────────────────────────────────

// A vector with a compile-time capacity and no allocation. Being allocation
// free makes it a literal type, which is what lets a value containing one be
// memoized as a per-type constant — a std::vector member cannot.
template <typename T, size_t N> struct SmallVec {
    std::array<T, N> v{};
    size_t n = 0;

    constexpr void push_back(const T &x) { v[n++] = x; }
    constexpr const T *begin() const { return v.data(); }
    constexpr const T *end() const { return v.data() + n; }
    constexpr size_t size() const { return n; }
    constexpr bool empty() const { return n == 0; }
    constexpr bool operator==(const SmallVec &) const = default;
};

// A flat aggregate of indexed slots: the storage std::tuple would give,
// without the constrained constructors that dominate compile time on a deep
// tree (they cost ~100x this on a 256-node chain). The index keeps the bases
// distinct when two slots share a type.
template <size_t I, typename T> struct Slot {
    T value;
};
template <typename Seq, typename... Ts> struct SlotsImpl;
template <size_t... Is, typename... Ts>
struct SlotsImpl<std::index_sequence<Is...>, Ts...> : Slot<Is, Ts>... {};

template <typename... Ts>
using Slots = SlotsImpl<std::index_sequence_for<Ts...>, Ts...>;

// The I-th slot, found by base conversion — T deduces from the slot.
template <size_t I, typename T>
constexpr const T &slot_get(const Slot<I, T> &s) {
    return s.value;
}

} // namespace tensor::detail
