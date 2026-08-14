#pragma once

// The process-global sampling state behind Gen.h. Separate from Rng.h,
// which stays pure arithmetic over integers so the same source can describe
// a shader; everything stateful lives here.

#include "Rng.h"

#include <atomic>
#include <cstdint>
#include <random>

namespace tensor::detail {

// Random by default: absent a seed(n) call the run draws its own, so
// nothing is silently reproducible without being asked for.
inline std::atomic<std::uint64_t> &seed_slot() {
    static std::atomic<std::uint64_t> s{[] {
        std::random_device rd;
        return (std::uint64_t(rd()) << 32) | std::uint64_t(rd());
    }()};
    return s;
}

// One counter for the whole process, so two samplers can never collide —
// distinctness by construction rather than by a hash that is merely
// unlikely to repeat. Inserting a draw shifts every later one, which is
// what a seeded generator has always done.
inline std::atomic<std::uint64_t> &draw_slot() {
    static std::atomic<std::uint64_t> c{0};
    return c;
}

// The key a sampler leaf carries, claimed where the sampler is WRITTEN.
// That is what makes eval a pure function of the expression: evaluating one
// twice gives the same numbers, and a sampler inside a loop body draws
// afresh each time round because the call itself runs again.
// uniform draws [0, 1), but an inverse CDF with unbounded support diverges
// at an endpoint — one draw of exactly 0 in 2^24 would return an infinity
// and quietly poison a Monte Carlo estimate. These shift the interval
// inside both ends so every such distribution is total.
template <typename T> constexpr T open_scale() {
    return T(1) - (sizeof(T) > 4 ? T(0x1p-53) : T(0x1p-24));
}
template <typename T> constexpr T open_shift() {
    return (sizeof(T) > 4 ? T(0x1p-53) : T(0x1p-24)) * T(0.5);
}

inline std::uint64_t claim_stream() {
    return stream_key(seed_slot().load(std::memory_order_relaxed),
                      draw_slot().fetch_add(1, std::memory_order_relaxed));
}

} // namespace tensor::detail
