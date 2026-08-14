#pragma once

// The counter-based random core. No reflection, no tensor types, no state —
// plain C++ over 32-bit integers, so the emitted shader runs this same
// arithmetic and a sample is a pure function of (seed, stream, coordinate).
//
// That purity is the whole design: it makes a sample independent of thread
// count, chunk boundaries and device, and it is what lets a rejection
// sampler draw a variable number of times without one cell's draws
// disturbing another's.

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace tensor::detail {

// Philox-4x32-10 (Random123). Deliberately 32-bit throughout: the 64-bit
// spelling of this multiply makes slangc emit OpCapability Int64, which is
// the same builds-then-fails-pipeline-creation cliff that keeps double off
// the GPU. Verified against all three published KAT vectors.
struct U4 {
    std::uint32_t x, y, z, w;
};

constexpr std::uint32_t philox_mulhi(std::uint32_t a, std::uint32_t b) {
    const std::uint32_t a0 = a & 0xffffu, a1 = a >> 16;
    const std::uint32_t b0 = b & 0xffffu, b1 = b >> 16;
    const std::uint32_t p00 = a0 * b0, p01 = a0 * b1;
    const std::uint32_t p10 = a1 * b0, p11 = a1 * b1;
    const std::uint32_t mid = (p00 >> 16) + (p01 & 0xffffu) + (p10 & 0xffffu);
    return p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16);
}

constexpr U4 philox(U4 c, std::uint32_t k0, std::uint32_t k1) {
    constexpr std::uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    constexpr std::uint32_t W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
    for (int r = 0; r < 10; ++r) {
        if (r > 0) {
            k0 += W0;
            k1 += W1;
        }
        const std::uint32_t hi0 = philox_mulhi(M0, c.x), lo0 = M0 * c.x;
        const std::uint32_t hi1 = philox_mulhi(M1, c.z), lo1 = M1 * c.z;
        c = {hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0};
    }
    return c;
}

// splitmix64, used to fold a seed and a stream number into one key. Nested
// rather than xored so the two cannot trade off against each other — a bare
// `seed ^ stream` makes (1, 0) and (0, 1) the SAME stream.
constexpr std::uint64_t mix64(std::uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

constexpr std::uint64_t stream_key(std::uint64_t seed, std::uint64_t stream) {
    return mix64(seed ^ mix64(stream));
}

// The bits one cell draws. The key already carries seed and stream, so the
// counter is just the coordinate and the draw index — which is what lets a
// cell take as many draws as it likes without touching any other cell.
constexpr U4 philox_cell(std::uint64_t key, std::uint64_t cell,
                         std::uint32_t draw = 0) {
    return philox({std::uint32_t(cell), std::uint32_t(cell >> 32), draw, 0u},
                  std::uint32_t(key), std::uint32_t(key >> 32));
}

// [0, 1) — uniformly spaced, every value exactly representable, and never
// 1.0, which is what makes log(1 - u) finite for every draw. A float takes
// 24 bits from one word (the spelling the shader runs); a double takes 53
// from two, and only ever runs on the host — the GPU rejects double at the
// element-type gate.
template <typename T>
constexpr T uniform01(std::uint32_t a, std::uint32_t b) {
    if constexpr (sizeof(T) > 4) {
        const std::uint64_t bits = (std::uint64_t(a) << 32) | b;
        return T(bits >> 11) * T(0x1p-53);
    } else
        return T(a >> 8) * T(0x1p-24f);
}

// Box-Muller. u1 is taken from the OPEN end so the log is always finite —
// the reason uniform01 excludes 1.0 rather than 0.
template <typename T> T normal01(U4 r) {
    const T u1 = T(1) - uniform01<T>(r.x, r.y); // (0, 1]
    const T u2 = uniform01<T>(r.z, r.w);        // [0, 1)
    return std::sqrt(T(-2) * std::log(u1)) *
           std::cos(T(6.283185307179586476925286766559) * u2);
}

// One cell's private stream, handed to a sample<f> function. Passed BY
// VALUE and mutated locally, so a sampler stays a pure function of its
// coordinate — which is what lets a rejection loop draw a variable number
// of times without any cell disturbing another. Draws are buffered four at
// a time because that is what one Philox call produces.
struct Rng {
    std::uint64_t key = 0;
    std::uint64_t cell = 0;
    std::uint32_t draw = 0;
    U4 buf{};
    std::uint32_t used = 4; // >= 4 forces a refill on the first call

    constexpr std::uint32_t bits() {
        if (used >= 4) {
            buf = philox_cell(key, cell, draw++);
            used = 0;
        }
        switch (used++) {
        case 0:
            return buf.x;
        case 1:
            return buf.y;
        case 2:
            return buf.z;
        default:
            return buf.w;
        }
    }

    // [0, 1), never 1.0 — the property every log in a sampler relies on.
    constexpr float uniform() { return uniform01<float>(bits(), bits()); }

    float normal() {
        const float u1 = 1.0f - uniform(); // (0, 1]
        const float u2 = uniform();
        return std::sqrt(-2.0f * std::log(u1)) *
               std::cos(6.283185307179586f * u2);
    }
};

} // namespace tensor::detail
