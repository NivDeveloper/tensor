// Test-particle Boltzmann relaxation in 3-D: two discs collide off-axis and
// thermalize. Per step — stream, measure n/p/E per cell, histogram the
// momenta, relax toward the Maxwellian by the RTA, resample, then shift and
// scale so the cell's momentum and energy come out unchanged.
//
//   g++-16 -std=c++26 -freflection -O3 -I../../include bgk.cpp && ./a.out

#include <Tensor/Gen.h>
#include <Tensor/Math.h>
#include <Tensor/Tensor.h>

#ifdef TENSOR_GPU_ENABLED
#include <Tensor/Gpu.h>
#ifndef BGK_NO_MAIN // only main opens a device; the benches bring their own
#include <gpud/Auto.h>
#endif
#endif

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <utility>

using namespace tensor;
using namespace tensor::math;
using tensor::indices::clamp;
using tensor::indices::i, tensor::indices::j, tensor::indices::m,
    tensor::indices::n;
using tensor::indices::operator""_c;

using f32 = float;
using idx = size_t;

constexpr idx N = 8192;       // particles
constexpr idx C = 4;          // cells per axis
constexpr idx CC = C * C * C; // cells in the grid
constexpr idx B = 24;         // momentum bins per cell, per component
constexpr f32 vmax = 2.5f;
constexpr f32 dv = 2.0f * vmax / f32(B);
constexpr f32 dt = 0.004f;
constexpr f32 tau = 0.01f;
constexpr f32 tpi = 6.283185307179586f;
constexpr int steps = 1000;

using Vecs = Tensor<f32, N, 3>;
using Cells = Tensor<f32, N>; // each particle's cell, as one number
using Grid = Tensor<f32, CC>;
using GridV = Tensor<f32, CC, 3>;

#ifdef TENSOR_GPU_ENABLED
gpud::Device *device = nullptr;
#endif

// Which eval every expression goes through, chosen at COMPILE time: a runtime
// `if (device)` would instantiate both paths for all twelve and double the
// code.
auto go(const auto &e) {
#if defined(TENSOR_GPU_ENABLED) && defined(BGK_RUNTIME_DEVICE)
    if (device)
        return eval(*device, e);
    return eval(e);
#elif defined(TENSOR_GPU_ENABLED)
    return eval(*device, e);
#else
    return eval(e);
#endif
}

// The conserved quantities per cell, and the equilibrium they imply.
struct Cell {
    Grid pop, inv, E, T, mu;
    GridV p;
};

// Each particle's cell as ONE number: bins per axis, clamped per AXIS, then
// combined row-major. The clamp must precede the combine — an axis reaching C
// aliases into the NEXT axis's slot, a valid id no write policy can catch.
Cells cells(const Vecs &pos) {
    auto a = Fmin(Fmax(bins<C>(pos, -0.5f, 0.5f), 0.0f), f32(C - 1));
    return go((a[i, 0_c] * f32(C) + a[i, 1_c]) * f32(C) + a[i, 2_c]);
}

Cell measure(const auto &at, const Vecs &mom) {
    auto sq = go(fold<1>(mom * mom));
    auto count = go(scatter<i>(at, 1.0f));
    auto inv = go(1.0f / Fmax(count, 1.0f));
    auto p = go(scatter<i>(at, mom[i, n]));
    auto E = go(scatter<i>(at, sq[i]));
    auto p2 = go(fold<1>(p * p));

    auto T = go(Fmax((E - p2 * inv) * inv * (1.0f / 3.0f), 1e-9f));
    auto mu = go(T * Log(Fmax(count, 1.0f) * f32(CC) / Pow(tpi * T, 1.5f)));

    return {std::move(count), std::move(inv), std::move(E),
            std::move(T),     std::move(mu),  std::move(p)};
}

Vecs resample(const auto &at, const Cell &c, const Vecs &mom,
              const Tensor<f32, B> &centre, f32 alpha) {

    // Momenta per cell
    auto hist = go(scatter<i>(at, clamp(bins<B>(mom[i, n], -vmax, vmax)), 1u));

    // The Maxwellian for these cell parameters, centred on the drift.
    // The cell index leads: free indices take first-appearance order.
    auto off = go(c.p[j, n] * -c.inv[j] + centre[m]);
    auto heat = go(Exp(off[j, n, m] * off[j, n, m] * -0.5f / c.T[j]));
    auto nrm = go(fold<2>(heat));

    // f(t+dt) = f_eq + (f - f_eq)·exp(-dt/tau)
    auto relaxed = go((1.0f - alpha) * c.pop[j] * heat[j, n, m] / nrm[j, n] +
                      hist[j, m, n] * alpha);

    // The CDF is the running sum of the relaxed histogram along its bins.
    auto cdf = go(scan<ops::Add, m>(relaxed[j, n, m]) * c.inv[j]);

    auto u1 = go(rng::Uniform<f32, N, 3>());
    auto hit = go(fold<m>(1.0f * (u1[i, n] > cdf[at, n, m])));
    return go(-vmax + (hit + rng::Uniform<f32, N, 3>()) * dv);
}

void step(Vecs &Pos, Vecs &Mom, const Tensor<f32, B> &centre, f32 alpha) {
    auto moved = go(Pos + Mom * dt);
    // periodic boundary conditions
    Pos = go(moved - Floor(moved + 0.5f));

    auto cid = cells(Pos);
    // The cell each particle deposits into and reads back from: built once,
    // and the same object serves both, since a destination is also a subscript.
    auto at = clamp<CC>(cid[i]);
    auto c = measure(at, Mom);
    auto q = resample(at, c, Mom, centre, alpha);

    // Shift and scale q -> b·q + a, fixed by the cell's own totals:
    // a = (p - b·Q)/n, b = sqrt((E - |p|²/n) / (Q2 - |Q|²/n)).
    auto qsq = go(fold<1>(q * q));
    auto Q = go(scatter<i>(at, q[i, n]));
    auto Q2 = go(scatter<i>(at, qsq[i]));
    auto p2 = go(fold<1>(c.p * c.p));
    auto q2 = go(fold<1>(Q * Q));
    auto b =
        go(Sqrt(Fmax(c.E - p2 * c.inv, 0.0f) / Fmax(Q2 - q2 * c.inv, 1e-9f)));
    auto shift = go((c.p[j, n] - b[j] * Q[j, n]) * c.inv[j]);

    // Under two particles there is nothing to resample from.
    auto live = go(1.0f * (c.pop[at] >= 2.0f));
    auto next = go(b[at] * q[i, n] + shift[at, n]);
    Mom = go(Mom[i, n] + live[i] * (next[i, n] - Mom[i, n]));
}

// The bin centres the histogram and the Maxwellian share.
Tensor<f32, B> bin_centres() {
    auto axis = eval(gen::Iota<B>(0.0f));
    return eval(-vmax + (axis + 0.5f) * dv);
}

// Two discs of particles flying at each other, off-axis. Draws, so the
// caller seeds first if it wants a particular run.
struct State {
    Vecs Pos, Mom;
};

State initial_state() {
    auto beam = eval(1.0f * (gen::Iota<N>(0.0f) < f32(N / 2)));
    auto sign = eval(2.0f * beam - 1.0f);
    auto rad = eval(0.15f * Sqrt(rng::Uniform<f32, N>())); // uniform by area
    auto ang = eval(tpi * rng::Uniform<f32, N>());

    auto x = eval(-0.25f * sign + 0.002f * rng::Normal<f32, N>());
    auto y = eval(0.05f * (1.0f - beam) + rad * Cos(ang));
    auto z = eval(rad * Sin(ang));
    auto vx = eval(0.80f * sign + 0.15f * rng::Normal<f32, N>());
    auto vy = eval(0.15f * rng::Normal<f32, N>());
    auto vz = eval(0.15f * rng::Normal<f32, N>());

    return {Vecs([&](idx q, idx d) {
                return d == 0 ? x[q] : d == 1 ? y[q] : z[q];
            }),
            Vecs([&](idx q, idx d) {
                return d == 0 ? vx[q] : d == 1 ? vy[q] : vz[q];
            })};
}

#ifndef BGK_NO_MAIN // bench/Bgk_bench.cpp includes this file

int main() {
#ifdef TENSOR_GPU_ENABLED
    // First, so every tensor is destroyed before the device that made it.
    auto owned = gpud::open_default();
    device = owned.get();
    if (!device) { // this build emits only device calls
        std::cout << "no device; build without TENSOR_GPU_ENABLED for the "
                     "CPU path\n";
        return 1;
    }
    std::cout << "running on the GPU\n";
#endif
    use_threads(8);
    rng::Seed(20260814);

    auto centre = bin_centres();
    auto [Pos, Mom] = initial_state();

    f32 alpha = Exp(-dt / tau);
    f32 p0 = eval(fold<0>(Mom))[0];
    f32 e0 = eval(fold(Mom * Mom));

    for (int s = 0; s < steps; ++s)
        step(Pos, Mom, centre, alpha);

    auto cid = cells(Pos);
    auto c = measure(clamp<CC>(cid[i]), Mom);
    f32 tot = eval(fold(c.pop));
    f32 p1 = eval(fold<0>(Mom))[0], e1 = eval(fold(Mom * Mom));

    // Per component in ONE pass, no cancellation-prone E[x²]-E[x]². Welford's
    // is the SAMPLE variance, so the population spread is (n-1)/n of it.
    auto spread = eval(fold<ops::Welford, 0>(Mom));

    std::cout << std::fixed << std::setprecision(4) << "T     "
              << eval(fold(c.pop * c.T)) / tot << "\nmu    "
              << eval(fold(c.pop * c.mu)) / tot << "\nsigma "
              << std::sqrt(spread[0].var * (f32(N - 1) / f32(N)))
              << "   (equipartition " << std::sqrt(e0 / (3.0f * f32(N))) << ")"
              << std::scientific << std::setprecision(2) << "\ndrift  momentum "
              << (p1 - p0) / f32(N) << "   energy " << (e1 - e0) / e0 << '\n';
}

#endif
