// Bose-Einstein condensation of a gluon gas — the thesis numerics
// (Numerics.nb), with the full 2→2 collision integral.
//
// Energy conservation E_i + E_j = E_k + E_l fixes the fourth index of every
// collision kernel, so each collision term is ONE index contraction over a
// dense kernel. The kinematic mask lives in the kernel fills, zeroing every
// term the notebook's sparse matrix had no entry for — so what an off-grid
// read of f resolves to cannot affect the result. Validated against the
// notebook's own evolution to ~1e-12 at t = 60 (README.md).

#include <Tensor/Math.h>
#include <Tensor/Tensor.h>

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

using namespace tensor;
using tensor::indices::i, tensor::indices::j, tensor::indices::k;
using tensor::indices::operator""_c;
using tensor::indices::zero;
using namespace tensor::math;

using f64 = double;
using idx = size_t;

// ── the run: the notebook's setupNumerics, weak screening ───────────────────
constexpr idx M = 32;            // grid cells
constexpr f64 Q = 1.0;           // momentum scale
constexpr f64 m = 0.2 * Q;       // thermal mass
constexpr f64 mu = 1.4 * Q;      // screening (exchange) mass
constexpr f64 Lambda = 4.7 * Q;  // UV end of the grid
constexpr f64 alpha = 0.3;       // coupling
constexpr f64 chi = 2.0;         // initial amplitude: overpopulated → condenses
constexpr f64 sigma = 0.055 * Q; // width of the initial step
constexpr f64 n0 = 1e-5;         // initial condensate density
constexpr f64 t_max = 60.0;

constexpr f64 dE = (Lambda - m) / f64(M);
constexpr f64 pi = 3.14159265358979323846;
constexpr f64 coupling2 = 72.0 * (4.0 * pi * alpha) * (4.0 * pi * alpha);

using Grid = Tensor<f64, M>;

// ∫dk κ² x = ∫dE E κ x on this grid, midpoint rule.
template <TensorExpr X> f64 integrate(const X &x) {
    return eval(fold<ops::Add>(x)) * dE;
}

// ── matrix elements, weakly screened ────────────────────────────────────────

// −π ∫ s²/√(a s² + b s + c) over [s−, s+]: closed forms on either side of
// a = 0, direct quadrature in the |a| ≤ 1e-3 band (the notebook's NIntegrate).
f64 s_integral(f64 Ei, f64 Ej, f64 Ek, f64 El, f64 sp, f64 sm, f64 mu0) {
    const f64 de2 = (Ei - Ek) * (Ei - Ek);
    const f64 a = de2 - mu0 * mu0;
    const f64 b = -4.0 * m * m * de2 - mu0 * mu0 * mu0 * mu0 +
                  2.0 * mu0 * mu0 * (Ei * El + Ej * Ek + 2.0 * m * m);
    const f64 c = mu0 * mu0 * (mu0 * mu0 - 4.0 * m * m) * (Ek + El) * (Ek + El);
    const f64 d2 = b * b - 4.0 * a * c;
    constexpr f64 tol = 1e-3;
    if (a < -tol) {
        auto I = [&](f64 s) {
            return (3.0 * b - 2.0 * a * s) * Sqrt(a * s * s + b * s + c) +
                   ((3.0 * b * b - 4.0 * a * c) / (2.0 * Sqrt(-a))) *
                       Asin((b + 2.0 * a * s) / Sqrt(d2));
        };
        return pi / (4.0 * a * a) * (I(sp) - I(sm));
    }
    if (a > tol) {
        auto I = [&](f64 s) {
            const f64 t = b + 2.0 * a * s;
            return (3.0 * b - 2.0 * a * s) * Sqrt(a * s * s + b * s + c) -
                   ((3.0 * b * b - 4.0 * a * c) / (2.0 * Sqrt(a))) *
                       (t < 0 ? -1.0 : 1.0) * Acosh(Abs(t) / Sqrt(d2));
        };
        return pi / (4.0 * a * a) * (I(sp) - I(sm));
    }
    auto g = [&](f64 s) {
        return s * s / Sqrt(Fmax(a * s * s + b * s + c, 1e-300));
    };
    constexpr int N = 256; // composite Simpson
    const f64 h = (sp - sm) / N;
    f64 acc = g(sm) + g(sp);
    for (int q = 1; q < N; ++q)
        acc += g(sm + q * h) * (q % 2 ? 4.0 : 2.0);
    return -pi * acc * h / 3.0;
}

// The 2→2 gas element w(i,j,k,l; μ): regulated minus unregulated.
f64 gas_element(f64 Ei, f64 Ej, f64 Ek, f64 El, f64 ki, f64 kj, f64 kk,
                f64 kl) {
    const bool lo = Ei * Ej < Ek * El;
    const f64 EE0 = lo ? Ei * Ej : Ek * El;
    const f64 kk0 = lo ? ki * kj : kk * kl;
    const f64 sp = 2.0 * (EE0 + m * m + kk0);
    const f64 sm = 2.0 * (EE0 + m * m - kk0);
    return coupling2 / (mu * mu) *
           (s_integral(Ei, Ej, Ek, El, sp, sm, mu) -
            s_integral(Ei, Ej, Ek, El, sp, sm, 0.0));
}

// ℳw(s,t,u): one leg in the condensate.
f64 condensate_element(f64 s, f64 t, f64 u) {
    return coupling2 * s * s *
           (1.0 / (t * (t - mu * mu)) + 1.0 / (u * (u - mu * mu)));
}

// ── the model: grids and kernels, built once ────────────────────────────────

struct Model {
    Grid E, K;                // energy and momentum grids
    Grid hw, hc;              // collision prefactors ℏw, ℏc
    Tensor<f64, M, M, M> W;   // 2→2 kernel,          l = i+j−k
    Tensor<f64, M, M> M2, M3; // condensate kernels,  l = i−k−1 and i+j+1
};

bool on_grid(std::ptrdiff_t l) { return 0 <= l && l < std::ptrdiff_t(M); }

// The notebook's trapezoid weight for one summed leg: ½ at either end.
f64 edge_w(std::ptrdiff_t q) {
    return q == 0 || q + 1 == std::ptrdiff_t(M) ? 0.5 : 1.0;
}

Model build_model() {
    Grid E([](idx a) { return m + f64(a + 1) * dE; });
    Grid K = eval(Sqrt(E * E - m * m));
    constexpr f64 two_pi_4 = 16.0 * pi * pi * pi * pi; // (2π)⁴
    Grid hw = eval((dE * dE / (32.0 * two_pi_4)) / (K * E));
    Grid hc = eval((dE / (64.0 * pi * m)) / (E * K));

    // The kinematic masks — l on the grid, i ≠ k, i ≠ l — and the trapezoid
    // weights are zeros and factors in the fills, exactly as the notebook's
    // SparseArray carries them, so the contractions below need no special
    // cases. Signed coordinates, because the derived fourth index can fall
    // off the grid on either side.
    using ix_t = std::ptrdiff_t;
    Tensor<f64, M, M, M> W([&](ix_t a, ix_t b, ix_t c) -> f64 {
        const ix_t l = a + b - c;
        if (!on_grid(l) || a == c || a == l)
            return 0.0;
        return edge_w(b) * edge_w(c) * edge_w(l) *
               gas_element(E[a], E[b], E[c], E[l], K[a], K[b], K[c], K[l]);
    });
    Tensor<f64, M, M> M2([&](ix_t a, ix_t c) -> f64 {
        const ix_t l = a - c - 1;
        if (!on_grid(l) || a == c || a == l)
            return 0.0;
        return edge_w(c) * edge_w(l) *
               condensate_element(2 * m * (m + E[a]), 2 * m * (E[c] - E[a]),
                                  2 * m * (m - E[c]));
    });
    Tensor<f64, M, M> M3([&](ix_t a, ix_t b) -> f64 {
        const ix_t l = a + b + 1;
        if (!on_grid(l) || a == l)
            return 0.0;
        return edge_w(b) * edge_w(l) *
               condensate_element(2 * m * (m + E[l]), 2 * m * (m - E[a]),
                                  2 * m * (E[a] - E[l]));
    });

    return {std::move(E),  std::move(K),  std::move(hw), std::move(hc),
            std::move(W),  std::move(M2), std::move(M3)};
}

// ── the time derivative: one contraction per collision term ─────────────────
//
//   gas_i  = ℏw_i Σ_jk W_ijk (b_i b_j f_k f_l − f_i f_j b_k b_l),  b = 1 + f
//   cond_i = ℏc_i Σ_j [ M2_ij (b_i f_j f_l − …) + 2 M3_ij (b_i b_j f_l − …) ]
//
// Gain − loss is one node, so each kernel streams once per evaluation; the
// ±1_c offsets are the notebook's 1-based grid seen from zero; zero(…)
// says a channel that leaves the grid does not exist.

struct Deriv {
    Grid df;
    f64 dnc;
};

Deriv deriv(const Model &mdl, const Grid &f, f64 nc) {
    const auto &[E, K, hw, hc, W, M2, M3] = mdl;
    const Grid b = eval(1.0 + f);

    const Grid gas =
        eval(fold<j, k>(W[i, j, k] * hw[i] *
                        (b[i] * b[j] * f[k] * f[zero(i + j - k)] -
                         f[i] * f[j] * b[k] * b[zero(i + j - k)])));

    const Grid cond = eval(fold<j>(
        hc[i] *
        (M2[i, j] * (b[i] * f[j] * f[zero(i - j - 1_c)] -
                     f[i] * b[j] * b[zero(i - j - 1_c)]) +
         2.0 * M3[i, j] * (b[i] * b[j] * f[zero(i + j + 1_c)] -
                           f[i] * f[j] * b[zero(i + j + 1_c)]))));

    // The condensate back-reacts on the gas; what the gas loses through the
    // condensate channels is what nc receives.
    return {eval(gas + nc * cond),
            -nc / (2.0 * pi * pi) * integrate(E * K * cond)};
}

// ── the notebook's RKF23: an embedded 2(3) pair, error over both f and nc ───

f64 rkf23(const Model &mdl, Grid &f, f64 &nc, f64 &dt, f64 tol, int accepted) {
    f64 t = 0.0;
    for (int done = 0; done < accepted;) {
        auto [y1, x1] = deriv(mdl, f, nc);
        auto [y2, x2] = deriv(mdl, eval(f + dt * y1), nc + dt * x1);
        auto [y3, x3] = deriv(mdl, eval(f + (0.25 * dt) * (y1 + y2)),
                              nc + 0.25 * dt * (x1 + x2));
        const f64 err_f = eval(fold<ops::Max>(Abs(y1 + y2 - 2.0 * y3)));
        const f64 err = dt / 3.0 * Fmax(err_f, Abs(x1 + x2 - 2 * x3));
        if (err <= tol) {
            ++done;
            t += dt;
            f = eval(f + (dt / 6.0) * (y1 + y2 + 4.0 * y3));
            nc += dt / 6.0 * (x1 + x2 + 4 * x3);
            dt *=
                Fmin(2.0, Fmax(1.0, 0.9 * (err > 0.0 ? Cbrt(tol / err) : 2.0)));
        } else {
            dt *= 0.9 * Cbrt(tol / err);
        }
    }
    return t;
}

// ── diagnostics ─────────────────────────────────────────────────────────────

f64 density(const Model &mdl, const Grid &f) {
    return integrate(mdl.E * mdl.K * f) / (2 * pi * pi);
}
f64 entropy(const Model &mdl, const Grid &f) {
    const auto fp = Fmax(f, 1e-300); // log(0) guard
    return integrate(mdl.E * mdl.K *
                     ((1.0 + fp) * Log(1.0 + fp) - fp * Log(fp))) /
           (2 * pi * pi);
}

// Least-squares line through B = ln((1+f)/f) over the populated cells:
// straight ⟺ Bose-Einstein, slope 1/T, intercept −μ/T.
struct BoseFit {
    f64 T, mu_chem, r2;
    int cells;
};

BoseFit fit_bose_transform(const Model &mdl, const Grid &f) {
    const Grid B = eval(Log((1.0 + f) / f));
    auto populated = [&](idx q) {
        return f.data()[q] > 1e-6 && std::isfinite(B.data()[q]);
    };
    f64 sx = 0, sy = 0, sxx = 0, sxy = 0;
    int n = 0;
    for (idx q = 0; q < M; ++q)
        if (populated(q)) {
            const f64 x = mdl.K.data()[q], y = B.data()[q];
            sx += x, sy += y, sxx += x * x, sxy += x * y, ++n;
        }
    const f64 slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    const f64 intercept = (sy - slope * sx) / n;
    f64 ss_res = 0, ss_tot = 0;
    for (idx q = 0; q < M; ++q)
        if (populated(q)) {
            const f64 r = B.data()[q] - (intercept + slope * mdl.K.data()[q]);
            const f64 d = B.data()[q] - sy / n;
            ss_res += r * r, ss_tot += d * d;
        }
    return {1.0 / slope, -intercept / slope, 1.0 - ss_res / ss_tot, n};
}

// ── the run ─────────────────────────────────────────────────────────────────

#ifndef BOLTZMANN_NO_MAIN // the validation harness includes this file
int main() {
    use_threads(4);
    const Model mdl = build_model();
    Grid f = eval(chi / (Exp((mdl.K - Q) / sigma) + 1.0));
    f64 nc = n0, t = 0.0;
    const f64 total0 = density(mdl, f) + nc;

    std::cout << "M = " << M << " cells over E ∈ [" << m + dE << ", " << Lambda
              << "], χ = " << chi << ", α = " << alpha << ", μ = " << mu
              << " (weak screening)\n\n"
              << "       t      n_gas        n_c      total    entropy\n"
              << std::fixed << std::setprecision(6);

    f64 next_print = 0.0;
    while (true) {
        if (t >= next_print) {
            const f64 n_gas = density(mdl, f);
            std::cout << std::setw(8) << t << std::setw(11) << n_gas
                      << std::setw(11) << nc << std::setw(11) << n_gas + nc
                      << std::setw(11) << entropy(mdl, f) << '\n';
            next_print += t_max / 15.0;
        }
        if (t >= t_max)
            break;
        f64 dt = 1e-3; // the notebook's runEvolution: dt resets each chunk
        t += rkf23(mdl, f, nc, dt, 1e-5, 5);
    }

    const BoseFit fit = fit_bose_transform(mdl, f);
    std::cout << std::setprecision(5) << "\nBose transform over " << fit.cells
              << " populated cells:  R² = " << fit.r2
              << "  (1 = exactly Bose-Einstein)\n"
              << std::setprecision(4) << "  T = " << fit.T
              << ",   μ_chem = " << fit.mu_chem << '\n'
              << "\nnumber balance over the run: " << std::scientific
              << std::setprecision(2)
              << (density(mdl, f) + nc - total0) / total0
              << " (relative; the 2→2 discretization conserves only up to "
                 "the trapezoid rule)\n";
}
#endif
