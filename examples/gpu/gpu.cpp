// Stencil expressions on the GPU — decorated reads composed into
// eval(dev, expr).
//
// Build with the GPU path enabled (this example is skipped otherwise):
//   make GPU=1        then:  ./build-gpu/examples/gpu/gpu
// Runtime needs a Vulkan ≥ 1.2 driver (macOS: brew install molten-vk) and
// slangc on the PATH. GPUD_LOG=1 shows why a device would not open;
// GPUD_BACKEND=vulkan|mock insists on a backend instead of auto-picking.

#include <Tensor/Gpu/Eval.h>
#include <Tensor/Math.h>
#include <Tensor/Tensor.h>

#include <gpud/Auto.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <utility>

using namespace tensor;
using namespace tensor::indices;
using namespace tensor::math;

int main() {
    // The application opens and owns the device; tensor only borrows it
    // for the duration of each eval. nullptr = no usable backend.
    auto dev = gpud::open_default();
    if (!dev) {
        std::cout << "no usable GPU device (set GPUD_LOG=1 to see why) — "
                     "the CPU spelling of everything here is eval(expr)\n";
        return EXIT_SUCCESS;
    }

    // Heat diffusion on a doubly-periodic plate,
    //   u ← u + r·(u_north + u_south + u_west + u_east − 4·u),
    // the classic 5-point Laplacian stencil. A Fourier mode
    // sin(kx·i)·sin(ky·j) is an exact eigenvector of that discrete
    // operator: every step scales it by exactly
    //   λ = 1 − 4r·(sin²(π·mx/N) + sin²(π·my/N)),
    // so the GPU result can be checked against pencil and paper.
    constexpr size_t N = 256;
    constexpr float r = 0.25f; // α·dt/dx², at the 2D stability limit
    constexpr int mx = 4, my = 3, steps = 200;

    constexpr float tau = 2 * std::numbers::pi_v<float>;
    constexpr float kx = tau * mx / N, ky = tau * my / N;

    // The initial mode as an index fill; w is the CPU twin of the same
    // diffusion.
    auto mode = [](size_t i, size_t j) { return Sin(kx * i) * Sin(ky * j); };
    Tensor<float, N, N> u(mode), w(mode);

    // A stencil is decorated reads composed through the ordinary
    // operators: offsets add to the reading index, wrap(…) makes the
    // plate periodic. A reusable stencil is just a lambda.
    auto lap = [](const auto &f) {
        return f[wrap(i - 1_c), j] + f[wrap(i + 1_c), j] +
               f[i, wrap(j - 1_c)] + f[i, wrap(j + 1_c)] - 4.0f * f[i, j];
    };

    // One expression type = one kernel: compiled on the first eval and
    // memoized on the device, then only re-dispatched — 200 launches, one
    // compile. The CPU line is the same spelling minus the device.
    for (int s = 0; s < steps; ++s) {
        u = eval(*dev, u[i, j] + r * lap(u)); // GPU
        w = eval(w[i, j] + r * lap(w));       // CPU
    }

    // The pencil-and-paper column: the mode's exact per-step decay factor
    // raised to the number of steps.
    auto sq = [](float x) { return x * x; };
    constexpr float pi = std::numbers::pi_v<float>;
    const float lam =
        1.0f - 4.0f * r * (sq(Sin(pi * mx / N)) + sq(Sin(pi * my / N)));
    const float decay = Pow(lam, float(steps));

    std::cout << N << "x" << N << " periodic plate on \"" << dev->dialect()
              << "\", " << steps << " steps at r = " << r << '\n'
              << std::fixed << std::setprecision(6) << "mode (" << mx << ", "
              << my << ") decays by λ = " << lam << " per step; λ^" << steps
              << " = " << decay << "\n\n"
              << std::setprecision(4)
              << "    i    j      u(0)    u(gpu)  analytic\n";
    for (auto [i, j] :
         {std::pair{8, 8}, {16, 24}, {48, 12}, {100, 50}, {200, 150}}) {
        const float u0 = mode(i, j);
        std::cout << std::setw(5) << i << std::setw(5) << j << std::setw(10)
                  << u0 << std::setw(10) << u[i, j] << std::setw(10)
                  << decay * u0 << '\n';
    }

    // GPU and CPU run the same operations from the same tree; where the
    // two compilers contract multiply-adds differently, the last bits can
    // drift.
    float dev_max = 0.0f;
    for (size_t i = 0; i < u.size(); ++i)
        dev_max = Fmax(dev_max, Abs(u.data()[i] - w.data()[i]));
    std::cout << "\nmax |gpu − cpu| after " << steps
              << " steps: " << std::scientific << std::setprecision(1)
              << dev_max << '\n';

    // The kernel each step dispatched — generated from the expression's
    // type at compile time. The wrap arithmetic in the subscripts is the
    // stencil's index algebra, one spelling shared by CPU formulas and
    // Slang.
    using Step = decltype(u[i, j] + r * lap(u));
    std::cout << "\nthe kernel behind each step:\n\n" << gpu_source<Step>();
}
