// Integrating an ensemble of pendulums — explicit time-stepping built from
// tensor expressions.
//
// Build with CMake (the `pendulum` target) or directly:
//   g++-16 -std=c++26 -freflection -I../../include pendulum.cpp

#include <Tensor/Gpu.h> // opt-in: lets us print the generated GPU kernel
#include <Tensor/Math.h>
#include <Tensor/Tensor.h>

#include <iomanip>
#include <iostream>

using namespace tensor;

int main() {
    // Five pendulums released from rest, from a small angle up to a wide
    // swing. The count is fixed at compile time, part of the tensor type.
    constexpr size_t N = 5;
    const float release[N] = {0.1f, 0.5f, 1.0f, 2.0f, 3.0f};

    Tensor<float, N> theta([&](size_t i) { return release[i]; });
    Tensor<float, N> omega([](size_t) { return 0.0f; });

    // Energy per unit mass (natural units): ½·omega² + (1 − cos theta).
    // Used only to measure how well the integrator conserves it.
    auto energy = [&] {
        return eval(0.5f * omega * omega + (1.0f - math::Cos(theta)));
    };
    auto energy0 = energy();

    // Symplectic (semi-implicit) Euler: kick the velocity with the force
    // −sin(theta) at the current angle, then drift the angle with the new
    // velocity. Each line is one fused elementwise pass; eval() is the only
    // way to evaluate, so the integrator is simply a sequence of evals.
    constexpr float dt = 0.02f;
    constexpr int steps = 314; // ≈ one small-angle period (2·pi)
    for (int s = 0; s < steps; ++s) {
        omega = eval(omega - math::Sin(theta) * dt);
        theta = eval(theta + omega * dt);
    }

    auto energy1 = energy();

    // Larger amplitudes swing slower — the nonlinear period grows with
    // amplitude — so after one small-angle period they lag in phase, a
    // signature the linear approximation misses. The relative energy drift
    // stays small and bounded: the symplectic integrator's payoff.
    std::cout << std::fixed << std::setprecision(4)
              << "  theta0    theta(T)     dE/E\n";
    for (size_t i = 0; i < N; ++i)
        std::cout << std::setw(8) << release[i] << std::setw(11) << (theta[i])
                  << std::setw(11) << (energy1[i] - energy0[i]) / energy0[i]
                  << '\n';

    // An update is also a compile-time value; the vocabulary op names
    // itself in the formula via its annotation:
    using Kick = decltype(omega - math::Sin(theta) * dt);
    std::cout << "\nvelocity kick: " << formula<Kick>() << '\n';

    // sin is a Slang intrinsic, so even the nonlinear kick lowers to a
    // complete compute shader:
    std::cout << '\n' << gpu_source<Kick>();
}
