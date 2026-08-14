// A two-layer perceptron fit by hand-written backprop — the einsum-style
// ML workload: with a batch axis every backprop quantity is ONE index
// contraction (the dW1 line fuses four factors with no materialized
// intermediate), activations are Tanh, the loss is a fold,
// and momentum updates are plain elementwise evals. The bias rides the
// augmented constant-1 feature, and tanh is the fold's epilogue —
// a layer is ONE eval, one pass.
//
// Direct compile:
//   g++-16 -std=c++26 -freflection -O3 -I../../include mlp.cpp && ./a.out

#include <Tensor/Gen.h>
#include <Tensor/Math.h>
#include <Tensor/Tensor.h>

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>

using namespace tensor;
using namespace tensor::math;
using tensor::indices::i, tensor::indices::j, tensor::indices::k;

using f64 = double;
using idx = size_t;

// ── the task: single-index regression y = sin(π c·x), 63 features + bias ────
constexpr idx B = 256;    // batch (full-batch descent)
constexpr idx IN = 64;    // 63 features + the constant-1 bias feature
constexpr idx HID = 64;   // tanh units
constexpr f64 lr = 0.03;  // step size
constexpr f64 beta = 0.9; // heavy-ball momentum
constexpr f64 pi = 3.14159265358979323846;

using Batch = Tensor<f64, B, IN>;
using W1t = Tensor<f64, IN, HID>;
using W2t = Tensor<f64, HID>;
using Vec = Tensor<f64, B>;

struct Mlp {
    W1t W1, vW1;
    W2t w2, vw2;
};

Mlp init_mlp() {
    Mlp m;
    // Weights from the sampler; rng::Seed() above makes the run reproducible.
    m.W1 = eval(0.15 * rng::Normal<f64, IN, HID>());
    m.w2 = eval(0.2 * rng::Normal<f64, HID>());
    // The velocities are READ on the first step, so they have to be
    // written first: a default-constructed Tensor is uninitialized.
    m.vW1 = eval(gen::Fill<IN, HID>(0.0));
    m.vw2 = eval(gen::Fill<HID>(0.0));
    return m;
}

struct Data {
    Batch X;
    Vec Y;
};

Data make_data() {
    // Features uniform on [-1, 1), then the last column set to the
    // constant-1 bias feature the augmented layout expects.
    Batch X = eval(2.0 * rng::Uniform<f64, B, IN>() - 1.0);
    for (idx b = 0; b < B; ++b)
        X[b, IN - 1] = 1.0;

    // fixed direction c, scaled so u = c·x has std ≈ 0.5: the target
    // sin(π u) is genuinely nonlinear over the sampled range
    const auto cs = eval(2.0 * rng::Uniform<f64, IN>() - 1.0);
    f64 c[IN]{}, sum2 = 0.0;
    for (idx q = 0; q + 1 < IN; ++q) {
        c[q] = cs[q];
        sum2 += c[q] * c[q];
    }
    const f64 s = std::sqrt(0.75 / sum2);
    Vec Y([&](idx b) {
        f64 u = 0.0;
        for (idx q = 0; q + 1 < IN; ++q)
            u += s * c[q] * X[b, q];
        return std::sin(pi * u);
    });
    return {std::move(X), std::move(Y)};
}

// One full-batch step: forward, fused-contraction gradients, momentum.
f64 train_step(Mlp &m, const Batch &X, const Vec &Y) {
    // A layer is ONE eval: the activation is the fold's epilogue.
    const auto H = eval(Tanh(fold<k>(X[i, k] * m.W1[k, j]))); // [B, HID]
    const auto p = eval(fold<j>(H[i, j] * m.w2[j]));          // [B]
    const auto e = eval(p - Y);

    const auto dw2 = eval(fold<i>(H[i, j] * e[i])); // [HID]
    const auto dW1 = eval(fold<i>(X[i, k] * e[i] * m.w2[j] *
                                      (1.0 - H[i, j] * H[i, j]))); // [IN,HID]

    const f64 s = 2.0 * lr / f64(B);
    m.vW1 = eval(beta * m.vW1 + dW1);
    m.vw2 = eval(beta * m.vw2 + dw2);
    m.W1 = eval(m.W1 - s * m.vW1);
    m.w2 = eval(m.w2 - s * m.vw2);
    return eval(fold<ops::Add>(e * e)) / f64(B);
}

f64 r_squared(const Mlp &m, const Batch &X, const Vec &Y) {
    const auto H = eval(Tanh(fold<k>(X[i, k] * m.W1[k, j])));
    const auto p = eval(fold<j>(H[i, j] * m.w2[j]));
    const f64 ybar = eval(fold<ops::Add>(Y)) / f64(B);
    const f64 ssr = eval(fold<ops::Add>((p - Y) * (p - Y)));
    const f64 sst = eval(fold<ops::Add>((Y - ybar) * (Y - ybar)));
    return 1.0 - ssr / sst;
}

#ifndef MLP_NO_MAIN
int main() {
    rng::Seed(20260814); // sampling is random by default; pin it so runs match
    const Data d = make_data();
    Mlp m = init_mlp();
    std::cout << std::fixed;
    for (idx epoch = 0; epoch <= 1500; ++epoch) {
        const f64 mse = train_step(m, d.X, d.Y);
        if (epoch % 250 == 0)
            std::cout << "epoch " << std::setw(5) << epoch << "   mse "
                      << std::setprecision(5) << mse << "\n";
    }
    std::cout << "\nfit over " << B
              << " samples:  R² = " << std::setprecision(4)
              << r_squared(m, d.X, d.Y) << "  (target sin(π c·x), " << IN - 1
              << " features + bias)\n";
}
#endif
